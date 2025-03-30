// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN

public struct QwenQuantization: Codable {
    public var groupSize: Int
    public var bits: Int
}

public struct QwenConfig: Codable {
    public var bosTokenId: Int
    public var eosTokenId: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var maxPositionEmbeddings: Int
    public var maxWindowLayers: Int
    public var numAttentionHeads: Int
    public var numHiddenLayers: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var tieWordEmbeddings: Bool
    public var useSlidingWindow: Bool
    public var vocabSize: Int
    public var quantization: QwenQuantization? = nil

    public func headDim() -> Int {
        self.hiddenSize / self.numAttentionHeads
    }
}

private class Mlp: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(_ cfg: QwenConfig) {
        self._gateProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

private class Attention: Module {
    let cfg: QwenConfig
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ cfg: QwenConfig) {
        self.cfg = cfg
        self.scale = 1.0 / sqrt(Float(cfg.headDim()))
        let headDim = cfg.headDim()
        self._qProj.wrappedValue = Linear(
            cfg.hiddenSize, cfg.numAttentionHeads * headDim, bias: true)
        self._kProj.wrappedValue = Linear(
            cfg.hiddenSize, cfg.numKeyValueHeads * headDim, bias: true)
        self._vProj.wrappedValue = Linear(
            cfg.hiddenSize, cfg.numKeyValueHeads * headDim, bias: true)
        self._oProj.wrappedValue = Linear(
            cfg.numAttentionHeads * headDim, cfg.hiddenSize, bias: false)
        self.rope =
            RoPE(dimensions: cfg.headDim(), traditional: false, base: Float(cfg.ropeTheta))
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let (B, T, H) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = cfg.headDim()
        var queryStates = qProj(x).reshaped(B, T, -1, headDim).transposed(0, 2, 1, 3)
        var keyStates = kProj(x).reshaped(B, T, -1, headDim).transposed(0, 2, 1, 3)
        var valueStates = vProj(x).reshaped(B, T, -1, headDim).transposed(0, 2, 1, 3)
        let offset = cache?.offset ?? 0
        queryStates = rope(queryStates, offset: offset)
        keyStates = rope(keyStates, offset: offset)
        if let cache {
            (keyStates, valueStates) = cache.update(keys: keyStates, values: valueStates)
        }
        // sliding window is not supported here
        var mask = mask
        if let m = mask {
            let maskLen = m.dim(-1)
            if keyStates.dim(2) < maskLen {
                let offset = maskLen - keyStates.dim(2)
                mask = m[0..., offset...]
            }
        }
        let x = MLXFast.scaledDotProductAttention(
            queries: queryStates, keys: keyStates, values: valueStates, scale: self.scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, T, H)
        return oProj(x)
    }
}

private class Layer: Module {
    @ModuleInfo(key: "mlp") var mlp: Mlp
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Attention

    init(_ cfg: QwenConfig) {
        self._mlp.wrappedValue = Mlp(cfg)
        self._inputNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._postAttnNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._selfAttn.wrappedValue = Attention(cfg)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache) -> MLXArray {
        var residual = x
        var x = x
        x = selfAttn(inputNorm(x), mask: mask, cache: cache)
        x = residual + x
        residual = x
        x = mlp(postAttnNorm(x))
        return residual + x
    }
}

public class QwenModel: Module {
    let cfg: QwenConfig
    private let norm: RMSNorm
    private let layers: [Layer]
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    public init(_ cfg: QwenConfig) {
        self.cfg = cfg
        self.layers = (0..<cfg.numHiddenLayers).map { _ in Layer(cfg) }
        self.norm = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize)
    }

    public func callAsFunction(_ x: MLXArray, cache: [KVCache]) -> MLXArray {
        var x = embedTokens(x)
        let mask = cache.first?.createAttentionMask(h: x)
        for (layer, c) in zip(self.layers, cache) {
            x = layer(x, mask: mask, cache: c)
        }
        return embedTokens.asLinear(norm(x))
    }

    public func makeCache(bSize: Int) -> [KVCache] {
        let kvHeads = cfg.numKeyValueHeads
        let cache = (0..<cfg.numHiddenLayers).map { _ in
            KVCacheSimple(headDim: .init(cfg.headDim()), kvHeads: kvHeads)
        }
        return cache
    }
}
