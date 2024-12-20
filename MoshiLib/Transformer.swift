// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN

public enum Norm {
    case layerNorm
    case rmsNorm
}

public enum PositionalEmbedding {
    case none
    case rope
}

public struct TransformerConfig {
    public var dModel: Int
    public var numHeads: Int
    public var numLayers: Int
    public var causal: Bool
    public var normFirst: Bool
    public var biasFF: Bool
    public var biasAttn: Bool
    public var layerScale: Float?
    public var positionalEmbedding: PositionalEmbedding
    public var useConvBias: Bool
    public var gating: Bool
    public var norm: Norm
    public var context: Int
    public var maxPeriod: Int
    public var maxSeqLen: Int
    public var kvRepeat: Int
    public var dimFeedForward: Int
    public var convLayout: Bool
    public var useRotatingKVCache: Bool

    public func headDim() -> Int {
        self.dModel / self.numHeads
    }

    // kyutai/neilz/mimi_exp/xps/f28fe6d5/.hydra/config.yaml
    public static func v1_300m() -> TransformerConfig {
        TransformerConfig(
            dModel: 1024,
            numHeads: 8,
            numLayers: 16,
            causal: true,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: nil,
            positionalEmbedding: .rope,
            useConvBias: false,
            gating: true,
            norm: .rmsNorm,
            context: 750,
            maxPeriod: 100000,
            maxSeqLen: 4096,
            kvRepeat: 1,
            dimFeedForward: 1024 * 4,
            convLayout: false,
            useRotatingKVCache: false
        )
    }

    // kyutai/neilz/mimi_exp/xps/8d2516b9/.hydra/config.yaml
    public static func v1_1b() -> TransformerConfig {
        TransformerConfig(
            dModel: 2048,
            numHeads: 16,
            numLayers: 16,
            causal: true,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: nil,
            positionalEmbedding: .rope,
            useConvBias: false,
            gating: true,
            norm: .rmsNorm,
            context: 3000,
            maxPeriod: 100000,
            maxSeqLen: 4096,
            kvRepeat: 1,
            dimFeedForward: 2048 * 4,
            convLayout: false,
            useRotatingKVCache: false
        )
    }

    // kyutai/neilz/mimi_exp/xps/b3f79570/.hydra/config.yaml
    public static func v1_2b() -> TransformerConfig {
        TransformerConfig(
            dModel: 2560,
            numHeads: 20,
            numLayers: 24,
            causal: true,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: nil,
            positionalEmbedding: .rope,
            useConvBias: false,
            gating: true,
            norm: .rmsNorm,
            context: 3000,
            maxPeriod: 100000,
            maxSeqLen: 4096,
            kvRepeat: 1,
            dimFeedForward: 2560 * 4,
            convLayout: false,
            useRotatingKVCache: false
        )
    }

    public static func v1_7b() -> TransformerConfig {
        TransformerConfig(
            dModel: 4096,
            numHeads: 32,
            numLayers: 32,
            causal: true,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: nil,
            positionalEmbedding: .rope,
            useConvBias: false,
            gating: true,
            norm: .rmsNorm,
            context: 3000,
            maxPeriod: 10000,
            maxSeqLen: 4096,
            kvRepeat: 1,
            dimFeedForward: 4096 * 4,
            convLayout: false,
            useRotatingKVCache: false
        )
    }
}

private class MlpGating: Module, UnaryLayer {
    @ModuleInfo(key: "linear_in") var linear_in: Linear
    @ModuleInfo(key: "linear_out") var linear_out: Linear

    init(_ cfg: TransformerConfig) {
        let hidden =
            cfg.dimFeedForward == 4 * cfg.dModel ? 11 * cfg.dModel / 4 : 2 * cfg.dimFeedForward / 3
        self._linear_in.wrappedValue = Linear(cfg.dModel, 2 * hidden, bias: cfg.biasFF)
        self._linear_out.wrappedValue = Linear(hidden, cfg.dModel, bias: cfg.biasFF)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = linear_in(x)
        let (B, T) = (x.dim(0), x.dim(1))
        let x_reshaped = x.reshaped(B, T, 2, -1)
        return linear_out(silu(x_reshaped[0..., 0..., 0]) * x_reshaped[0..., 0..., 1])
    }
}

private class MlpNoGating: Module, UnaryLayer {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ cfg: TransformerConfig) {
        self._linear1.wrappedValue = Linear(cfg.dModel, cfg.dimFeedForward, bias: cfg.biasFF)
        self._linear2.wrappedValue = Linear(cfg.dimFeedForward, cfg.dModel, bias: cfg.biasFF)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(geluApproximate(linear1(x)))
    }
}

private class Attention: Module {
    let cfg: TransformerConfig
    let scale: Float
    let rope: RoPE?

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ cfg: TransformerConfig) {
        self.cfg = cfg
        self.scale = 1.0 / sqrt(Float(cfg.headDim()))
        let numKV = cfg.numHeads / cfg.kvRepeat
        let outDim = cfg.dModel + 2 * numKV * cfg.dModel / cfg.numHeads
        self._inProj.wrappedValue = Linear(cfg.dModel, outDim, bias: cfg.biasAttn)
        self._outProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.biasAttn)
        self.rope =
            switch cfg.positionalEmbedding {
            case .none: nil
            case .rope:
                RoPE(dimensions: cfg.headDim(), traditional: true, base: Float(cfg.maxPeriod))
            }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let (B, T, H) = (x.dim(0), x.dim(1), x.dim(2))
        let qkv = inProj(x).reshaped(B, T, 3, cfg.numHeads, cfg.headDim())
        var q = qkv[0..., 0..., 0].transposed(0, 2, 1, 3)
        var k = qkv[0..., 0..., 1].transposed(0, 2, 1, 3)
        var v = qkv[0..., 0..., 2].transposed(0, 2, 1, 3)
        if let rope {
            let offset = cache?.offset ?? 0
            q = rope(q, offset: offset)
            k = rope(k, offset: offset)
        }
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }
        let kLen = k.dim(2)
        let kTargetLen = T + min(self.cfg.context, kLen - T)
        if kTargetLen < kLen {
            let offset = kLen - kTargetLen
            k = k[0..., 0..., offset...]
            v = v[0..., 0..., offset...]
        }

        var mask = mask
        if let m = mask {
            let maskLen = m.dim(-1)
            if k.dim(2) < maskLen {
                let offset = maskLen - k.dim(2)
                mask = m[0..., offset...]
            }
        }
        let x = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: self.scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, T, H)
        return outProj(x)
    }
}

class LayerScale: Module, UnaryLayer {
    @ModuleInfo(key: "scale") var scale: MLXArray

    init(_ dModel: Int, initValue: Float) {
        self._scale.wrappedValue = ones([dModel]) * initValue
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * self.scale
    }
}

private class TransformerLayer: Module {
    @ModuleInfo(key: "gating") var gating: UnaryLayer
    @ModuleInfo(key: "norm1") var norm1: UnaryLayer
    @ModuleInfo(key: "norm2") var norm2: UnaryLayer
    @ModuleInfo(key: "layer_scale_1") var layerScale1: LayerScale?
    @ModuleInfo(key: "layer_scale_2") var layerScale2: LayerScale?
    @ModuleInfo(key: "self_attn") var selfAttn: Attention

    init(_ cfg: TransformerConfig) {
        self._gating.wrappedValue = cfg.gating ? MlpGating(cfg) : MlpNoGating(cfg)
        self._norm1.wrappedValue =
            switch cfg.norm {
            case .layerNorm:
                LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
            case .rmsNorm: RMSNorm(dimensions: cfg.dModel, eps: 1e-8)
            }
        self._norm2.wrappedValue =
            switch cfg.norm {
            case .layerNorm:
                LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
            case .rmsNorm: RMSNorm(dimensions: cfg.dModel, eps: 1e-8)
            }
        self._selfAttn.wrappedValue = Attention(cfg)

        if let scale = cfg.layerScale {
            self._layerScale1.wrappedValue = LayerScale(cfg.dModel, initValue: scale)
            self._layerScale2.wrappedValue = LayerScale(cfg.dModel, initValue: scale)
        } else {
            self._layerScale1.wrappedValue = nil
            self._layerScale2.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache) -> MLXArray {
        var residual = x
        var x = x
        x = selfAttn(norm1(x), mask: mask, cache: cache)
        if let layerScale1 = self.layerScale1 {
            x = layerScale1(x)
        }
        x = residual + x
        residual = x
        x = gating(norm2(x))
        if let layerScale2 = self.layerScale2 {
            x = layerScale2(x)
        }
        return residual + x
    }
}

public class Transformer: Module {
    let cfg: TransformerConfig
    private let layers: [TransformerLayer]

    public init(_ cfg: TransformerConfig) {
        self.cfg = cfg
        self.layers = (0..<cfg.numLayers).map { _ in TransformerLayer(cfg) }
    }

    public func callAsFunction(_ x: MLXArray, cache: [KVCache]) -> MLXArray {
        var x = x
        let mask = cache.first?.createAttentionMask(h: x)
        for (layer, c) in zip(self.layers, cache) {
            x = layer(x, mask: mask, cache: c)
        }
        return x
    }

    public func makeCache(bSize: Int) -> [KVCache] {
        let kvHeads = cfg.numHeads / cfg.kvRepeat
        let dtype = self.layers.first!.selfAttn.inProj.weight.dtype
        let cache = (0..<cfg.numLayers).map { _ in
            let cache: KVCache
            if cfg.useRotatingKVCache {
                cache = RotatingKVCache(
                    bSize: bSize, numHeads: kvHeads, maxSize: cfg.context, headDim: cfg.headDim(),
                    dtype: dtype)
            } else {
                cache = KVCacheSimple(headDim: .init(cfg.headDim()), kvHeads: kvHeads)
            }
            return cache
        }
        return cache
    }
}

public class ProjectedTransformer: Module {
    let convLayout: Bool
    @ModuleInfo(key: "transformer") var transformer: Transformer
    @ModuleInfo(key: "input_proj") var inputProj: Linear?
    @ModuleInfo(key: "output_proj") var outputProjs: [Linear?]

    init(_ cfg: TransformerConfig, inputDim: Int, outputDims: [Int]) {
        self.convLayout = cfg.convLayout
        self._transformer.wrappedValue = Transformer(cfg)
        if inputDim == cfg.dModel {
            self._inputProj.wrappedValue = nil
        } else {
            self._inputProj.wrappedValue = Linear(inputDim, cfg.dModel, bias: false)
        }
        var outputProjs: [Linear?] = []
        for outputDim in outputDims {
            let outputProj =
                outputDim != cfg.dModel ? Linear(cfg.dModel, outputDim, bias: false) : nil
            outputProjs.append(outputProj)
        }
        self._outputProjs.wrappedValue = outputProjs
    }

    public func callAsFunction(_ x: MLXArray, cache: [KVCache]) -> [MLXArray] {
        var x = x
        if self.convLayout {
            x = x.swappedAxes(1, 2)
        }
        if let inputProj = self.inputProj {
            x = inputProj(x)
        }
        x = self.transformer(x, cache: cache)
        var outs: [MLXArray] = []
        for outputProj in self.outputProjs {
            var out = outputProj?(x) ?? x
            if self.convLayout {
                out = out.swappedAxes(1, 2)
            }
            outs.append(out)
        }
        return outs
    }

    public func makeCache(bSize: Int) -> [KVCache] {
        self.transformer.makeCache(bSize: bSize)
    }
}
