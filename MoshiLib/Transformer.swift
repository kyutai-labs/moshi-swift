// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN

public struct TransformerConfig {
    var dModel: Int
    var numHeads: Int
    var numLayers: Int
    var causal: Bool
    var normFirst: Bool
    var biasFF: Bool
    var biasAttn: Bool
    var layerScale: Float?
    var positionalEmbedding: String
    var useConvBias: Bool
    var gating: Bool
    var norm: String
    var context: Int
    var maxPeriod: Int
    var maxSeqLen: Int
    var kvRepeat: Int
    var dimFeedForward: Int
    var convLayout: Bool

    func headDim() -> Int {
        self.dModel / self.numHeads
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
            // TODO: Use proper types rather than strings here.
            positionalEmbedding: "rope",
            useConvBias: false,
            gating: true,
            norm: "rms_norm",
            context: 3000,
            maxPeriod: 10000,
            maxSeqLen: 4096,
            kvRepeat: 1,
            dimFeedForward: 4096 * 4,
            convLayout: false
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

    @ModuleInfo(key: "in_proj") var in_proj: Linear
    @ModuleInfo(key: "out_proj") var out_proj: Linear

    init(_ cfg: TransformerConfig) {
        self.cfg = cfg
        self.scale = 1.0 / sqrt(Float(cfg.headDim()))
        let numKV = cfg.numHeads / cfg.kvRepeat
        let outDim = cfg.dModel + 2 * numKV * cfg.dModel / cfg.numHeads
        self._in_proj.wrappedValue = Linear(cfg.dModel, outDim, bias: cfg.biasAttn)
        self._out_proj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.biasAttn)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // TODO
        x
    }
}

private class TransformerLayer: Module {
    @ModuleInfo(key: "gating") var gating: UnaryLayer
    @ModuleInfo(key: "norm1") var norm1: UnaryLayer
    @ModuleInfo(key: "norm2") var norm2: UnaryLayer
    @ModuleInfo(key: "self_attn") var selfAttn: Attention

    init(_ cfg: TransformerConfig) {
        self._gating.wrappedValue = cfg.gating ? MlpGating(cfg) : MlpNoGating(cfg)
        self._norm1.wrappedValue =
            cfg.norm == "layer_norm"
            ? LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
            : RMSNorm(dimensions: cfg.dModel, eps: 1e-8)
        self._norm2.wrappedValue =
            cfg.norm == "layer_norm"
            ? LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
            : RMSNorm(dimensions: cfg.dModel, eps: 1e-8)
        self._selfAttn.wrappedValue = Attention(cfg)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // TODO
        x
    }
}

public class Transformer: Module {
    private let layers: [TransformerLayer]

    public init(_ cfg: TransformerConfig) {
        self.layers = (0..<cfg.numLayers).map { _ in TransformerLayer(cfg) }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // TODO
        x
    }
}
