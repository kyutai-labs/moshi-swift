// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN

public struct TransformerConfig: Codable, Sendable {
    var dModel: Int
    var numHeads: Int
    var numLayers: Int
    var causal: Bool
    var normFirst: Bool
    var biasFF: Bool
    var biasAttn: Bool
    var layerScale: Float?
    var positionalEmbeddeing: String
    var useConvBias: Bool
    var gating: Bool
    var norm: String
    var context: Int
    var maxPeriod: Int
    var maxSeqLen: Int
    var dimFeedForward: Int
    var convLayout: Bool

    func headDim() -> Int {
        self.dModel / self.numHeads
    }
}

public struct DepformerConfig: Codable, Sendable {
    var transformer: TransformerConfig
    var numSlices: Int
}

public struct LmConfig: Codable, Sendable {
    var transformer: TransformerConfig
    var depformer: DepformerConfig
    var textInVocabSize: Int
    var textOutVocabSize: Int
    var audioVocabSize: Int
    var audioCodebooks: Int
    var audioDelays: [Int]

    func audioEOSToken() -> Int {
        self.audioVocabSize - 2
    }

    func audioPaddingToken() -> Int {
        self.audioVocabSize - 1
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
