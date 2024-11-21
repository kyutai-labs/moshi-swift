// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN

public struct DepformerConfig {
    var transformer: TransformerConfig
    var numSlices: Int
}

class DepformerSlice: Module {
    @ModuleInfo(key: "emb") var emb: Embedding
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "transformer") var transformer: Transformer

    public init(
        inVocabSize: Int, outVocabSize: Int, mainTransformerDim: Int, cfg: TransformerConfig
    ) {
        self._emb.wrappedValue = Embedding(embeddingCount: inVocabSize, dimensions: cfg.dModel)
        self._linearIn.wrappedValue = Linear(mainTransformerDim, cfg.dModel, bias: false)
        self._linearOut.wrappedValue = Linear(cfg.dModel, outVocabSize, bias: false)
        self._transformer.wrappedValue = Transformer(cfg)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // TODO
        x
    }
}

class Depformer: Module {
    let slices: [DepformerSlice]

    public init(_ cfg: LmConfig) {
        self.slices = (0..<cfg.depformer.numSlices).map { idx in
            DepformerSlice(
                inVocabSize: idx == 0 ? cfg.textInVocabSize : cfg.audioVocabSize,
                outVocabSize: cfg.audioVocabSize - 1,
                mainTransformerDim: cfg.transformer.dModel,
                cfg: cfg.depformer.transformer)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // TODO
        x
    }
}

public struct LmConfig {
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

    public static func v0_1() -> LmConfig {
        let depformer = DepformerConfig(
            transformer:
                TransformerConfig(
                    dModel: 1024,
                    numHeads: 16,
                    numLayers: 6,
                    causal: true,
                    normFirst: true,
                    biasFF: false,
                    biasAttn: false,
                    layerScale: nil,
                    // TODO: Use proper types rather than strings here.
                    positionalEmbedding: "none",
                    useConvBias: false,
                    gating: true,
                    norm: "rms_norm",
                    context: 8,
                    maxPeriod: 10000,
                    maxSeqLen: 4096,
                    kvRepeat: 1,
                    dimFeedForward: 1024 * 4,
                    convLayout: false
                ), numSlices: 8)
        return LmConfig(
            transformer: TransformerConfig.v1_7b(),
            depformer: depformer,
            textInVocabSize: 32001,
            textOutVocabSize: 32000,
            audioVocabSize: 2049,
            audioCodebooks: 16,
            audioDelays: [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        )
    }
}

public class LM: Module {
    @ModuleInfo(key: "depformer") var depformer: Depformer
    @ModuleInfo(key: "transformer") var transformer: Transformer
    @ModuleInfo(key: "text_emb") var textEmb: Embedding
    @ModuleInfo(key: "out_norm") var outNorm: Module
    @ModuleInfo(key: "text_linear") var textLinear: Linear
    @ModuleInfo(key: "audio_embs") var audioEmbs: [Embedding]

    public init(_ cfg: LmConfig) {
        self._transformer.wrappedValue = Transformer(cfg.transformer)
        self._depformer.wrappedValue = Depformer(cfg)
        self._textEmb.wrappedValue = Embedding(
            embeddingCount: cfg.textInVocabSize, dimensions: cfg.transformer.dModel)
        self._outNorm.wrappedValue =
            cfg.transformer.norm == "layer_norm"
            ? LayerNorm(dimensions: cfg.transformer.dModel, eps: 1e-5)
            : RMSNorm(dimensions: cfg.transformer.dModel, eps: 1e-8)
        self._textLinear.wrappedValue = Linear(
            cfg.transformer.dModel, cfg.textOutVocabSize, bias: false)
        self._audioEmbs.wrappedValue = (0..<cfg.audioCodebooks).map { _ in
            Embedding(embeddingCount: cfg.audioVocabSize, dimensions: cfg.transformer.dModel)
        }

    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // TODO
        x
    }
}
