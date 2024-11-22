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
}

class Depformer: Module {
    let cfg: LmConfig
    let transformerCache: [KVCache]
    let slices: [DepformerSlice]

    public init(_ cfg: LmConfig) {
        self.cfg = cfg
        self.slices = (0..<cfg.depformer.numSlices).map { idx in
            DepformerSlice(
                inVocabSize: idx == 0 ? cfg.textInVocabSize : cfg.audioVocabSize,
                outVocabSize: cfg.audioVocabSize - 1,
                mainTransformerDim: cfg.transformer.dModel,
                cfg: cfg.depformer.transformer)
        }
        self.transformerCache = self.slices[0].transformer.makeCache()
    }

    public func sample(
        mainTransformerOut: MLXArray, stepIdx: Int, sampler: Sampler, textToken: MLXArray
    ) -> MLXArray {
        for c in self.transformerCache {
            c.reset()
        }
        var lastToken = textToken
        var tokens: [MLXArray] = []
        for (sliceIdx, slice) in slices.enumerated() {
            if sliceIdx == 0 || stepIdx < self.cfg.audioDelays[sliceIdx - 1] {
                lastToken = MLXArray([self.cfg.audioPaddingToken()])
            }
            var xs = slice.linearIn(mainTransformerOut) + slice.emb(lastToken)
            xs = slice.transformer(xs, cache: self.transformerCache)
            let logits = slice.linearOut(xs)
            (lastToken, _) = sampler(logits: logits[0])
            tokens.append(lastToken)
        }
        return concatenated(tokens)
    }
}

public struct LmConfig {
    public var transformer: TransformerConfig
    public var depformer: DepformerConfig
    public var textInVocabSize: Int
    public var textOutVocabSize: Int
    public var audioVocabSize: Int
    public var audioCodebooks: Int
    public var audioDelays: [Int]

    func audioEOSToken() -> Int {
        self.audioVocabSize - 2
    }

    func audioPaddingToken() -> Int {
        self.audioVocabSize - 1
    }

    public static func moshi_v0_1() -> LmConfig {
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
                    positionalEmbedding: .none,
                    useConvBias: false,
                    gating: true,
                    norm: .rmsNorm,
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
    let transformerCache: [KVCache]
    @ModuleInfo(key: "depformer") var depformer: Depformer
    @ModuleInfo(key: "transformer") public var transformer: Transformer
    @ModuleInfo(key: "text_emb") var textEmb: Embedding
    @ModuleInfo(key: "out_norm") var outNorm: UnaryLayer
    @ModuleInfo(key: "text_linear") var textLinear: Linear
    @ModuleInfo(key: "audio_embs") var audioEmbs: [Embedding]

    public init(_ cfg: LmConfig) {
        self._transformer.wrappedValue = Transformer(cfg.transformer)
        self._depformer.wrappedValue = Depformer(cfg)
        self._textEmb.wrappedValue = Embedding(
            embeddingCount: cfg.textInVocabSize, dimensions: cfg.transformer.dModel)
        self._outNorm.wrappedValue =
            switch cfg.transformer.norm {
            case .layerNorm:
                LayerNorm(dimensions: cfg.transformer.dModel, eps: 1e-5)
            case .rmsNorm: RMSNorm(dimensions: cfg.transformer.dModel, eps: 1e-8)
            }
        self._textLinear.wrappedValue = Linear(
            cfg.transformer.dModel, cfg.textOutVocabSize, bias: false)
        self._audioEmbs.wrappedValue = (0..<cfg.audioCodebooks).map { _ in
            Embedding(embeddingCount: cfg.audioVocabSize, dimensions: cfg.transformer.dModel)
        }
        self.transformerCache = self._transformer.wrappedValue.makeCache()

    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = textEmb(x)
        x = transformer(x, cache: self.transformerCache)
        return textLinear(outNorm(x))
    }
}
