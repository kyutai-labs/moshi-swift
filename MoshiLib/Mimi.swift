// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN
import MLXRandom

public struct MimiConfig {
    public var channels: Int
    public var sampleRate: Float
    public var frameRate: Float
    public var renormalize: Bool
    // public var resampleMethod: String
    public var seanet: SeanetConfig
    public var transformer: TransformerConfig
    public var quantizerNQ: Int
    public var quantizerBins: Int
    public var quantizerDim: Int

    public static func v0_1(numCodebooks: Int = 16) -> MimiConfig {
        let seanet = SeanetConfig.v0_1()
        let transformer = TransformerConfig(
            dModel: seanet.dimension, numHeads: 8, numLayers: 8, causal: true, normFirst: true,
            biasFF: false, biasAttn: false, positionalEmbedding: .rope, useConvBias: true,
            gating: false, norm: .layerNorm, context: 250, maxPeriod: 10000, maxSeqLen: 8192,
            kvRepeat: 1, dimFeedForward: 2048, convLayout: true
        )
        return MimiConfig(
            channels: 1,
            sampleRate: 24000, frameRate: 12.5, renormalize: true, seanet: seanet,
            transformer: transformer, quantizerNQ: numCodebooks, quantizerBins: 2048,
            quantizerDim: 256)
    }
}

public class Mimi: Module {
    let cfg: MimiConfig
    @ModuleInfo(key: "encoder") var encoder: SeanetEncoder
    @ModuleInfo(key: "decoder") var decoder: SeanetDecoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: ProjectedTransformer
    @ModuleInfo(key: "decoder_transformer") var decoderTransformer: ProjectedTransformer
    @ModuleInfo(key: "downsample") var downsample: ConvDownsample1d
    @ModuleInfo(key: "upsample") var upsample: ConvTrUpsample1d
    @ModuleInfo(key: "quantizer") var quantizer: SplitResidualVectorQuantizer

    init(_ cfg: MimiConfig) {
        let dim = cfg.seanet.dimension
        self.cfg = cfg
        self._encoder.wrappedValue = SeanetEncoder(cfg.seanet)
        self._decoder.wrappedValue = SeanetDecoder(cfg.seanet)
        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            dim: cfg.quantizerDim, inputDim: dim, outputDim: dim, nQ: cfg.quantizerNQ,
            bins: cfg.quantizerBins)
        self._encoderTransformer.wrappedValue = ProjectedTransformer(
            cfg.transformer, inputDim: dim, outputDims: [dim])
        self._decoderTransformer.wrappedValue = ProjectedTransformer(
            cfg.transformer, inputDim: dim, outputDims: [dim])
    }
}
