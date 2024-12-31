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

    public static func mimi_2024_07(numCodebooks: Int = 16) -> MimiConfig {
        let seanet = SeanetConfig.v0_1()
        let transformer = TransformerConfig(
            dModel: seanet.dimension,
            numHeads: 8,
            numLayers: 8,
            causal: true,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: 0.01,
            positionalEmbedding: .rope,
            useConvBias: true,
            gating: false,
            norm: .layerNorm,
            context: 250,
            maxPeriod: 10000,
            maxSeqLen: 8192,
            kvRepeat: 1,
            dimFeedForward: 2048,
            convLayout: true,
            useRotatingKVCache: true
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
    let encoderCache: [KVCache]
    let decoderCache: [KVCache]
    @ModuleInfo(key: "encoder") var encoder: SeanetEncoder
    @ModuleInfo(key: "decoder") var decoder: SeanetDecoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: ProjectedTransformer
    @ModuleInfo(key: "decoder_transformer") var decoderTransformer: ProjectedTransformer
    @ModuleInfo(key: "downsample") var downsample: ConvDownsample1d
    @ModuleInfo(key: "upsample") var upsample: ConvTrUpsample1d
    @ModuleInfo(key: "quantizer") var quantizer: SplitResidualVectorQuantizer

    public init(_ cfg: MimiConfig, bSize: Int) {
        let dim = cfg.seanet.dimension
        self.cfg = cfg
        let encoderFrameRate = cfg.sampleRate / Float(cfg.seanet.ratios.reduce(1, *))
        let downsampleStride = Int(encoderFrameRate / cfg.frameRate)
        self._encoder.wrappedValue = SeanetEncoder(cfg.seanet)
        self._decoder.wrappedValue = SeanetDecoder(cfg.seanet)
        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            dim: cfg.quantizerDim, inputDim: dim, outputDim: dim, nQ: cfg.quantizerNQ,
            bins: cfg.quantizerBins)
        self._encoderTransformer.wrappedValue = ProjectedTransformer(
            cfg.transformer, inputDim: dim, outputDims: [dim])
        self._decoderTransformer.wrappedValue = ProjectedTransformer(
            cfg.transformer, inputDim: dim, outputDims: [dim])
        self._downsample.wrappedValue = ConvDownsample1d(
            stride: downsampleStride, dim: dim, causal: true)
        self._upsample.wrappedValue = ConvTrUpsample1d(
            stride: downsampleStride, dim: dim, causal: true)
        self.encoderCache = self._encoderTransformer.wrappedValue.makeCache(bSize: bSize)
        self.decoderCache = self._decoderTransformer.wrappedValue.makeCache(bSize: bSize)
    }

    public func warmup() {
        let pcm = MLXArray.zeros([1, 1, 1920 * 4])
        let codes = self.encode(pcm)
        let pcmOut = self.decode(codes)
        eval(pcmOut)
    }

    public func encode(_ x: MLXArray) -> MLXArray {
        self.encoder.resetState()
        self.encoderCache.forEach { c in c.reset() }
        var x = self.encoder(x)
        x = self.encoderTransformer(x, cache: self.encoderCache)[0]
        x = self.downsample(x)
        let codes = self.quantizer.encode(x)
        return codes
    }

    public func encodeStep(_ x: StreamArray) -> StreamArray {
        var x = self.encoder.step(x)
        x = x.map { self.encoderTransformer($0, cache: self.encoderCache)[0] }
        x = self.downsample.step(x)
        let codes = x.map(self.quantizer.encode)
        return codes
    }

    public func decode(_ codes: MLXArray) -> MLXArray {
        self.decoder.resetState()
        self.decoderCache.forEach { c in c.reset() }
        let emb = self.quantizer.decode(codes)
        let embUp = self.upsample(emb)
        let outs = self.decoderTransformer(embUp, cache: self.decoderCache)
        return self.decoder(outs[0])
    }

    public func decodeStep(_ codes: StreamArray) -> StreamArray {
        var emb = codes.map { self.quantizer.decode($0) }
        emb = self.upsample.step(emb)
        let out = emb.map { self.decoderTransformer($0, cache: self.decoderCache)[0] }
        return self.decoder.step(out)
    }

    public func resetState() {
        self.encoder.resetState()
        self.decoder.resetState()
        self.encoderCache.forEach { c in c.reset() }
        self.decoderCache.forEach { c in c.reset() }
    }
}
