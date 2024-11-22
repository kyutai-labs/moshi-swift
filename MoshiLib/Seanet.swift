// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN
import MLXRandom

public struct SeanetConfig {
    public var dimension: Int
    public var channels: Int
    public var causal: Bool
    public var nFilters: Int
    public var nResidualLayers: Int
    public var ratios: [Int]
    // public var activation: String: hardcoded to Elu(1) for now
    // public var norm: Norm
    public var kernelSize: Int
    public var residualKernelSize: Int
    public var lastKernelSize: Int
    public var dilationBase: Int
    public var padMode: PadMode
    public var trueSkip: Bool
    public var compress: Int
    public var disableNormOuterBlocks: Int
    // public var finalActivation: String?: hardcoded to None for now

    public static func v0_1() -> SeanetConfig {
        SeanetConfig(
            dimension: 512, channels: 1, causal: true, nFilters: 64, nResidualLayers: 1,
            ratios: [8, 6, 5, 4], kernelSize: 7, residualKernelSize: 3, lastKernelSize: 3,
            dilationBase: 2, padMode: .constant, trueSkip: true, compress: 2,
            disableNormOuterBlocks: 0)
    }
}

class SeanetResnetBlock: Module, UnaryLayer {
    @ModuleInfo(key: "block") var block: [StreamableConv1d]
    @ModuleInfo(key: "shortcut") var shortcut: StreamableConv1d?

    init(_ cfg: SeanetConfig) {
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("TODO")
    }

    // TODO: Streaming implementation.
}

class EncoderLayer: Module, UnaryLayer {
    @ModuleInfo(key: "residuals") var residuals: [SeanetResnetBlock]
    @ModuleInfo(key: "downsample") var downsample: StreamableConv1d

    init(_ cfg: SeanetConfig) {
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("TODO")
    }

    // TODO: Streaming implementation.
}

class SeanetEncoder {
    @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
    @ModuleInfo(key: "layers") var layers: [EncoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

    init(_ cfg: SeanetConfig) {
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("TODO")
    }

    // TODO: Streaming implementation.
}

class DecoderLayer: Module, UnaryLayer {
    @ModuleInfo(key: "upsample") var upsample: StreamableConvTranspose1d
    @ModuleInfo(key: "residuals") var residuals: [SeanetResnetBlock]

    init(_ cfg: SeanetConfig) {
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("TODO")
    }

    // TODO: Streaming implementation.
}

class SeanetDecoder {
    @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
    @ModuleInfo(key: "layers") var layers: [DecoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

    init(_ cfg: SeanetConfig) {
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("TODO")
    }

    // TODO: Streaming implementation.
}
