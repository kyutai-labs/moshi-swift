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

class SeanetResnetBlock: Module, UnaryLayer, StreamingLayer {
    let skipOp: StreamingBinOp
    @ModuleInfo(key: "block") var block: [StreamableConv1d]
    @ModuleInfo(key: "shortcut") var shortcut: StreamableConv1d?

    init(_ cfg: SeanetConfig, dim: Int, kSizesAndDilations: [(Int, Int)]) {
        self.skipOp = StreamingBinOp(.add, axis: -1)
        var block: [StreamableConv1d] = []
        var shortcut: StreamableConv1d? = nil
        let hidden = dim / cfg.compress

        for (i, (kSize, dilation)) in kSizesAndDilations.enumerated() {
            let inC = i == 0 ? dim : hidden
            let outC = i == kSizesAndDilations.count - 1 ? dim : hidden
            let c = StreamableConv1d(
                inC: inC, outC: outC, kSize: kSize, stride: 1, dilation: dilation, groups: 1,
                bias: true, causal: cfg.causal, padMode: cfg.padMode)
            block.append(c)
        }
        if !cfg.trueSkip {
            shortcut = StreamableConv1d(
                inC: dim, outC: dim, kSize: 1, stride: 1, dilation: 1, groups: 1, bias: true,
                causal: cfg.causal, padMode: cfg.padMode)
        }

        self._block.wrappedValue = block
        self._shortcut.wrappedValue = shortcut
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var x = x
        for b in self.block {
            x = b(elu(x, alpha: 1.0))
        }
        if let shortcut = self.shortcut {
            return x + shortcut(residual)
        } else {
            return x + residual
        }
    }

    func resetState() {
        self.skipOp.resetState()
        for b in self.block {
            b.resetState()
        }
        if let s = self.shortcut {
            s.resetState()
        }
    }

    func step(_ x: StreamArray) -> StreamArray {
        let residual = x
        var x = x
        for b in self.block {
            if let inner = x.inner {
                x = StreamArray(elu(inner, alpha: 1.0))
            }
            x = b.step(x)
        }
        if let shortcut = self.shortcut {
            return self.skipOp.step(x, shortcut.step(residual))
        } else {
            return self.skipOp.step(x, residual)
        }
    }
}

class EncoderLayer: Module, UnaryLayer, StreamingLayer {
    @ModuleInfo(key: "residuals") var residuals: [SeanetResnetBlock]
    @ModuleInfo(key: "downsample") var downsample: StreamableConv1d

    init(_ cfg: SeanetConfig) {
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("TODO")
    }

    func resetState() {
        for r in self.residuals {
            r.resetState()
        }
        self.downsample.resetState()
    }

    func step(_ x: StreamArray) -> StreamArray {
        // TODO: Add activations
        var x = x
        for r in self.residuals {
            x = r.step(x)
        }
        return self.downsample.step(x)
    }
}

class SeanetEncoder: Module, StreamingLayer {
    @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
    @ModuleInfo(key: "layers") var layers: [EncoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

    init(_ cfg: SeanetConfig) {
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("TODO")
    }

    func resetState() {
        self.initConv1d.resetState()
        self.finalConv1d.resetState()
        for l in self.layers {
            l.resetState()
        }
    }

    func step(_ x: StreamArray) -> StreamArray {
        // TODO: Add activations
        var x = self.initConv1d.step(x)
        for layer in self.layers {
            x = layer.step(x)
        }
        return self.finalConv1d.step(x)
    }
}

class DecoderLayer: Module, UnaryLayer, StreamingLayer {
    @ModuleInfo(key: "upsample") var upsample: StreamableConvTranspose1d
    @ModuleInfo(key: "residuals") var residuals: [SeanetResnetBlock]

    init(_ cfg: SeanetConfig) {
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("TODO")
    }

    func resetState() {
        self.upsample.resetState()
        for r in self.residuals {
            r.resetState()
        }
    }

    func step(_ x: StreamArray) -> StreamArray {
        fatalError("TODO")
    }
}

class SeanetDecoder: Module, StreamingLayer {
    @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
    @ModuleInfo(key: "layers") var layers: [DecoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

    init(_ cfg: SeanetConfig) {
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("TODO")
    }

    func resetState() {
        self.initConv1d.resetState()
        self.finalConv1d.resetState()
        for l in self.layers {
            l.resetState()
        }
    }

    func step(_ x: StreamArray) -> StreamArray {
        var x = self.initConv1d.step(x)
        for layer in self.layers {
            x = layer.step(x)
        }
        return self.finalConv1d.step(x)
    }
}
