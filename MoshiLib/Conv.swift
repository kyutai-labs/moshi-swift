// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN
import MLXRandom

// Conv1d + dilation
class Conv1d: Module, UnaryLayer {
    let weight: MLXArray
    let bias: MLXArray?
    let padding: Int
    let groups: Int
    let stride: Int
    let dilation: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize))

        self.weight = uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels / groups])
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding
        self.groups = groups
        self.stride = stride
        self.dilation = dilation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(
            x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias {
            y = y + bias
        }
        return y
    }
}

// ConvTranspose1d + groups
class ConvTransposed1d: Module, UnaryLayer {

    let weight: MLXArray
    let bias: MLXArray?
    let padding: Int
    let stride: Int
    let groups: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize))

        self.weight = uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels])
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding
        self.stride = stride
        self.groups = groups
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convTransposed1d(x, weight, stride: stride, padding: padding, groups: groups)
        if let bias {
            y = y + bias
        }
        return y
    }
}

// TODO: Handle weight-norm either when importing the weights or at runtime.
class NormConv1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(
        inC: Int, outC: Int, kSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inC, outputChannels: outC, kernelSize: kSize, stride: stride,
            padding: padding,
            groups: groups, dilation: dilation, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        self.conv(x)
    }
}

class NormConvTranspose1d: Module, UnaryLayer {
    @ModuleInfo(key: "convtr") var convtr: ConvTransposed1d

    init(
        inC: Int, outC: Int, kSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self._convtr.wrappedValue = ConvTransposed1d(
            inputChannels: inC, outputChannels: outC, kernelSize: kSize, stride: stride,
            padding: padding, groups: groups, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        self.convtr(x)
    }
}

func getExtraPaddingForConv1d(_ x: MLXArray, kSize: Int, stride: Int, paddingTotal: Int) -> Int {
    let len = x.dim(-1)
    let nFrames = Float(max(len + paddingTotal - kSize, 0)) / Float(stride) + 1.0
    let idealLen = (Int(nFrames.rounded(.up)) - 1) * stride + kSize - paddingTotal
    return max(0, idealLen - len)
}

func unpad1d(_ x: MLXArray, unpadL: Int, unpadR: Int) -> MLXArray {
    let len = x.dim(-1)
    let left = unpadL
    let right = len - unpadR
    return x[.ellipsis, left..<right]
}

class StreamableConv1d: Module, UnaryLayer {
    let padMode: PadMode
    let causal: Bool
    @ModuleInfo(key: "conv") var conv: NormConv1d

    init(
        inC: Int, outC: Int, kSize: Int, stride: Int, dilation: Int, groups: Int, bias: Bool,
        causal: Bool, padMode: PadMode
    ) {
        self.causal = causal
        self.padMode = padMode
        self._conv.wrappedValue = NormConv1d(
            inC: inC, outC: outC, kSize: kSize, groups: groups, dilation: dilation, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var kSize = self.conv.conv.weight.dim(-1)
        // Effective kernel size with dilations.
        kSize = (kSize - 1) * self.conv.conv.dilation + 1
        fatalError("todo")
    }
}

class StreamableConvTranspose1d: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("todo")
    }
}

class ConvDownsample1d: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("todo")
    }
}

class ConvTrUpsample1d: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("todo")
    }
}
