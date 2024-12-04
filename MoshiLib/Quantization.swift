// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN

class EuclideanCodebook: Module {
    let epsilon: Float
    let dim: Int
    var embedding: MLXArray
    var c2: MLXArray
    @ModuleInfo(key: "_initialized") var initialized: MLXArray
    @ModuleInfo(key: "embedding_sum") var embeddingSum: MLXArray
    @ModuleInfo(key: "cluster_usage") var clusterUsage: MLXArray

    init(dim: Int, codebookSize: Int) {
        self.epsilon = 1e-5
        self.dim = dim
        self._initialized.wrappedValue = MLXArray.zeros([1], dtype: .float32)
        self._embeddingSum.wrappedValue = MLXArray.zeros([codebookSize, dim], dtype: .float32)
        self._clusterUsage.wrappedValue = MLXArray.zeros([codebookSize], dtype: .float32)
        let clusterUsage = maximum(self._clusterUsage.wrappedValue, self.epsilon)[0..., .newAxis]
        self.embedding = self._embeddingSum.wrappedValue / clusterUsage
        self.c2 = self.embedding.square().sum(axis: -1) / 2
    }

    override func update(parameters: ModuleParameters, verify: Module.VerifyUpdate) throws -> Self {
        try super.update(parameters: parameters, verify: verify)
        let clusterUsage = maximum(self._clusterUsage.wrappedValue, self.epsilon)[0..., .newAxis]
        self.embedding = self._embeddingSum.wrappedValue / clusterUsage
        self.c2 = self.embedding.square().sum(axis: -1) / 2
        return self
    }

    func encode(_ x: MLXArray) -> MLXArray {
        let targetShape = Array(x.shape.dropLast())
        let x = x.flattened(end: -2)
        let dotProd = x.matmul(embedding.swappedAxes(-1, -2))
        return (c2 - dotProd).argMin(axis: -1).reshaped(targetShape)
    }

    func decode(_ indexes: MLXArray) -> MLXArray {
        let finalDims = indexes.shape + [self.dim]
        let indexes = indexes.flattened()
        return embedding.take(indexes, axis: 0).reshaped(finalDims)
    }
}

class VectorQuantization: Module {
    @ModuleInfo(key: "project_in") var projectIn: Linear?
    @ModuleInfo(key: "project_out") var projectOut: Linear?
    @ModuleInfo(key: "_codebook") var codebook: EuclideanCodebook

    init(dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        let codebookDim = codebookDim ?? dim
        if codebookDim == dim {
            self._projectIn.wrappedValue = nil
            self._projectOut.wrappedValue = nil
        } else {
            self._projectIn.wrappedValue = Linear(dim, codebookDim)
            self._projectOut.wrappedValue = Linear(codebookDim, dim)
        }
        self._codebook.wrappedValue = EuclideanCodebook(
            dim: codebookDim, codebookSize: codebookSize)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        var x = x.swappedAxes(-1, -2)
        if let projectIn = self.projectIn {
            x = projectIn(x)
        }
        return self.codebook.encode(x)
    }

    func decode(_ indexes: MLXArray) -> MLXArray {
        var quantized = self.codebook.decode(indexes)
        if let projectOut = self.projectOut {
            quantized = projectOut(quantized)
        }
        return quantized.swappedAxes(-1, -2)
    }
}

class ResidualVectorQuantization: Module {
    @ModuleInfo(key: "layers") var layers: [VectorQuantization]

    init(nQ: Int, dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        self._layers.wrappedValue = (0..<nQ).map { _ in
            VectorQuantization(dim: dim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
    }

    func encode(_ x: MLXArray) -> MLXArray {
        var codes: [MLXArray] = []
        var residual = x
        for layer in self.layers {
            let indices = layer.encode(residual)
            let quantized = layer.decode(indices)
            residual = residual - quantized
            codes.append(indices)
        }
        return stacked(codes, axis: 0)
    }

    func decode(_ indexes: MLXArray) -> MLXArray {
        let seqLen = indexes.dim(0)
        var quantized = self.layers[0].decode(indexes[0])
        for i in 1..<seqLen {
            quantized = quantized + self.layers[i].decode(indexes[i])
        }
        return quantized
    }
}

class ResidualVectorQuantizer: Module {
    @ModuleInfo(key: "vq") var vq: ResidualVectorQuantization
    @ModuleInfo(key: "input_proj") var inputProj: Conv1d?
    @ModuleInfo(key: "output_proj") var outputProj: Conv1d?

    init(dim: Int, inputDim: Int?, outputDim: Int?, nQ: Int, bins: Int, forceProjection: Bool) {
        let inputDim = inputDim ?? dim
        let outputDim = outputDim ?? dim
        if inputDim == dim && !forceProjection {
            self._inputProj.wrappedValue = nil
        } else {
            self._inputProj.wrappedValue = Conv1d(
                inputChannels: inputDim, outputChannels: dim, kernelSize: 1, bias: false)
        }
        if outputDim == dim && !forceProjection {
            self._outputProj.wrappedValue = nil
        } else {
            self._outputProj.wrappedValue = Conv1d(
                inputChannels: dim, outputChannels: outputDim, kernelSize: 1, bias: false)
        }
        self._vq.wrappedValue = ResidualVectorQuantization(
            nQ: nQ, dim: dim, codebookSize: bins, codebookDim: nil)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        var x = x
        if let inputProj = self.inputProj {
            x = inputProj(x)
        }
        return self.vq.encode(x).swappedAxes(0, 1)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        let codes = codes.swappedAxes(0, 1)
        var quantized = self.vq.decode(codes)
        if let outputProj = self.outputProj {
            quantized = outputProj(quantized)
        }
        return quantized
    }
}

class SplitResidualVectorQuantizer: Module {
    let nQ: Int
    @ModuleInfo(key: "rvq_first") var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo(key: "rvq_rest") var rvqRest: ResidualVectorQuantizer

    init(dim: Int, inputDim: Int?, outputDim: Int?, nQ: Int, bins: Int) {
        self.nQ = nQ
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            dim: dim, inputDim: inputDim, outputDim: outputDim, nQ: 1, bins: bins,
            forceProjection: true)
        self._rvqRest.wrappedValue = ResidualVectorQuantizer(
            dim: dim, inputDim: inputDim, outputDim: outputDim, nQ: nQ - 1, bins: bins,
            forceProjection: true)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        var codes = self.rvqFirst.encode(x)
        if self.nQ > 1 {
            let restCodes = self.rvqRest.encode(x)
            codes = concatenated([codes, restCodes], axis: 1)
        }
        return codes
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = self.rvqFirst.decode(codes[0..., ..<1])
        if self.nQ > 1 {
            quantized = quantized + self.rvqRest.decode(codes[0..., 1...])
        }
        return quantized
    }
}
