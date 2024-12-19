// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN
import MLXRandom

func topKSampling(logits: MLXArray, topK: Int, temp: Float) -> MLXArray {
    let c = logits.dim(-1)
    let sortedIndices = argSort(logits, axis: -1)
    let sortedLogits = logits[0..., sortedIndices.squeezed(axis: 0)]
    let topLogits = `where`(MLXArray(0..<c) .>= c - topK, sortedLogits, -Float.infinity)
    let sortedToken = categorical(topLogits / temp)
    let token = sortedIndices.squeezed(axis: 0)[sortedToken]
    return token
}

func topPSampling(logits: MLXArray, topP: Float, temp: Float) -> MLXArray {
    let probs = softmax(logits * (1 / temp), axis: -1)
    let sortedIndices = argSort(probs, axis: -1)
    let sortedProbs = probs[0..., sortedIndices.squeezed(axis: 0)]
    let cumulativeProbs = cumsum(sortedProbs, axis: -1)
    let topProbs = `where`(cumulativeProbs .> 1 - topP, sortedProbs, 0)
    let sortedToken = categorical(topProbs.log())
    let token = sortedIndices.squeezed(axis: 0)[sortedToken]
    return token
}

func categoricalSampling(logits: MLXArray, temp: Float) -> MLXArray {
    categorical(logits * (1 / temp))
}

public class Sampler {
    let temp: Float
    let topP: Float = 0.95
    let topK: Int = 0

    public init(temp: Float = 0.8) {
        self.temp = temp
    }

    public func callAsFunction(logits: MLXArray) -> (MLXArray, MLXArray) {
        if logits.shape.count != 2 {
            fatalError("expected a two dimensions logits array, got \(logits.shape)")
        }
        let logProbs = logits - logits.logSumExp()
        var tokens: MLXArray
        if temp <= 0.0 {
            tokens = logProbs.argMax(axis: -1)
        } else if self.topP > 0.0 && self.topP < 1.0 {
            tokens = topPSampling(logits: logits, topP: self.topP, temp: self.temp)
        } else if self.topK != 0 {
            tokens = topKSampling(logits: logits, topK: self.topK, temp: self.temp)
        } else {
            tokens = categoricalSampling(logits: logits, temp: self.temp)
        }
        return (tokens, logProbs)
    }
}
