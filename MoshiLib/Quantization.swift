// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN

class EuclideanCodebook {
    let epsilon: Float
    let dim: Int
    var embedding: MLXArray? = nil
    var c2: MLXArray? = nil
    @ModuleInfo(key: "_initialized") var initialized: MLXArray
    @ModuleInfo(key: "embedding_sum") var embeddingSum: MLXArray
    @ModuleInfo(key: "cluster_usage") var clusterUsage: MLXArray

    init(dim: Int, codebookSize: Int) {
        self.epsilon = 1e-5
        self.dim = dim
        self._initialized.wrappedValue = MLXArray.zeros([1], dtype: .float32)
        self._embeddingSum.wrappedValue = MLXArray.zeros([codebookSize], dtype: .float32)
        self._clusterUsage.wrappedValue = MLXArray.zeros([codebookSize, dim], dtype: .float32)
    }

    // Precompute the embedding and c2 tensors
    func embeddingAndC2() -> (MLXArray, MLXArray) {
        var embedding: MLXArray
        switch self.embedding {
        case .none:
            let clusterUsage = maximum(self.clusterUsage, self.epsilon)[0..., .newAxis]
            embedding = self.embeddingSum / clusterUsage
            self.embedding = embedding
        case .some(let e): embedding = e
        }

        var c2: MLXArray
        switch self.c2 {
        case .none:
            c2 = embedding.square().sum(axis: -1) / 2
            self.c2 = c2
        case .some(let e): c2 = e
        }

        return (embedding, c2)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        let (embedding, c2) = self.embeddingAndC2()
        let targetShape = x.shape
        let x = x.flattened(start: -2)
        let dotProd = x.matmul(embedding.transposed())
        return (c2 - dotProd).argMax(axis: -1).reshaped(targetShape)
    }

    func decode(_ indexes: MLXArray) -> MLXArray {
        let finalDims = indexes.shape + [self.dim]
        let indexes = indexes.flattened()
        let (embedding, _) = self.embeddingAndC2()
        return embedding.take(indexes, axis: 0).reshaped(finalDims)
    }
}
