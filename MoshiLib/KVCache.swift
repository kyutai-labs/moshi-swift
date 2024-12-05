// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Parts of this file came from:
// https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/LLM/KVCache.swift
// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Interface for Key/Value cache for LLMs.
///
/// See ``LLMModel/newCache(parameters:)-47tyu``
public protocol KVCache: Evaluatable {

    /// get the current offset
    var offset: Int { get }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func reset()
    func createAttentionMask(h: MLXArray) -> MLXArray?
}

func createAdditiveCausalMask(n: Int, offset: Int) -> MLXArray {
    let rinds = MLXArray(Int32(0)..<Int32(offset + n))
    let linds = offset != 0 ? MLXArray(Int32(offset)..<Int32(offset + n)) : rinds
    let mask = linds[0..., .newAxis] .< rinds[.newAxis]
    return mask * Float32(-1e9)
}

/// See https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/base.py#L11
class KVCacheSimple: KVCache, Evaluatable {
    let kHeadDim: Int
    let vHeadDim: Int
    let kvHeads: Int

    var keys: MLXArray?
    var values: MLXArray?

    var offset = 0
    var step = 256

    init(headDim: IntOrPair, kvHeads: Int) {
        self.kHeadDim = headDim.first
        self.vHeadDim = headDim.second
        self.kvHeads = kvHeads
    }

    public func reset() {
        self.keys = nil
        self.values = nil
        self.offset = 0
        self.step = 256
    }

    public func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous..<self.offset, 0...] = keys
        self.values?[.ellipsis, previous..<self.offset, 0...] = values

        return (
            self.keys![.ellipsis, ..<self.offset, 0...],
            self.values![.ellipsis, ..<self.offset, 0...]
        )
    }

    /// create an attention mask using the parameters from the KVCache.
    ///
    /// See also ``MultiHeadAttention/createAdditiveCausalMask(_:dtype:)`` -- same idea
    /// but doesn't honor the cache offset.
    func createAttentionMask(h: MLXArray) -> MLXArray? {
        let t = h.dim(1)
        if t > 1 {
            let rinds = MLXArray(Int32(0)..<Int32(offset + t))
            let linds = offset != 0 ? MLXArray(Int32(offset)..<Int32(offset + t)) : rinds
            let mask = linds[0..., .newAxis] .< rinds[.newAxis]
            return (mask * Float32(-1e9)).asType(h.dtype)
        }
        return nil
    }
}

class RotatingKVCache: KVCache, Evaluatable {
    let keys: MLXArray
    let values: MLXArray
    let maxSize: Int
    var offset: Int = 0

    init(bSize: Int, numHeads: Int, maxSize: Int, headDim: Int, dtype: DType) {
        self.keys = MLXArray.zeros([bSize, numHeads, maxSize, headDim], dtype: dtype)
        self.values = MLXArray.zeros([bSize, numHeads, maxSize, headDim], dtype: dtype)
        self.maxSize = maxSize
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let t = keys.dim(2)
        if t > self.maxSize {
            fatalError("query to update with shape \(keys.shape) larger than maxSize \(maxSize)")
        }
        let currentOffset = self.offset % self.maxSize
        let tMax = min(self.maxSize, currentOffset + t)
        self.keys[0..., 0..., currentOffset..<tMax] = keys[0..., 0..., 0..<(tMax - currentOffset)]
        self.values[0..., 0..., currentOffset..<tMax] =
            values[0..., 0..., 0..<(tMax - currentOffset)]
        let leftToCopy = t - tMax + currentOffset
        if 0 < leftToCopy {
            self.keys[0..., 0..., 0..<leftToCopy] = keys[0..., 0..., (tMax - currentOffset)...]
            self.values[0..., 0..., 0..<leftToCopy] = values[0..., 0..., (tMax - currentOffset)...]
        }
        self.offset += t
        return (self.keys, self.values)
    }

    func reset() {
        offset = 0
    }

    func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    func createAttentionMask(h: MLXArray) -> MLXArray? {
        let t = h.dim(1)
        let finalOffset = self.offset + t
        let finalOffsetMod = finalOffset % self.maxSize
        // As a default we use finalOffset + 1 so that these slices cannot be seen.
        var rinds = Array(repeating: Int32(finalOffset + 1), count: self.maxSize)
        for i in 0..<finalOffsetMod {
            rinds[i] = Int32(finalOffset + i - finalOffsetMod)
        }
        if finalOffsetMod != finalOffset {
            for i in finalOffsetMod..<rinds.count {
                rinds[i] = Int32(finalOffset + i - finalOffsetMod - rinds.count)
            }
        }
        let linds = MLXArray(Int32(self.offset)..<Int32(self.offset + t))
        let mask = linds[0..., .newAxis] .< MLXArray(rinds)[.newAxis]
        let res = (mask * Float32(-1e9)).asType(h.dtype)
        return res
    }
}
