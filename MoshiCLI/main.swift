// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Foundation
import MLX
import MLXNN
import MoshiLib

let arr = MLXArray(stride(from: Int32(2), through: 8, by: 2), [2, 2])

print(arr)
print(arr.dtype)
print(arr.shape)
print(arr.ndim)
print(arr.asType(.int64))

// print a row
print(arr[1])

// print a value
print(arr[0, 1].item(Int32.self))

let weights = try loadArrays(
    url: URL(fileURLWithPath: "/Users/laurent/github/moshi-swift/model.safetensors"))
print(weights.keys)
let parameters = ModuleParameters.unflattened(weights)
let cfg = LmConfig.v0_1()
let model = LM(cfg)
try model.update(parameters: parameters, verify: [.all])
eval(model)
print(model)
let xs = MLXArray.zeros([1, 1, cfg.transformer.dModel])
let cache = model.transformer.makeCache()
print(model.transformer(xs, cache: cache))
