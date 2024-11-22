// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Foundation
import MLX
import MLXNN
import MoshiLib

let weights = try loadArrays(
    url: URL(fileURLWithPath: "/Users/laurent/github/moshi-swift/model.safetensors"))
print(weights.keys)
let parameters = ModuleParameters.unflattened(weights)
let cfg = LmConfig.moshi_v0_1()
let model = LM(cfg)
try model.update(parameters: parameters, verify: [.all])
eval(model)
print(model)
let token_ids = MLXArray([42], [1, 1])
let out = model(token_ids)
print(out.shape, out.dtype, out.ndim)
