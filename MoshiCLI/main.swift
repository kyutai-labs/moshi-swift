// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import MLX
import MLXNN
import MoshiLib

let homeDirectory = NSHomeDirectory()

func runTransformer() throws {
    let weights = try loadArrays(
        url: URL(fileURLWithPath: homeDirectory + "/tmp/model.safetensors"))
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
}

func runMic() {
    // Example Usage
    let microphoneCapture = MicrophoneCapture()
    microphoneCapture.startCapturing()
    sleep(100)
    // Call `microphoneCapture.stopCapturing()` when you're done.
}

try runMimi(dir: homeDirectory + "/tmp")
