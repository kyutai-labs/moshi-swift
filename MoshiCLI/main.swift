// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import MLX
import MLXNN
import MoshiLib

func runTransformer(baseDir: URL) throws {
    let weights = try loadArrays(url: baseDir.appendingPathComponent("model.safetensors"))
    print(weights.keys)
    let parameters = ModuleParameters.unflattened(weights)
    let cfg = LmConfig.moshi_2024_07()
    let model = LM(cfg)
    try model.update(parameters: parameters, verify: [.all])
    eval(model)
    print(model)
    let token_ids = MLXArray([42], [1, 1])
    let out = model(token_ids)
    print(out.shape, out.dtype, out.ndim)
}

func runMic(baseDir: URL) throws {
    let model = try makeMimi(baseDir: baseDir)
    let microphoneCapture = MicrophoneCapture()
    microphoneCapture.startCapturing()

    var allCodes: [MLXArray] = []
    var allPcms: [[Float]] = []
    var cnt = 0
    while let pcm = microphoneCapture.receive() {
        print("received audio data", pcm.count)
        allPcms.append(pcm)
        let pcm = MLXArray(pcm)[.newAxis, .newAxis]
        let codes = model.encodeStep(StreamArray(pcm))
        if let codes = codes.asArray() {
            codes.eval()
            allCodes.append(codes)
            print("converted to codes", codes.shape)
            if allCodes.count % 100 == 0 {
                let codes = concatenated(allCodes, axis: 2)
                cnt += 1
                try save(
                    arrays: ["codes": codes],
                    url: baseDir.appendingPathComponent("mic-codes\(cnt).safetensors"))
                let pcm = allPcms.flatMap { $0 }
                try writeWAVFile(
                    pcm,
                    sampleRate: 24000.0,
                    outputURL: baseDir.appendingPathComponent("mic-pcm\(cnt).wav"))
            }
        }
    }
    microphoneCapture.stopCapturing()
}

let baseDir = URL(fileURLWithPath: CommandLine.arguments[1])
// try runMimi(baseDir: baseDir)
// try runMic(baseDir: baseDir)
try runAsr(baseDir: baseDir, asrDelayInSteps: 25)
// try runAsrMic(baseDir: baseDir, asrDelayInSteps: 25)
