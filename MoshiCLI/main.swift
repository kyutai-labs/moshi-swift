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

let args = CommandLine.arguments
if args.count != 3 {
    fatalError("usage: \(args[0]) cmd dir")
}
let baseDir = URL(fileURLWithPath: args[2])

switch args[1] {
case "moshi-1b":
    try runMoshiMic("moshi-1b-299feac8@50.safetensors", baseDir: baseDir, cfg: LmConfig.moshi1b())
case "moshi-7b":
    try runMoshiMic("model.safetensors", baseDir: baseDir, cfg: LmConfig.moshi_2024_07())
case "moshi-1b-file":
    try runMoshi("moshi-1b-299feac8@50.safetensors", baseDir: baseDir, cfg: LmConfig.moshi1b())
case "moshi-7b-file":
    try runMoshi("model.safetensors", baseDir: baseDir, cfg: LmConfig.moshi_2024_07())
case "mimi":
    try runMimi(baseDir: baseDir, streaming: false)
case "mimi-streaming":
    try runMimi(baseDir: baseDir, streaming: true)
case "mic":
    try runMic(baseDir: baseDir)
case "asr-file":
    try runAsr(baseDir: baseDir, asrDelayInSteps: 25)
case "asr":
    try runAsrMic(baseDir: baseDir, asrDelayInSteps: 25)
case "code-to-audio-file":
    try runCodeToAudio(baseDir: baseDir, writeFile: true)
case "code-to-audio":
    try runCodeToAudio(baseDir: baseDir, writeFile: false)
case "transformer":
    try runTransformer(baseDir: baseDir)
case let other:
    fatalError("unknown command '\(other)'")
}
