// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import MoshiLib

func downloadFromHub(id: String, filename: String) throws -> URL {
    var url: URL? = nil
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        let repo = Hub.Repo(id: id)
        url = try await Hub.snapshot(from: repo, matching: filename)
        semaphore.signal()
    }
    semaphore.wait()
    return url!.appending(path: filename)
}

let args = CommandLine.arguments
if args.count < 3 {
    fatalError("usage: \(args[0]) cmd dir")
}
let baseDir = URL(fileURLWithPath: args[2])

switch args[1] {
case "moshi-1b":
    let fileName = args.count <= 3 ? "moshi-1b-299feac8@50.safetensors" : args[3]
    try runMoshiMic(fileName, baseDir: baseDir, cfg: LmConfig.moshi1b())
case "moshi-7b":
    let fileName = args.count <= 3 ? "model.safetensors" : args[3]
    try runMoshiMic(fileName, baseDir: baseDir, cfg: LmConfig.moshi_2024_07())
case "moshi-1b-file":
    let fileName = args.count <= 3 ? "moshi-1b-299feac8@50.safetensors" : args[3]
    try runMoshi(fileName, baseDir: baseDir, cfg: LmConfig.moshi1b())
case "moshi-7b-file":
    let fileName = args.count <= 3 ? "model.safetensors" : args[3]
    try runMoshi(fileName, baseDir: baseDir, cfg: LmConfig.moshi_2024_07())
case "mimi":
    try runMimi(baseDir: baseDir, streaming: false)
case "mimi-streaming":
    try runMimi(baseDir: baseDir, streaming: true)
case "asr-file":
    try runAsr(baseDir: baseDir, asrDelayInSteps: 25)
case "asr":
    try runAsrMic(baseDir: baseDir, asrDelayInSteps: 25)
case "codes-to-audio-file":
    try runCodesToAudio(baseDir: baseDir, writeFile: true)
case "codes-to-audio":
    try runCodesToAudio(baseDir: baseDir, writeFile: false)
case "audio-to-codes":
    try runAudioToCodes(baseDir: baseDir)
case let other:
    fatalError("unknown command '\(other)'")
}
