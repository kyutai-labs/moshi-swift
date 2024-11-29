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
if args.count < 2 {
    fatalError("usage: \(args[0]) cmd")
}
switch args[1] {
case "moshi-1b":
    let url: URL
    if args.count <= 2 {
        url = URL(fileURLWithPath: "moshi-1b-299feac8@50.safetensors")
    } else {
        url = URL(fileURLWithPath: args[2])
    }
    try runMoshiMic(url, cfg: LmConfig.moshi1b())
case "moshi-7b":
    let url: URL
    if args.count <= 2 {
        print("downloading the weights from the hub, this may take a while...")
        url = try downloadFromHub(id: "kyutai/moshiko-mlx-q4", filename: "model.q4.safetensors")
    } else {
        url = URL(fileURLWithPath: args[2])
    }
    try runMoshiMic(url, cfg: LmConfig.moshi_2024_07())
case "moshi-1b-file":
    let url: URL
    if args.count <= 2 {
        url = URL(fileURLWithPath: "moshi-1b-299feac8@50.safetensors")
    } else {
        url = URL(fileURLWithPath: args[2])
    }
    try runMoshi(url, cfg: LmConfig.moshi1b())
case "moshi-7b-file":
    let url: URL
    if args.count <= 2 {
        print("downloading the weights from the hub, this may take a while...")
        url = try downloadFromHub(id: "kyutai/moshiko-mlx-q4", filename: "model.q4.safetensors")
    } else {
        url = URL(fileURLWithPath: args[2])
    }
    try runMoshi(url, cfg: LmConfig.moshi_2024_07())
case "mimi":
    try runMimi(streaming: false)
case "mimi-streaming":
    try runMimi(streaming: true)
case "asr-file":
    let url = URL(fileURLWithPath: "asr-1b-8d2516b9@150.safetensors")
    try runAsr(url, asrDelayInSteps: 25)
case "asr":
    let url = URL(fileURLWithPath: "asr-1b-8d2516b9@150.safetensors")
    try runAsrMic(url, asrDelayInSteps: 25)
case "codes-to-audio-file":
    try runCodesToAudio(writeFile: true)
case "codes-to-audio":
    try runCodesToAudio(writeFile: false)
case "audio-to-codes":
    try runAudioToCodes()
case let other:
    fatalError("unknown command '\(other)'")
}
