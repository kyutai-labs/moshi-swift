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
    let targetURL = HubApi().localRepoLocation(Hub.Repo(id: id)).appending(path: filename)
    if FileManager.default.fileExists(atPath: targetURL.path) {
        print("using cached file \(targetURL.path)")
        return targetURL
    }
    var url: URL? = nil
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        let repo = Hub.Repo(id: id)
        url = try await Hub.snapshot(from: repo, matching: filename) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            print("\rretrieving \(filename): \(pct)%", terminator: "")
        }
        semaphore.signal()
    }
    semaphore.wait()
    print("\rretrieved \(filename)")
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
        url = try downloadFromHub(id: "kyutai/moshiko-mlx-q8", filename: "model.q8.safetensors")
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
        url = try downloadFromHub(id: "kyutai/moshiko-mlx-q8", filename: "model.q8.safetensors")
    } else {
        url = URL(fileURLWithPath: args[2])
    }
    try runMoshi(url, cfg: LmConfig.moshi_2024_07())
case "mimi":
    try runMimi(streaming: false)
case "mimi-streaming":
    try runMimi(streaming: true)
case "asr-file":
    let url: URL
    if args.count <= 2 {
        url = URL(fileURLWithPath: "asr-300m-f28fe6d5@450.safetensors")
    } else {
        url = URL(fileURLWithPath: args[2])
    }
    let weights = try loadArrays(url: url)
    let cfg =
        switch weights["out_norm.weight"]?.shape {
        case .none: fatalError("no out_norm.weight tensor in \(url)")
        case .some([1024]): LmConfig.asr300m()
        case .some([2048]): LmConfig.asr1b()
        case .some(let s): fatalError("unexpected shape for out_norm.weight \(s)")
        }
    try runAsr(url, cfg, asrDelayInSteps: 25)
case "asr":
    let url: URL
    if args.count <= 2 {
        url = URL(fileURLWithPath: "asr-300m-f28fe6d5@450.safetensors")
    } else {
        url = URL(fileURLWithPath: args[2])
    }
    let weights = try loadArrays(url: url)
    let cfg =
        switch weights["out_norm.weight"]?.shape {
        case .none: fatalError("no out_norm.weight tensor in \(url)")
        case .some([1024]): LmConfig.asr300m()
        case .some([2048]): LmConfig.asr1b()
        case .some(let s): fatalError("unexpected shape for out_norm.weight \(s)")
        }
    try runAsrMic(url, cfg, asrDelayInSteps: 25)
case "codes-to-audio-file":
    try runCodesToAudio(writeFile: true)
case "codes-to-audio":
    try runCodesToAudio(writeFile: false)
case "audio-to-codes":
    try runAudioToCodes()
case let other:
    fatalError("unknown command '\(other)'")
}
