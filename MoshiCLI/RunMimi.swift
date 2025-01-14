// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import MoshiLib

func makeMimi(numCodebooks: Int) throws -> Mimi {
    let cfg = MimiConfig.mimi_2024_07(numCodebooks: numCodebooks)
    let model = Mimi(cfg, bSize: 1)

    let url = try downloadFromHub(
        id: "lmz/moshi-swift",
        filename: "tokenizer-dbaa9758-checkpoint125.safetensors")
    let origWeights = try loadArrays(url: url)
    var weights: [String: MLXArray] = [:]
    for (var key, var weight) in origWeights {
        // Mutating the keys while iterating over the map seems pretty dodgy, not sure what the idiomatic
        // way to do this is in swift. Hopefully this is copy on write and it's all good :)
        if key.hasPrefix("encoder.model") {
            key.replace("encoder.model.", with: "encoder.")
        }
        if key.hasPrefix("decoder.model") {
            key.replace("decoder.model.", with: "decoder.")
        }
        if key.hasSuffix(".in_proj_weight") {
            key.replace(".in_proj_weight", with: ".in_proj.weight")
        }
        if key.hasSuffix(".linear1.weight") {
            key.replace(".linear1.weight", with: ".gating.linear1.weight")
        }
        if key.hasSuffix(".linear2.weight") {
            key.replace(".linear2.weight", with: ".gating.linear2.weight")
        }
        // Awfully hardcoded matching between the pytorch layers and their mlx equivalent :(
        for (layerIdx, decoderIdx) in [2, 5, 8, 11].enumerated() {
            key.replace("decoder.\(decoderIdx).", with: "decoder.layers.\(layerIdx).upsample.")
            key.replace(
                "decoder.\(decoderIdx + 1).", with: "decoder.layers.\(layerIdx).residuals.0.")
        }
        for (layerIdx, encoderIdx) in [1, 4, 7, 10].enumerated() {
            key.replace("encoder.\(encoderIdx).", with: "encoder.layers.\(layerIdx).residuals.0.")
            key.replace(
                "encoder.\(encoderIdx + 2).", with: "encoder.layers.\(layerIdx).downsample.")
        }
        key.replace("decoder.0.", with: "decoder.init_conv1d.")
        key.replace("decoder.14.", with: "decoder.final_conv1d.")
        key.replace("encoder.0.", with: "encoder.init_conv1d.")
        key.replace("encoder.14.", with: "encoder.final_conv1d.")
        key.replace(".block.1.", with: ".block.0.")
        key.replace(".block.3.", with: ".block.1.")
        // PyTorch layout for conv weights is outC, inC, kSize, for MLX it's outC, kSize, inC
        if key.hasSuffix(".conv.weight") || key.hasSuffix(".output_proj.weight")
            || key.hasSuffix(".input_proj.weight")
        {
            weight = weight.swappedAxes(-1, -2)
        }
        // PyTorch layout for conv-transposed weights is inC, outC, kSize, for MLX it's outC, kSize, inC
        if key.hasSuffix(".convtr.weight") {
            weight = weight.transposed(axes: [1, 2, 0])
        }

        // print(key, weight.shape)
        weights[key] = weight
    }
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])
    return model
}

func runMimi(streaming: Bool, audioFile: URL?, channel: Int = 0) throws {
    let model = try makeMimi(numCodebooks: 16)
    print("using device \(Device.defaultDevice().description)")
    let sampleURL =
        switch audioFile {
        case .none: try downloadFromHub(id: "lmz/moshi-swift", filename: "bria-24khz.mp3")
        case .some(let url): url
        }
    let pcm = readAudioToPCMArray(fileURL: sampleURL, channel: channel)!

    if streaming {
        let chunkSize = 1920
        var pcmOuts: [[Float]] = []
        var elapsedTimes: [Double] = []
        var nSteps = 0
        for start in stride(from: 0, to: pcm.count, by: chunkSize) {
            let startTime = CFAbsoluteTimeGetCurrent()
            let pct = 100 * start / pcm.count
            let end = min(start + chunkSize, pcm.count)
            let pcmA = MLXArray(pcm[start..<end])[.newAxis, .newAxis]
            let codes = model.encodeStep(StreamArray(pcmA))
            if start == 0 {
                print(codes.asArray()?[0, 0..., 0].asArray(Int.self), "\n", codes.shape)
            }
            let pcmOut = model.decodeStep(codes)
            if let p = pcmOut.asArray() {
                let p: [Float] = p[0, 0].asArray(Float.self)
                pcmOuts.append(p)
            }
            let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime
            elapsedTimes.append(elapsedTime)
            nSteps += 1
            if nSteps % 10 == 0 {
                print("\rprocessing \(pct)%", terminator: "")
                fflush(stdout)
            }
        }
        do {
            let elapsedTimes = elapsedTimes[1...]
            let avgTimeMs = elapsedTimes.reduce(0, +) / Double(elapsedTimes.count) * 1000.0
            let minTimeMs = elapsedTimes.min()! * 1000.0
            let maxTimeMs = elapsedTimes.max()! * 1000.0
            print(
                "\r\(nSteps) steps, avg time \(Int(avgTimeMs))ms, min \(Int(minTimeMs))ms, max \(Int(maxTimeMs))ms"
            )
        }
        try writeWAVFile(
            pcmOuts.flatMap { $0 },
            sampleRate: 24000,
            outputURL: URL(fileURLWithPath: "bria-out.wav"))
    } else {
        let pcmA = MLXArray(pcm)[.newAxis, .newAxis, 0..<240000]
        print("pcm loaded from file", pcmA.shape, pcmA.dtype)
        let out = model.encode(pcmA)
        print("quantized", out.shape)
        try save(
            arrays: ["codes": out],
            url: URL(fileURLWithPath: "bria-codes.safetensors"))
        print(out)
        let pcmOut = model.decode(out)
        print("pcm generated", pcmOut.shape)
        let pcmOutA: [Float] = pcmOut[0, 0].asArray(Float.self)
        print("data extracted")
        try writeWAVFile(
            pcmOutA,
            sampleRate: 24000,
            outputURL: URL(fileURLWithPath: "bria-out.wav"))
    }
}

public func runCodesToAudio(writeFile: Bool) throws {
    let model = try makeMimi(numCodebooks: 16)
    print("using device \(Device.defaultDevice().description)")

    let codes = try loadArrays(
        url: URL(fileURLWithPath: "moshi-codes.safetensors"))["codes"]!
    print("loaded codes with shape", codes.shape)
    let (_, _, steps) = codes.shape3

    if writeFile {
        var pcmOuts: [[Float]] = []
        for stepIdx in 0..<steps {
            let codes = codes[0..., 0..., stepIdx...stepIdx]
            let pcmOut = model.decodeStep(StreamArray(codes))
            if let p = pcmOut.asArray() {
                let p: [Float] = p[0, 0].asArray(Float.self)
                pcmOuts.append(p)
            }
        }
        try writeWAVFile(
            pcmOuts.flatMap { $0 },
            sampleRate: 24000,
            outputURL: URL(fileURLWithPath: "mimi-out.wav"))
    } else {
        let player = AudioPlayer(sampleRate: 24000)
        try player.startPlaying()
        for stepIdx in 0..<steps {
            let codes = codes[0..., 0..., stepIdx...stepIdx]
            let pcmOut = model.decodeStep(StreamArray(codes))
            if let p = pcmOut.asArray() {
                let p: [Float] = p[0, 0].asArray(Float.self)
                while !player.send(p) {
                    usleep(1000)
                }
            }
        }
    }
}

public func runAudioToCodes() throws {
    let model = try makeMimi(numCodebooks: 16)
    print("using device \(Device.defaultDevice().description)")

    let microphoneCapture = MicrophoneCapture()
    microphoneCapture.startCapturing()
    var cnt = 0
    var allCodes: [MLXArray] = []
    while let pcm = microphoneCapture.receive() {
        let pcm = MLXArray(pcm)[.newAxis, .newAxis]
        let codes = model.encodeStep(StreamArray(pcm))
        if let codes = codes.asArray() {
            eval(codes)
            allCodes.append(codes)
            cnt += codes.count
            if cnt > 125 {
                break
            }
        }
    }
    try save(
        arrays: ["codes": concatenated(allCodes, axis: -1)],
        url: URL(fileURLWithPath: "mimi-codes.safetensors"))
    microphoneCapture.stopCapturing()

}
