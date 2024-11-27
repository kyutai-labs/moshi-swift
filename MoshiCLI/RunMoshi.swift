// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import MLX
import MLXNN
import MoshiLib

func makeMoshi(_ filename: URL) throws -> LM {
    let weights = try loadArrays(url: filename)
    let parameters = ModuleParameters.unflattened(weights)
    let cfg = LmConfig.asr1b()
    let model = LM(cfg)
    try model.update(parameters: parameters, verify: [.all])
    eval(model)
    return model
}

func makeMoshiAsr(_ filename: URL) throws -> LM {
    let weights = try loadArrays(url: filename)
    let parameters = ModuleParameters.unflattened(weights)
    let cfg = LmConfig.asr1b()
    let model = LM(cfg)
    try model.update(parameters: parameters, verify: [.all])
    eval(model)
    return model
}

func loadVocab(_ fileURL: URL) throws -> [Int: String] {
    let jsonData = try Data(contentsOf: fileURL)
    let dictionary = try JSONDecoder().decode([Int: String].self, from: jsonData)
    return dictionary
}

func runMoshi(baseDir: URL) throws {
    let mimi = try makeMimi(baseDir: baseDir)
    let moshi = try makeMoshi(baseDir.appendingPathComponent("moshi-1b-1e20921d@50.safetensors"))
    let vocab = try loadVocab(baseDir.appendingPathComponent("tokenizer_spm_48k_multi6_2.json"))
    print("using device \(Device.defaultDevice().description)")
    let maxSteps = moshi.cfg.transformer.maxSeqLen
    let gen = LMGen(moshi, maxSteps: maxSteps, audioSampler: Sampler(), textSampler: Sampler())

    let pcm = readAudioToPCMArray(fileURL: baseDir.appendingPathComponent("bria-24khz.mp3"))!
    let chunkSize = 1920
    let codebooks = moshi.cfg.audioCodebooks
    for start in stride(from: 0, to: pcm.count, by: chunkSize) {
        let end = min(start + chunkSize, pcm.count)
        let pcmA = MLXArray(pcm[start..<end])[.newAxis, .newAxis]
        let codes = mimi.encodeStep(StreamArray(pcmA))
        if let codes = codes.asArray() {
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                if let textToken = gen.step(otherAudioTokens: codes[0..., 0..., step]) {
                    let textTokenI: Int = textToken[0].item()
                    if textTokenI != 0 && textTokenI != 3 {
                        if var v = vocab[textTokenI] {
                            v.replace("▁", with: " ")
                            print(v, terminator: "")
                            fflush(stdout)
                        }
                    }
                }
                if let audioTokens = gen.lastAudioTokens() {
                    // TODO: store the resulting audio
                }
            }
        }
    }
    print()
}

func runAsr(baseDir: URL, asrDelayInSteps: Int) throws {
    let mimi = try makeMimi(baseDir: baseDir)
    let moshi = try makeMoshiAsr(baseDir.appendingPathComponent("asr-1b-8d2516b9@150.safetensors"))
    let vocab = try loadVocab(baseDir.appendingPathComponent("tokenizer_spm_48k_multi6_2.json"))
    print("using device \(Device.defaultDevice().description)")

    let pcm = readAudioToPCMArray(fileURL: baseDir.appendingPathComponent("bria-24khz.mp3"))!
    let chunkSize = 1920
    let sampler = Sampler(temp: 0.0)
    let codebooks = moshi.cfg.audioCodebooks
    var prevTextToken = moshi.cfg.textInitToken()
    do {
        let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
        let audioIds = (0..<16).map { _ in MLXArray([moshi.cfg.audioPaddingToken()]) }
        let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
        let (textToken, _) = sampler(logits: textLogits)
        let textTokenI: Int = textToken[0].item()
        print("sampled first", textTokenI)
        prevTextToken = textTokenI
    }
    var cnt = 0
    for start in stride(from: 0, to: pcm.count, by: chunkSize) {
        let end = min(start + chunkSize, pcm.count)
        let pcmA = MLXArray(pcm[start..<end])[.newAxis, .newAxis]
        let codes = mimi.encodeStep(StreamArray(pcmA))
        if let codes = codes.asArray() {
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                var textIds: MLXArray? = nil
                if asrDelayInSteps < cnt {
                    textIds = MLXArray([prevTextToken]).reshaped([1, 1])
                }
                let audioIds = (0..<codebooks).map { codes[0..., $0, step].reshaped(1, 1) }
                let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
                let (textToken, _) = sampler(logits: textLogits)
                let textTokenI: Int = textToken[0].item()
                if textTokenI != 0 && textTokenI != 3 && asrDelayInSteps <= cnt {
                    if var v = vocab[textTokenI] {
                        v.replace("▁", with: " ")
                        print(v, terminator: "")
                        fflush(stdout)
                    }
                }
                prevTextToken = textTokenI
                cnt += 1
            }
        }
    }
    print()
}

func runAsrMic(baseDir: URL, asrDelayInSteps: Int) throws {
    let mimi = try makeMimi(baseDir: baseDir)
    let moshi = try makeMoshiAsr(baseDir.appendingPathComponent("asr-1b-8d2516b9@150.safetensors"))
    let vocab = try loadVocab(baseDir.appendingPathComponent("tokenizer_spm_48k_multi6_2.json"))
    print("using device \(Device.defaultDevice().description)")

    let sampler = Sampler(temp: 0.0)
    let codebooks = moshi.cfg.audioCodebooks
    var prevTextToken = moshi.cfg.textInitToken()
    do {
        let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
        let audioIds = (0..<16).map { _ in MLXArray([moshi.cfg.audioPaddingToken()]) }
        let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
        let (textToken, _) = sampler(logits: textLogits)
        let textTokenI: Int = textToken[0].item()
        print("sampled first", textTokenI)
        prevTextToken = textTokenI
    }

    let microphoneCapture = MicrophoneCapture()
    microphoneCapture.startCapturing()

    var cnt = 0
    while let pcm = microphoneCapture.receive() {
        let pcm = MLXArray(pcm)[.newAxis, .newAxis]
        let codes = mimi.encodeStep(StreamArray(pcm))
        if let codes = codes.asArray() {
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                var textIds: MLXArray? = nil
                if asrDelayInSteps < cnt {
                    textIds = MLXArray([prevTextToken]).reshaped([1, 1])
                }
                let audioIds = (0..<codebooks).map { codes[0..., $0, step].reshaped(1, 1) }
                let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
                let (textToken, _) = sampler(logits: textLogits)
                let textTokenI: Int = textToken[0].item()
                if textTokenI != 0 && textTokenI != 3 && asrDelayInSteps <= cnt {
                    if var v = vocab[textTokenI] {
                        v.replace("▁", with: " ")
                        print(v, terminator: "")
                        fflush(stdout)
                    }
                }
                prevTextToken = textTokenI
                cnt += 1
            }
        }
    }
    microphoneCapture.stopCapturing()
}
