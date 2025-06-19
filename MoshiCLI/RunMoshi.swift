// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import MLX
import MLXNN
import MoshiLib

func makeMoshi(_ url: URL, _ cfg: LmConfig) throws -> LM {
    let weights = try loadArrays(url: url)
    let parameters = ModuleParameters.unflattened(weights)
    let model = LM(cfg, bSize: 1)
    if url.lastPathComponent.hasSuffix(".q4.safetensors") {
        quantize(model: model, groupSize: 32, bits: 4)
    } else if url.lastPathComponent.hasSuffix(".q6.safetensors") {
        quantize(model: model, groupSize: 64, bits: 6)
    } else if url.lastPathComponent.hasSuffix(".q8.safetensors") {
        quantize(model: model, groupSize: 64, bits: 8)
    }
    try model.update(parameters: parameters, verify: [.all])
    eval(model)
    return model
}

func loadVocab(_ cfg: LmConfig) throws -> [Int: String] {
    let filename =
        switch cfg.textOutVocabSize {
        case 48000: "tokenizer_spm_48k_multi6_2.json"
        case 32000: "tokenizer_spm_32k_3.json"
        case 8000: "tokenizer_spm_8k_0.json"
        case 4000: "test_en_audio_4000.json"
        case let other: fatalError("unexpected text vocab size \(other)")
        }
    let fileURL = try downloadFromHub(id: "lmz/moshi-swift", filename: filename)
    let jsonData = try Data(contentsOf: fileURL)
    let dictionary = try JSONDecoder().decode([Int: String].self, from: jsonData)
    return dictionary
}

func runMoshiMic(_ url: URL, cfg: LmConfig) throws {
    let mimi = try makeMimi(numCodebooks: 16)
    let moshi = try makeMoshi(url, cfg)
    let vocab = try loadVocab(cfg)
    print("using device \(Device.defaultDevice().description)")
    print("warming up mimi")
    mimi.warmup()
    print("warming up moshi")
    moshi.warmup()
    print("done warming up")

    let maxSteps = moshi.cfg.transformer.maxSeqLen
    let gen = LMGen(moshi, maxSteps: maxSteps, audioSampler: Sampler(), textSampler: Sampler())

    let microphoneCapture = MicrophoneCapture()
    microphoneCapture.startCapturing()
    let player = AudioPlayer(sampleRate: 24000)
    try player.startPlaying()
    print("started the audio loops")

    while let pcm = microphoneCapture.receive() {
        let pcm = MLXArray(pcm)[.newAxis, .newAxis]
        let codes = mimi.encodeStep(StreamArray(pcm))
        if let codes = codes.asArray() {
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                if let textToken = gen.step(otherAudioTokens: codes[0..., 0..<8, step]) {
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
                    let pcmOut = mimi.decodeStep(StreamArray(audioTokens[0..., 0..., .newAxis]))
                    if let p = pcmOut.asArray() {
                        let _ = player.send(p.asArray(Float.self))
                    }
                }
            }
        }
    }
    print()
    microphoneCapture.stopCapturing()
}

func runMoshi(_ url: URL, cfg: LmConfig, audioFile: URL?, channel: Int = 0) throws {
    let stats = PerfStats()
    let mimi = try makeMimi(numCodebooks: 16)
    let moshi = try makeMoshi(url, cfg)
    let vocab = try loadVocab(cfg)
    print("warming up mimi")
    mimi.warmup()
    print("warming up moshi")
    moshi.warmup()
    print("done warming up")

    let maxSteps = moshi.cfg.transformer.maxSeqLen
    let gen = LMGen(
        moshi, maxSteps: maxSteps, audioSampler: Sampler(), textSampler: Sampler(), cb: stats)

    let sampleURL =
        switch audioFile {
        case .none: try downloadFromHub(id: "lmz/moshi-swift", filename: "bria-24khz.mp3")
        case .some(let url): url
        }
    let pcm = readAudioToPCMArray(fileURL: sampleURL, channel: channel)!
    let chunkSize = 1920
    var pcmOuts: [[Float]] = []
    var allAudioTokens: [MLXArray] = []
    for start in stride(from: 0, to: pcm.count, by: chunkSize) {
        let end = min(start + chunkSize, pcm.count)
        let pcmA = MLXArray(pcm[start..<end])[.newAxis, .newAxis]
        stats.onEvent(.beginEncode)
        let codes = mimi.encodeStep(StreamArray(pcmA))
        if let codes = codes.asArray() {
            eval(codes)
        }
        stats.onEvent(.endEncode)
        if let codes = codes.asArray() {
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                let textToken = gen.step(otherAudioTokens: codes[0..., 0..<8, step])
                if let textToken = textToken {
                    let textTokenI: Int = textToken[0].item()
                    if textTokenI != 0 && textTokenI != 3 {
                        if var v = vocab[textTokenI] {
                            v.replace("▁", with: " ")
                            print(v, terminator: "")
                            fflush(stdout)
                        }
                    }
                }
                let audioTokens = gen.lastAudioTokens()
                if let audioTokens = audioTokens {
                    let audioTokens = audioTokens[0..., 0..., .newAxis]
                    allAudioTokens.append(audioTokens)
                    stats.onEvent(.beginDecode)
                    let pcmOut = mimi.decodeStep(StreamArray(audioTokens))
                    if let p = pcmOut.asArray() {
                        let p: [Float] = p[0, 0].asArray(Float.self)
                        pcmOuts.append(p)
                    }
                    stats.onEvent(.endDecode)
                }
            }
        }
    }
    print()
    try save(
        arrays: ["codes": concatenated(allAudioTokens, axis: -1)],
        url: URL(fileURLWithPath: "moshi-codes.safetensors"))
    try stats.writeJSONTrace(url: URL(fileURLWithPath: "moshi-trace.json"))
    try writeWAVFile(
        pcmOuts.flatMap { $0 },
        sampleRate: 24000,
        outputURL: URL(fileURLWithPath: "moshi-out.wav"))
}

func runAsr(_ url: URL, _ cfg: LmConfig, audioFile: URL?, channel: Int) throws {
    let mimi = try makeMimi(numCodebooks: 32)
    let moshi = try makeMoshi(url, cfg)
    let vocab = try loadVocab(cfg)
    print("using device \(Device.defaultDevice().description)")
    print("warming up mimi")
    mimi.warmup()
    print("warming up moshi")
    moshi.warmup()
    print("done warming up")
    let asr = ASR(moshi, mimi, vocab: vocab)
    asr.reset()

    let sampleURL =
        switch audioFile {
        case .none: try downloadFromHub(id: "lmz/moshi-swift", filename: "bria-24khz.mp3")
        case .some(let url): url
        }
    let pcm = readAudioToPCMArray(fileURL: sampleURL, channel: channel)!
    let chunkSize = 1920
    asr.reset()
    for start in stride(from: 0, to: pcm.count, by: chunkSize) {
        let end = min(start + chunkSize, pcm.count)
        let pcmA = MLXArray(pcm[start..<end])[.newAxis, .newAxis]
        let tokens = asr.onPcmInput(pcmA)
        for token in tokens {
            print(token, terminator: "")
            fflush(stdout)
        }
    }
    print()
}

func runAsrMic(_ url: URL, _ cfg: LmConfig) throws {
    let mimi = try makeMimi(numCodebooks: 32)
    let moshi = try makeMoshi(url, cfg)
    let vocab = try loadVocab(cfg)
    print("using device \(Device.defaultDevice().description)")
    print("warming up mimi")
    mimi.warmup()
    print("warming up moshi")
    moshi.warmup()
    print("done warming up")
    let asr = ASR(moshi, mimi, vocab: vocab)
    asr.reset()

    let microphoneCapture = MicrophoneCapture()
    microphoneCapture.startCapturing()

    while let pcm = microphoneCapture.receive() {
        let pcm = MLXArray(pcm)[.newAxis, .newAxis]
        let tokens = asr.onPcmInput(pcm)
        for token in tokens {
            print(token, terminator: "")
            fflush(stdout)
        }
    }
    microphoneCapture.stopCapturing()
}
