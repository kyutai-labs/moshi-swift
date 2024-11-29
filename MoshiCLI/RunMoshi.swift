// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import MLX
import MLXNN
import MoshiLib
import os.signpost

enum EventKind {
    case beginStep
    case endStep
    case beginDepformer
    case endDepformer
    case beginDecode
    case endDecode
    case beginEncode
    case endEncode
}

struct ChromeTraceEvent: Codable {
    let name: String
    let cat: String
    let ph: String
    let ts: Int
    let pid: Int
    let tid: Int
}

class PerfStats {
    private let log: OSLog
    private var events: [(CFAbsoluteTime, EventKind)] = []

    init() {
        self.log = OSLog(subsystem: "org.kyutai.moshi", category: "Performance")
    }

    func append(_ kind: EventKind) {
        events.append((CFAbsoluteTimeGetCurrent(), kind))
    }

    func beginStep() {
        os_signpost(.begin, log: log, name: "step")
        append(.beginStep)
    }

    func endStep() {
        os_signpost(.end, log: log, name: "step")
        append(.endStep)
    }

    func beginDepformer() {
        os_signpost(.begin, log: log, name: "depformer")
        append(.beginDepformer)
    }

    func endDepformer() {
        os_signpost(.end, log: log, name: "depformer")
        append(.endDepformer)
    }

    func beginEncode() {
        os_signpost(.begin, log: log, name: "encode")
        append(.beginEncode)
    }

    func endEncode() {
        os_signpost(.end, log: log, name: "encode")
        append(.endEncode)
    }

    func beginDecode() {
        os_signpost(.begin, log: log, name: "decode")
        append(.beginDecode)
    }

    func endDecode() {
        os_signpost(.end, log: log, name: "decode")
        append(.endDecode)
    }

    func writeJSONTrace(url: URL) throws {
        let encoder = JSONEncoder()
        var traceEvents: [ChromeTraceEvent] = []
        for (time, kind) in events {
            let ts = Int((time - events[0].0) * 1e6)
            let (name, ph) =
                switch kind {
                case .beginStep: ("step", "B")
                case .endStep: ("step", "E")
                case .beginEncode: ("encode", "B")
                case .endEncode: ("encode", "E")
                case .beginDepformer: ("depformer", "B")
                case .endDepformer: ("depformer", "E")
                case .beginDecode: ("decode", "B")
                case .endDecode: ("decode", "E")
                }
            traceEvents.append(
                ChromeTraceEvent(name: name, cat: "", ph: ph, ts: ts, pid: 42, tid: 1))
        }
        let jsonData = try encoder.encode(traceEvents)
        try jsonData.write(to: url)
    }
}

func makeMoshi(_ url: URL, _ cfg: LmConfig) throws -> LM {
    let weights = try loadArrays(url: url)
    let parameters = ModuleParameters.unflattened(weights)
    let model = LM(cfg)
    if url.lastPathComponent.hasSuffix(".q4.safetensors") {
        quantize(model: model, groupSize: 32, bits: 4)
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
        case let other: fatalError("unexpected text vocab size \(other)")
        }
    let fileURL = try downloadFromHub(id: "lmz/moshi-swift", filename: filename)
    let jsonData = try Data(contentsOf: fileURL)
    let dictionary = try JSONDecoder().decode([Int: String].self, from: jsonData)
    return dictionary
}

func runMoshiMic(_ filename: String, baseDir: URL, cfg: LmConfig) throws {
    let mimi = try makeMimi()
    let moshi = try makeMoshi(baseDir.appendingPathComponent(filename), cfg)
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
                        player.send(p.asArray(Float.self))
                    }
                }
            }
        }
    }
    print()
    microphoneCapture.stopCapturing()
}

func runMoshi(_ filename: String, baseDir: URL, cfg: LmConfig) throws {
    let stats = PerfStats()
    let mimi = try makeMimi()
    let moshi = try makeMoshi(baseDir.appendingPathComponent(filename), cfg)
    let vocab = try loadVocab(cfg)
    print("warming up mimi")
    mimi.warmup()
    print("warming up moshi")
    moshi.warmup()
    print("done warming up")

    let maxSteps = moshi.cfg.transformer.maxSeqLen
    let gen = LMGen(moshi, maxSteps: maxSteps, audioSampler: Sampler(), textSampler: Sampler())

    let pcm = readAudioToPCMArray(fileURL: baseDir.appendingPathComponent("bria-24khz.mp3"))!
    let chunkSize = 1920
    var pcmOuts: [[Float]] = []
    var allAudioTokens: [MLXArray] = []
    for start in stride(from: 0, to: pcm.count, by: chunkSize) {
        let end = min(start + chunkSize, pcm.count)
        let pcmA = MLXArray(pcm[start..<end])[.newAxis, .newAxis]
        stats.beginEncode()
        let codes = mimi.encodeStep(StreamArray(pcmA))
        if let codes = codes.asArray() {
            eval(codes)
        }
        stats.endEncode()
        if let codes = codes.asArray() {
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                stats.beginStep()
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
                stats.endStep()
                stats.beginDepformer()
                let audioTokens = gen.lastAudioTokens()
                stats.endDepformer()
                if let audioTokens = audioTokens {
                    let audioTokens = audioTokens[0..., 0..., .newAxis]
                    allAudioTokens.append(audioTokens)
                    stats.beginDecode()
                    let pcmOut = mimi.decodeStep(StreamArray(audioTokens))
                    if let p = pcmOut.asArray() {
                        let p: [Float] = p[0, 0].asArray(Float.self)
                        pcmOuts.append(p)
                    }
                    stats.endDecode()
                }
            }
        }
    }
    print()
    try save(
        arrays: ["codes": concatenated(allAudioTokens, axis: -1)],
        url: baseDir.appendingPathComponent("moshi-codes.safetensors"))
    try stats.writeJSONTrace(url: baseDir.appendingPathComponent("moshi-trace.json"))
    try writeWAVFile(
        pcmOuts.flatMap { $0 },
        sampleRate: 24000,
        outputURL: baseDir.appendingPathComponent("moshi-out.wav"))
}

func runAsr(baseDir: URL, asrDelayInSteps: Int) throws {
    let mimi = try makeMimi()
    let cfg = LmConfig.asr1b()
    let moshi = try makeMoshi(
        baseDir.appendingPathComponent("asr-1b-8d2516b9@150.safetensors"), cfg)
    let vocab = try loadVocab(cfg)
    print("using device \(Device.defaultDevice().description)")
    print("warming up mimi")
    mimi.warmup()
    print("warming up moshi")
    moshi.warmup()
    print("done warming up")

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
    let mimi = try makeMimi()
    let cfg = LmConfig.asr1b()
    let moshi = try makeMoshi(
        baseDir.appendingPathComponent("asr-1b-8d2516b9@150.safetensors"), cfg)
    let vocab = try loadVocab(cfg)
    print("using device \(Device.defaultDevice().description)")
    print("warming up mimi")
    mimi.warmup()
    print("warming up moshi")
    moshi.warmup()
    print("done warming up")

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
