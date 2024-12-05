// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Hub
import MLX
import MLXNN
import MLXRandom
import Metal
import MoshiLib
import SwiftUI
import Synchronization

struct ContentView: View {
    @State var model = Evaluator()
    @Environment(DeviceStat.self) private var deviceStat

    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text(model.modelInfo)
            if model.progress != nil {
                ProgressView(model.progress!)
            }
            if !model.running {
                Button("Start Moshi", action: generate)
            } else {
                Button("Stop", action: stopGenerate)
            }
            if !model.running {
                Button("Start Mimi", action: generateMimi)
            } else {
                Button("Stop", action: stopGenerate)
            }
            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Group {
                        Text(model.output)
                            .textSelection(.enabled)
                    }
                    .onChange(of: model.output) { _, _ in
                        sp.scrollTo("bottom")
                    }
                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }
        }
        .padding()
        .toolbar {
            ToolbarItem {
                Label(
                    "Memory Usage: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))",
                    systemImage: "info.circle.fill"
                )
                .labelStyle(.titleAndIcon)
                .padding(.horizontal)
                .help(
                    Text(
                        """
                        Active Memory: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))/\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
                        Cache Memory: \(deviceStat.gpuUsage.cacheMemory.formatted(.byteCount(style: .memory)))/\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
                        Peak Memory: \(deviceStat.gpuUsage.peakMemory.formatted(.byteCount(style: .memory)))
                        """
                    )
                )
            }
        }
    }

    private func generate() {
        Task {
            await model.generate(moshi: true)
        }
    }

    private func generateMimi() {
        Task {
            await model.generate(moshi: false)
        }
    }

    private func stopGenerate() {
        Task {
            await model.stopGenerate()
        }
    }
}

#Preview {
    ContentView()
}

@Observable
@MainActor
class Evaluator {
    var running = false
    var modelInfo = "...moshi..."
    var stat = ""
    var output = ""
    var progress: Progress? = nil
    let shouldStop: Atomic<Bool> = .init(false)

    enum LoadState {
        case idle
        case loaded(ModelState)
    }

    var loadState = LoadState.idle

    func downloadFromHub(id: String, filename: String) async throws -> URL {
        let downloadDir = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let api = HubApi(downloadBase: downloadDir)
        let repo = Hub.Repo(id: id)
        let targetURL = api.localRepoLocation(repo).appending(path: filename)
        if FileManager.default.fileExists(atPath: targetURL.path) {
            print("using cached file \(targetURL.path)")
            return targetURL
        }
        let url = try await api.snapshot(from: repo, matching: filename) { progress in
            Task { @MainActor in
                self.progress = progress
            }
        }
        // TODO: also set this back to nil on errors.
        self.progress = nil
        return url.appending(path: filename)
    }

    func loadVocab(_ cfg: LmConfig) async throws -> [Int: String] {
        let filename =
            switch cfg.textOutVocabSize {
            case 48000: "tokenizer_spm_48k_multi6_2.json"
            case 32000: "tokenizer_spm_32k_3.json"
            case 8000: "tokenizer_spm_8k_0.json"
            case let other: fatalError("unexpected text vocab size \(other)")
            }
        let fileURL = try await downloadFromHub(id: "lmz/moshi-swift", filename: filename)
        let jsonData = try Data(contentsOf: fileURL)
        let dictionary = try JSONDecoder().decode([Int: String].self, from: jsonData)
        return dictionary
    }

    func makeMoshi(_ url: URL, _ cfg: LmConfig) throws -> LM {
        let weights = try loadArrays(url: url)
        let parameters = ModuleParameters.unflattened(weights)
        let model = LM(cfg, bSize: 1)
        if url.lastPathComponent.hasSuffix(".q4.safetensors") {
            quantize(model: model, groupSize: 32, bits: 4)
        } else if url.lastPathComponent.hasSuffix(".q8.safetensors") {
            quantize(model: model, groupSize: 64, bits: 8)
        }
        try model.update(parameters: parameters, verify: [.all])
        eval(model)
        return model
    }

    func makeMimi() async throws -> Mimi {
        let cfg = MimiConfig.mimi_2024_07()
        let model = Mimi(cfg, bSize: 1)

        let url = try await downloadFromHub(
            id: "kyutai/moshiko-mlx-q4",
            filename: "tokenizer-e351c8d8-checkpoint125.safetensors")
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
                key.replace(
                    "encoder.\(encoderIdx).", with: "encoder.layers.\(layerIdx).residuals.0.")
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

    func stopGenerate() async {
        self.shouldStop.store(true, ordering: .relaxed)
    }

    func generate(moshi: Bool) async {
        guard !running else { return }

        self.shouldStop.store(false, ordering: .relaxed)
        self.modelInfo = "starting"
        self.output = ""
        running = true
        do {
            let model = try await load(moshi: moshi)
            try await model.perform { model in
                model.reset()
                // TODO: Do not create a fresh audio input/output on each session.
                let microphoneCapture = MicrophoneCapture()
                microphoneCapture.startCapturing()
                let ap = AudioPlayer(sampleRate: 24000)
                try ap.startPlaying()
                print("started the audio loops")

                while let pcm = microphoneCapture.receive() {
                    if shouldStop.load(ordering: .relaxed) {
                        break
                    }
                    let pcm = MLXArray(pcm)[.newAxis, .newAxis]
                    if !model.onMicrophonePcm(pcm, ap: ap, ev: self) {
                        break
                    }
                }
                print()
                microphoneCapture.stopCapturing()
            }
            self.modelInfo = "finished generating"
        } catch {
            self.modelInfo = "failed: \(error)"
        }
        running = false
    }

    func load(moshi: Bool) async throws -> ModelState {
        switch self.loadState {
        case .idle:
            let m: ModelState
            if moshi {
                let model = try await MoshiModel(self)
                m = ModelState(model)
            } else {
                let model = try await MimiModel(self)
                m = ModelState(model)
            }
            self.loadState = .loaded(m)
            return m
        case .loaded(let m):
            return m
        }
    }

    func setModelInfo(_ s: String) {
        modelInfo = s
    }
}

protocol Model {
    init(_ ev: Evaluator) async throws
    mutating func reset()
    // If onMicrophonePcm returns true continue, otherwise break.
    mutating func onMicrophonePcm(_ pcm: MLXArray, ap: AudioPlayer, ev: Evaluator) -> Bool
}

struct MimiModel: Model {
    let mimi: Mimi
    let codes: MLXArray
    var currentStep: Int
    let totalSteps: Int

    init(_ ev: Evaluator) async throws {
        await ev.setModelInfo("building model")
        self.mimi = try await ev.makeMimi()
        await ev.setModelInfo("model built")
        let codeURL = try await ev.downloadFromHub(
            id: "lmz/moshi-swift", filename: "bria-codes.safetensors")
        self.codes = try loadArrays(url: codeURL)["codes"]!
        self.currentStep = 0
        self.totalSteps = codes.dim(-1)
    }

    mutating func reset() {
        mimi.resetState()
    }

    mutating func onMicrophonePcm(_ pcm: MLXArray, ap: AudioPlayer, ev: Evaluator) -> Bool {
        let micCodes = mimi.encodeStep(StreamArray(pcm))
        if let micCodes = micCodes.asArray() {
            // As of 2024-12-04, there is a memory leak if this eval is removed, this is
            // triggered even without the decoding and audio playing, only the line
            // below is needed:
            // let audioTokens = codes[.ellipsis, currentStep...currentStep]
            eval(micCodes)
            let (_, _, steps) = micCodes.shape3
            for _ in 0..<steps {
                if currentStep >= totalSteps {
                    break
                }
                let audioTokens = codes[.ellipsis, currentStep...currentStep]
                let pcmOut = mimi.decodeStep(StreamArray(audioTokens))
                if let p = pcmOut.asArray() {
                    let _ = ap.send(p.asArray(Float.self))
                }
                currentStep += 1
            }
        }
        return currentStep < totalSteps
    }
}

struct AsrModel: Model {
    let moshi: LM
    let vocab: [Int: String]
    let mimi: Mimi
    var prevTextToken: Int = 0
    var cnt: Int = 0
    let sampler: Sampler = Sampler(temp: 0.0)
    let asrDelayInSteps: Int = 25

    init(_ ev: Evaluator) async throws {
        await ev.setModelInfo("building model")
        let url = Bundle.main.url(
            forResource: "asr-300m-f28fe6d5@100", withExtension: "safetensors")!
        let cfg = LmConfig.asr300m()
        self.moshi = try await ev.makeMoshi(url, cfg)
        self.mimi = try await ev.makeMimi()
        await ev.setModelInfo("model built")
        let maxSteps = cfg.transformer.maxSeqLen
        self.vocab = try await ev.loadVocab(cfg)
        await ev.setModelInfo("warming up mimi")
        self.mimi.warmup()
        await ev.setModelInfo("warming up moshi")
        self.moshi.warmup()
        await ev.setModelInfo("done warming up")
    }

    mutating func reset() {
        mimi.resetState()
        moshi.resetCache()
        prevTextToken = self.moshi.cfg.textInitToken()
        cnt = 0
        let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
        let audioIds = (0..<16).map { _ in MLXArray([moshi.cfg.audioPaddingToken()]) }
        let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
        let (textToken, _) = sampler(logits: textLogits)
        let textTokenI: Int = textToken[0].item()
        print("sampled first", textTokenI)
        prevTextToken = textTokenI
    }

    mutating func onMicrophonePcm(_ pcm: MLXArray, ap: AudioPlayer, ev: Evaluator) -> Bool {
        let codebooks = moshi.cfg.audioCodebooks
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
                        Task { @MainActor in
                            ev.output += v
                        }
                    }
                }
                prevTextToken = textTokenI
                cnt += 1
            }
        }
        return true
    }
}

struct MoshiModel: Model {
    let moshi: LM
    let vocab: [Int: String]
    let mimi: Mimi
    let gen: LMGen

    init(_ ev: Evaluator) async throws {
        await ev.setModelInfo("building model")
        let url = try await ev.downloadFromHub(
            id: "kyutai/moshiko-mlx-q8", filename: "model.q8.safetensors")
        let cfg = LmConfig.moshi_2024_07()
        self.moshi = try await ev.makeMoshi(url, cfg)
        self.mimi = try await ev.makeMimi()
        await ev.setModelInfo("model built")
        let maxSteps = cfg.transformer.maxSeqLen
        self.gen = LMGen(
            moshi, maxSteps: maxSteps, audioSampler: Sampler(), textSampler: Sampler())
        self.vocab = try await ev.loadVocab(cfg)
        await ev.setModelInfo("warming up mimi")
        self.mimi.warmup()
        await ev.setModelInfo("warming up moshi")
        self.moshi.warmup()
        await ev.setModelInfo("done warming up")
    }

    mutating func reset() {
        mimi.resetState()
        gen.reset()
    }

    mutating func onMicrophonePcm(_ pcm: MLXArray, ap: AudioPlayer, ev: Evaluator) -> Bool {
        let codes = mimi.encodeStep(StreamArray(pcm))
        if let codes = codes.asArray() {
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                if let textToken = gen.step(
                    otherAudioTokens: codes[0..., 0..<8, step])
                {
                    let textTokenI: Int = textToken[0].item()
                    if textTokenI != 0 && textTokenI != 3 {
                        if let v = vocab[textTokenI] {
                            let v = v.replacing("▁", with: " ")
                            print(v, terminator: "")
                            fflush(stdout)
                            Task { @MainActor in
                                ev.output += v
                            }
                        }
                    }
                }
                if let audioTokens = gen.lastAudioTokens() {
                    let pcmOut = mimi.decodeStep(StreamArray(audioTokens[0..., 0..., .newAxis]))
                    if let p = pcmOut.asArray() {
                        let _ = ap.send(p.asArray(Float.self))
                    }
                }
            }
        }
        return true
    }
}

actor ModelState {
    let model: Model

    init(_ model: Model) {
        self.model = model
    }

    public func perform<R>(_ action: @Sendable (inout Model) async throws -> R)
        async rethrows
        -> R
    {
        var model = model
        let result = try await action(&model)
        return result
    }
}
