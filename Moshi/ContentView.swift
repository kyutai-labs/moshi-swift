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
            await model.generate()
        }
    }

    private func generateMimi() {
        Task {
            await model.generateMimi()
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
        case loaded(Model)
    }

    enum LoadStateMimi {
        case idle
        case loaded(MimiModel)
    }

    var loadState = LoadState.idle
    var loadStateMimi = LoadStateMimi.idle

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

    func generate() async {
        guard !running else { return }

        self.shouldStop.store(false, ordering: .relaxed)
        self.modelInfo = "starting"
        self.output = ""
        running = true
        do {
            let model = try await load()
            try await model.perform { vocab, mimi, gen in
                mimi.resetState()
                gen.reset()
                // TODO: Do not create a fresh audio input/output on each session.
                let microphoneCapture = MicrophoneCapture()
                microphoneCapture.startCapturing()
                let player = AudioPlayer(sampleRate: 24000)
                try player.startPlaying()
                print("started the audio loops")
                let mimi = await model.mimi
                let gen = await model.gen

                while let pcm = microphoneCapture.receive() {
                    if shouldStop.load(ordering: .relaxed) {
                        break
                    }
                    let pcm = MLXArray(pcm)[.newAxis, .newAxis]
                    let codes = mimi.encodeStep(StreamArray(pcm))
                    if let codes = codes.asArray() {
                        let (_, _, steps) = codes.shape3
                        for step in 0..<steps {
                            if let textToken = gen.step(
                                otherAudioTokens: codes[0..., 0..<8, step])
                            {
                                let textTokenI: Int = textToken[0].item()
                                if textTokenI != 0 && textTokenI != 3 {
                                    if let v = model.vocab[textTokenI] {
                                        let v = v.replacing("▁", with: " ")
                                        print(v, terminator: "")
                                        fflush(stdout)
                                        Task { @MainActor in
                                            self.output += v
                                        }
                                    }
                                }
                            }
                            if let audioTokens = gen.lastAudioTokens() {
                                let pcmOut = mimi.decodeStep(
                                    StreamArray(audioTokens[0..., 0..., .newAxis]))
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
            self.modelInfo = "finished generating"
        } catch {
            self.modelInfo = "failed: \(error)"
        }
        running = false
    }

    func generateMimi() async {
        guard !running else { return }

        self.shouldStop.store(false, ordering: .relaxed)
        self.modelInfo = "starting"
        self.output = ""
        running = true
        do {
            let model = try await loadMimi()
            try await model.perform { codes, mimi in
                mimi.resetState()
                // TODO: Do not create a fresh audio input/output on each session.
                let microphoneCapture = MicrophoneCapture()
                var currentStep = 0
                let totalSteps = codes.dim(-1)
                microphoneCapture.startCapturing()
                let player = AudioPlayer(sampleRate: 24000)
                try player.startPlaying()

                Task { @MainActor in
                    self.modelInfo = "started the audio loops"
                }
                while let pcm = microphoneCapture.receive() {
                    if shouldStop.load(ordering: .relaxed) {
                        break
                    }
                    let pcm = MLXArray(pcm)[.newAxis, .newAxis]
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
                                let _ = player.send(p.asArray(Float.self))
                            }
                            currentStep += 1
                        }
                    }
                    if currentStep >= totalSteps {
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

    func load() async throws -> Model {
        switch self.loadState {
        case .idle:
            self.modelInfo = "downloading model"
            let url = try await downloadFromHub(
                id: "kyutai/moshiko-mlx-q8", filename: "model.q8.safetensors")
            let cfg = LmConfig.moshi_2024_07()
            let moshi = try makeMoshi(url, cfg)
            let mimi = try await makeMimi()
            self.modelInfo = "downloaded model"
            let maxSteps = cfg.transformer.maxSeqLen
            let gen = LMGen(
                moshi, maxSteps: maxSteps, audioSampler: Sampler(), textSampler: Sampler())
            let vocab = try await loadVocab(cfg)
            self.modelInfo = "warming up mimi"
            mimi.warmup()
            self.modelInfo = "warming up moshi"
            moshi.warmup()
            self.modelInfo = "done warming up"
            let m = Model(moshi: moshi, vocab: vocab, mimi: mimi, gen: gen)
            self.loadState = .loaded(m)
            return m
        case .loaded(let m):
            return m
        }
    }

    func loadMimi() async throws -> MimiModel {
        switch self.loadStateMimi {
        case .idle:
            self.modelInfo = "downloading model"
            let mimi = try await makeMimi()
            self.modelInfo = "warming up mimi"
            mimi.warmup()
            self.modelInfo = "done warming up"
            let codeURL = try await downloadFromHub(
                id: "lmz/moshi-swift", filename: "bria-codes.safetensors")
            let codes = try loadArrays(url: codeURL)["codes"]!
            let m = MimiModel(mimi: mimi, codes: codes)
            self.loadStateMimi = .loaded(m)
            return m
        case .loaded(let m):
            return m
        }
    }
}

actor Model {
    let moshi: LM
    let vocab: [Int: String]
    let mimi: Mimi
    let gen: LMGen

    init(moshi: LM, vocab: [Int: String], mimi: Mimi, gen: LMGen) {
        self.moshi = moshi
        self.vocab = vocab
        self.mimi = mimi
        self.gen = gen
    }

    public func perform<R>(_ action: @Sendable ([Int: String], Mimi, LMGen) async throws -> R) async rethrows
        -> R
    {
        try await action(vocab, mimi, gen)
    }
}

actor MimiModel {
    let mimi: Mimi
    let codes: MLXArray

    init(mimi: Mimi, codes: MLXArray) {
        self.mimi = mimi
        self.codes = codes
    }

    public func perform<R>(_ action: @Sendable (MLXArray, Mimi) async throws -> R) async rethrows
        -> R
    {
        try await action(codes, mimi)
    }
}
