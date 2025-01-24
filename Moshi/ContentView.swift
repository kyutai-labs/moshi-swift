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

struct CustomError: Error {
    let message: String

    init(_ s: String) {
        message = s
    }
}

enum ModelSelect: String, CaseIterable, Identifiable {
    case mimi
    case asr
    case moshi
    case helium

    var id: Self { return self }
}

struct ContentView: View {
    @State var model = Evaluator()
    @State var selectedModel: ModelSelect = .mimi
    @Environment(DeviceStat.self) private var deviceStat
    @State private var displayStats = false

    var body: some View {
        NavigationSplitView(
            sidebar: {
                List {
                    ForEach(ModelSelect.allCases) { modelType in
                        NavigationLink(
                            modelType.rawValue,
                            destination: {
                                ModelView(
                                    model: $model, modelType: modelType, displayStats: $displayStats
                                )
                            })
                    }
                }
                .navigationTitle("Available Models")
                #if os(iOS)
                    .navigationBarTitleDisplayMode(.inline)
                #endif

            },
            detail: {
                Text("Please choose a model type")
            })
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
    var urls: (URL, URL)? = nil
    var stat = ""
    var output = ""
    var progress: Progress? = nil
    var statsSummary: StatsSummary = StatsSummary()
    var bufferedDuration: Double = 0.0
    var totalDuration: Double = 0.0
    let shouldStop: Atomic<Bool> = .init(false)
    let cb: PerfStats = PerfStats()

    enum LoadState {
        case idle
        case loaded(ModelState, ModelSelect)
    }

    var loadState = LoadState.idle

    func downloadFromHub(id: String, filename: String) async throws -> URL {
        guard
            let downloadDir = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else {
            throw CustomError("cannot find the app support directory")
        }
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
            case 4000: "test_en_audio_4000.json"
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
        } else if url.lastPathComponent.hasSuffix(".q6.safetensors") {
            quantize(model: model, groupSize: 64, bits: 6)
        } else if url.lastPathComponent.hasSuffix(".q8.safetensors") {
            quantize(model: model, groupSize: 64, bits: 8)
        }
        try model.update(parameters: parameters, verify: [.all])
        eval(model)
        return model
    }

    func makeHelium(_ url: URL, _ cfg: LmConfig) throws -> LM {
        let weights = try loadArrays(url: url)
        let parameters = ModuleParameters.unflattened(weights)
        let model = LM(cfg, bSize: 1)
        if url.lastPathComponent.hasSuffix("q4.safetensors") {
            quantize(model: model, groupSize: 64, bits: 4)
        } else if url.lastPathComponent.hasSuffix("q8.safetensors") {
            quantize(model: model, groupSize: 64, bits: 8)
        }
        try model.update(parameters: parameters, verify: [.all])
        eval(model)
        return model
    }

    func makeMimi(numCodebooks: Int) async throws -> Mimi {
        let cfg = MimiConfig.mimi_2024_07(numCodebooks: numCodebooks)
        let model = Mimi(cfg, bSize: 1)

        let url = try await downloadFromHub(
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

    func generate(_ sm: ModelSelect) async {
        guard !running else { return }

        self.shouldStop.store(false, ordering: .relaxed)
        self.modelInfo = "starting"
        self.output = ""
        self.totalDuration = 0.0
        running = true
        do {
            let model = try await load(sm)
            let urls = try await model.perform { model in
                model.reset()
                await self.cb.onReset()
                // TODO: Do not create a fresh audio input/output on each session.
                let microphoneCapture = MicrophoneCapture()
                microphoneCapture.startCapturing()
                let ap = AudioPlayer(sampleRate: 24000)
                try ap.startPlaying()
                print("started the audio loops")

                var step = 0
                let startTime = CFAbsoluteTimeGetCurrent()
                while let pcm = microphoneCapture.receive() {
                    if shouldStop.load(ordering: .relaxed) {
                        break
                    }
                    let pcm = MLXArray(pcm)[.newAxis, .newAxis]
                    if !model.onMicrophonePcm(pcm, ap: ap, ev: self) {
                        break
                    }
                    step += 1
                    let currentStep = step
                    Task { @MainActor in
                        if currentStep % 5 == 0 {
                            self.bufferedDuration =
                                ap.bufferedDuration() + microphoneCapture.bufferedDuration()
                            self.totalDuration =  CFAbsoluteTimeGetCurrent() - startTime
                        }
                        if currentStep % 20 == 0 {
                            self.statsSummary = self.cb.getSummary(maxEvents: 100)
                        }
                    }
                }
                print()
                let traceURL = FileManager.default.temporaryDirectory.appendingPathComponent(
                    "moshi-trace.json")
                try await self.cb.writeJSONTrace(url: traceURL)
                let codesURL = FileManager.default.temporaryDirectory.appendingPathComponent(
                    "moshi-codes.safetensors")
                try await self.cb.writeCodes(url: codesURL)
                microphoneCapture.stopCapturing()
                return (traceURL, codesURL)
            }
            self.urls = urls
            self.modelInfo = "finished generating"
        } catch {
            self.modelInfo = "failed: \(error)"
        }
        self.bufferedDuration = 0.0
        self.statsSummary = StatsSummary()
        running = false
    }

    func load(_ sm: ModelSelect) async throws -> ModelState {
        if case .loaded(let m, sm) = self.loadState {
            return m
        }
        // Start by reseting loadState so as to release the memory used
        // by the currently loaded model if any.
        self.loadState = .idle
        let m: ModelState
        switch sm {
        case .moshi:
            let model = try await MoshiModel(self, self.cb)
            m = ModelState(model)
        case .mimi:
            let model = try await MimiModel(self, self.cb)
            m = ModelState(model)
        case .asr:
            let model = try await AsrModel(self, self.cb)
            m = ModelState(model)
        case .helium:
            let model = try await HeliumModel(self, self.cb)
            m = ModelState(model)
        }
        self.loadState = .loaded(m, sm)
        return m
    }

    func setModelInfo(_ s: String) {
        modelInfo = s
    }
}

protocol Model {
    init(_ ev: Evaluator, _ cb: Callbacks) async throws
    mutating func reset()
    // If onMicrophonePcm returns true continue, otherwise break.
    mutating func onMicrophonePcm(_ pcm: MLXArray, ap: AudioPlayer, ev: Evaluator) -> Bool
}

struct MimiModel: Model {
    let mimi: Mimi
    let codes: MLXArray
    var currentStep: Int
    let totalSteps: Int
    let cb: Callbacks

    init(_ ev: Evaluator, _ cb: Callbacks) async throws {
        await ev.setModelInfo("building model")
        self.mimi = try await ev.makeMimi(numCodebooks: 16)
        await ev.setModelInfo("model built")
        let codeURL = try await ev.downloadFromHub(
            id: "lmz/moshi-swift", filename: "bria-codes.safetensors")
        guard let codes = try loadArrays(url: codeURL)["codes"] else {
            throw CustomError("no codes in \(codeURL)")
        }
        self.codes = codes
        self.currentStep = 0
        self.totalSteps = codes.dim(-1)
        self.cb = cb
    }

    mutating func reset() {
        mimi.resetState()
    }

    mutating func onMicrophonePcm(_ pcm: MLXArray, ap: AudioPlayer, ev: Evaluator) -> Bool {
        let sampleText =
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        let sampleWords = sampleText.components(separatedBy: " ")
        self.cb.onEvent(.beginEncode)
        let micCodes = mimi.encodeStep(StreamArray(pcm))
        micCodes.eval()
        self.cb.onEvent(.endEncode)
        if let micCodes = micCodes.asArray() {
            // As of 2024-12-04, there is a memory leak if this eval is removed, this is
            // triggered even without the decoding and audio playing, only the line
            // below is needed:
            // let audioTokens = codes[.ellipsis, currentStep...currentStep]
            eval(micCodes)
            self.cb.onInputAudioTokens(micCodes)
            let (_, _, steps) = micCodes.shape3
            for _ in 0..<steps {
                if currentStep >= totalSteps {
                    break
                }
                let audioTokens = codes[.ellipsis, currentStep...currentStep]
                self.cb.onEvent(.beginDecode)
                self.cb.onOutputAudioTokens(audioTokens)
                let pcmOut = mimi.decodeStep(StreamArray(audioTokens))
                pcmOut.eval()
                self.cb.onEvent(.endDecode)
                if let p = pcmOut.asArray() {
                    let _ = ap.send(p.asArray(Float.self))
                }
                if currentStep % 4 == 0 {
                    let v = sampleWords[(currentStep / 4) % sampleWords.count] + " "
                    Task { @MainActor in
                        ev.output += v
                    }
                }
                currentStep += 1
            }
        }
        return currentStep < totalSteps
    }
}

struct HeliumModel: Model {
    let helium: LM
    let vocab: [Int: String]

    init(_ ev: Evaluator, _ cb: Callbacks) async throws {
        await ev.setModelInfo("building model")
        let cfg = LmConfig.helium2b()
        let url = try await ev.downloadFromHub(
            id: "kyutai/helium-1-preview-2b-mlx", filename: "helium-1-preview-2b-q4.safetensors")
        helium = try await ev.makeHelium(url, cfg)
        await ev.setModelInfo("model built")
        vocab = try await ev.loadVocab(cfg)
        await ev.setModelInfo("warming up helium")
        helium.warmup()
        await ev.setModelInfo("done warming up")
    }

    mutating func reset() {
    }

    mutating func onMicrophonePcm(_ pcm: MLXArray, ap: AudioPlayer, ev: Evaluator) -> Bool {
        let maxSteps = helium.cfg.transformer.maxSeqLen
        let sampler = Sampler()
        let cb = PerfStats()

        var lastToken = MLXArray([1])
        for stepIdx in 0...min(500, maxSteps) {
            let (textToken, _) = helium.sample(
                textIds: lastToken.reshaped([1, 1]), audioIds: [], stepIdx: stepIdx,
                textSampler: sampler,
                audioSampler: sampler, cb: cb)
            let textTokenI: Int = textToken[0].item()
            if var v = vocab[textTokenI] {
                if v == "<0x0A>" {
                    Task { @MainActor in
                        ev.output += "\n"
                    }
                } else {
                    v.replace("▁", with: " ")
                    Task { @MainActor in
                        ev.output += v
                    }
                }
            }
            lastToken = textToken
        }

        return false
    }
}

struct AsrModel: Model {
    var asr: ASR

    init(_ ev: Evaluator, _ cb: Callbacks) async throws {
        await ev.setModelInfo("building model")
        guard
            let url = Bundle.main.url(
                forResource: "asr-300m-ba46d99f@1000", withExtension: "safetensors")
        else {
            throw CustomError("cannot retrieve local model")
        }
        let cfg = LmConfig.asr300m()
        let moshi = try await ev.makeMoshi(url, cfg)
        let mimi = try await ev.makeMimi(numCodebooks: 32)
        await ev.setModelInfo("model built")
        let vocab = try await ev.loadVocab(cfg)
        await ev.setModelInfo("warming up mimi")
        mimi.warmup()
        await ev.setModelInfo("warming up moshi")
        moshi.warmup()
        await ev.setModelInfo("done warming up")
        self.asr = ASR(moshi, mimi, vocab: vocab, cb: cb)
    }

    mutating func reset() {
        asr.reset()
    }

    mutating func onMicrophonePcm(_ pcm: MLXArray, ap: AudioPlayer, ev: Evaluator) -> Bool {
        let tokens = asr.onPcmInput(pcm)
        for v in tokens {
            print(v, terminator: "")
            fflush(stdout)
            Task { @MainActor in
                ev.output += v
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
    let cb: Callbacks

    init(_ ev: Evaluator, _ cb: Callbacks) async throws {
        await ev.setModelInfo("building model")
        let url: URL
        let cfg: LmConfig
        let localURL = Bundle.main.url(
            forResource: "moshi-1b-9e90ea6c@6.q8", withExtension: "safetensors")
        switch localURL {
        case .none:
            url = try await ev.downloadFromHub(
                id: "kyutai/moshiko-mlx-q8", filename: "model.q8.safetensors")
            cfg = LmConfig.moshi_2024_07()
        case .some(let localURL):
            url = localURL
            cfg = LmConfig.moshi1b(audioDelay: 2)
        }
        self.moshi = try await ev.makeMoshi(url, cfg)
        self.mimi = try await ev.makeMimi(numCodebooks: 16)
        await ev.setModelInfo("model built")
        let maxSteps = cfg.transformer.maxSeqLen
        self.cb = cb
        self.gen = LMGen(
            moshi, maxSteps: maxSteps, audioSampler: Sampler(), textSampler: Sampler(), cb: self.cb)
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
        cb.onEvent(.beginEncode)
        let codes = mimi.encodeStep(StreamArray(pcm))
        codes.eval()
        cb.onEvent(.endEncode)
        if let codes = codes.asArray() {
            self.cb.onInputAudioTokens(codes)
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                if let textToken = gen.step(
                    otherAudioTokens: codes[0..., 0..<8, step])
                {
                    let textTokenI: Int = textToken[0].item()
                    self.cb.onOutputTextToken(textTokenI)
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
                    let audioTokens = audioTokens[0..., 0..., .newAxis]
                    self.cb.onOutputAudioTokens(audioTokens)
                    cb.onEvent(.beginDecode)
                    let pcmOut = mimi.decodeStep(StreamArray(audioTokens))
                    pcmOut.eval()
                    cb.onEvent(.endDecode)
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
