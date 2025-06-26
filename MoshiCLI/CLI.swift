// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import ArgumentParser
import Foundation
import Hub
import MLX
import MLXNN
import MoshiLib
import Tokenizers

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
        do {
            url = try await Hub.snapshot(from: repo, matching: filename) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                print("\rretrieving \(filename): \(pct)%", terminator: "")
            }
        } catch {
            fatalError("cannot fetch \(id) \(filename): \(error)")
        }
        semaphore.signal()
    }
    semaphore.wait()
    print("\rretrieved \(filename)")
    return url!.appending(path: filename)
}

func maybeDownloadFromHub(filename: String) throws -> URL {
    // dowloadFromHub(id: id, filename: filename)
    let prefix = "hf://"
    if filename.hasPrefix(prefix) {
        let rest = filename.dropFirst(prefix.count)
        let components = rest.split(separator: "/", omittingEmptySubsequences: false)
        let id = components[0..<components.count - 1].joined(separator: "/")
        let filename = String(components.last!)
        return try downloadFromHub(id: id, filename: filename)
    } else {
        return URL(fileURLWithPath: filename)
    }
}

func makeTokenizer(hfRepo: String) throws -> any Tokenizer {
    var tokenizer: (any Tokenizer)? = nil
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do {
            tokenizer = try await AutoTokenizer.from(pretrained: hfRepo)
        } catch {
            fatalError("cannot build tokenizer \(error)")
        }
        semaphore.signal()
    }
    semaphore.wait()
    return tokenizer!
}

@main
struct Moshi: ParsableCommand {
    static let configuration = CommandConfiguration(
        subcommands: [
            Run.self, RunHelium.self, RunMimi.self, AudioToCodes.self, CodesToAudio.self,
            RunAsr.self, RunQwen.self,
        ]
    )
}

public enum Config: String, CaseIterable, ExpressibleByArgument {
    case moshi1b
    case moshi7b
}

struct Run: ParsableCommand {
    @Argument(help: "the model to run")
    var model: String

    @Option(help: "the file to process, use 'mic' for the microphone")
    var input: String?

    @Option(help: "the audio delay to apply")
    var audioDelay: Int = 2

    @Option(help: "the audio channel from the input file to be used")
    var channel: Int = 0

    @Option(help: "the config size")
    var config: Config = .moshi1b

    mutating func run() throws {
        let model = URL(fileURLWithPath: model)
        let cfg =
            switch config {
            case .moshi1b: LmConfig.moshi1b(audioDelay: audioDelay)
            case .moshi7b: LmConfig.moshi_2024_07()
            }

        switch input {
        case .none:
            try runMoshi(model, cfg: cfg, audioFile: nil)
        case .some("mic"): try runMoshiMic(model, cfg: cfg)
        case .some(let input):
            let audioFile = URL(fileURLWithPath: input)
            try runMoshi(
                model, cfg: cfg, audioFile: audioFile, channel: channel)
        }
    }
}

struct RunMimi: ParsableCommand {
    @Option(help: "whether to use the streaming mode or not")
    var streaming: Bool = false

    @Option(help: "the file to process")
    var input: String?

    @Option(help: "the audio channel from the input file to be used")
    var channel: Int = 0

    mutating func run() throws {
        let audioFile = input.flatMap { URL(fileURLWithPath: $0) }
        try runMimi(streaming: streaming, audioFile: audioFile, channel: channel)
    }
}

public enum HeliumConfig: String, CaseIterable, ExpressibleByArgument {
    case q4
    case q6
    case q8
    case bf16
}

struct RunQwen: ParsableCommand {
    @Option(help: "the config")
    var hfRepo: String = "Qwen/Qwen2.5-0.5B-Instruct"

    @Option(help: "the prompt to be used")
    var prompt: String = "Describe the swift programming language."

    @Option(help: "the number of tokens to generate")
    var n: Int = 256

    mutating func run() throws {
        let tokenizer = try makeTokenizer(hfRepo: hfRepo)
        let messages = [["role": "user", "content": prompt]]
        let encodedPrompt = try tokenizer.applyChatTemplate(messages: messages)
        let configUrl = try downloadFromHub(id: hfRepo, filename: "config.json")
        let configData = try Data(contentsOf: configUrl)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(QwenConfig.self, from: configData)
        print("config \(config)")
        let modelUrl = try downloadFromHub(id: hfRepo, filename: "model.safetensors")
        print("model \(modelUrl)")
        let weights = try loadArrays(url: modelUrl)
        guard let modelItem = ModuleParameters.unflattened(weights)["model"] else {
            fatalError("no model key in {configUrl}")
        }
        let parameters =
            switch modelItem {
            case .dictionary(let d): NestedDictionary(values: d)
            default: fatalError("model key in {configUrl} is not a dict")
            }

        let model = QwenModel(config)
        if let q = config.quantization {
            quantize(model: model, groupSize: q.groupSize, bits: q.bits)
        }
        try model.update(parameters: parameters, verify: [.all])
        eval(model)
        let cache = model.makeCache(bSize: 1)
        let sampler = Sampler()
        var lastToken = config.bosTokenId
        let startTime = CFAbsoluteTimeGetCurrent()
        var nTokens = 0
        for index in 0...(n + prompt.count) {
            let logits = model(MLXArray([lastToken]).reshaped(1, 1), cache: cache)
            if index < encodedPrompt.count {
                lastToken = encodedPrompt[index]
            } else {
                let (tok, _) = sampler(logits: logits[0])
                lastToken = tok.item<Int>()
            }
            let s = tokenizer.decode(tokens: [lastToken])
            print("\(s)", terminator: "")
            fflush(stdout)
            nTokens += 1
        }
        print()
        let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime
        print("\(nTokens) tokens generated, \(Double(nTokens) / elapsedTime) tok/s")
    }
}

struct RunHelium: ParsableCommand {
    @Option(help: "the config")
    var config: HeliumConfig = .q4

    mutating func run() throws {
        let cfg = LmConfig.helium2b()
        let filename =
            switch config {
            case .q4: "helium-1-preview-2b-q4.safetensors"
            case .q6: "helium-1-preview-2b-q6.safetensors"
            case .q8: "helium-1-preview-2b-q8.safetensors"
            case .bf16: "helium-1-preview-2b-bf16.safetensors"
            }
        let url = try downloadFromHub(id: "kyutai/helium-1-preview-2b-mlx", filename: filename)
        try runHelium(url, cfg: cfg)
    }
}

struct AudioToCodes: ParsableCommand {
    mutating func run() throws {
        try runAudioToCodes()
    }
}

struct CodesToAudio: ParsableCommand {
    @Option(help: "whether to write the output file or not")
    var writeFile: Bool = false

    mutating func run() throws {
        try runCodesToAudio(writeFile: writeFile)
    }
}

struct RunAsr: ParsableCommand {
    @Argument(help: "the model to run")
    var model: String

    @Option(help: "the file to process, use 'mic' for the microphone")
    var input: String?

    @Option(help: "the audio channel from the input file to be used")
    var channel: Int = 0

    mutating func run() throws {
        let model = try maybeDownloadFromHub(filename: model)
        let weights = try loadArrays(url: model)
        let cfg =
            switch weights["out_norm.weight"]?.shape {
            case .none: fatalError("no out_norm.weight tensor in \(model)")
            case .some([1024]): LmConfig.asr300m()
            case .some([2048]): LmConfig.asr1b()
            case .some([2560]): LmConfig.asr2b()
            case .some(let s): fatalError("unexpected shape for out_norm.weight \(s)")
            }
        print("here2")
        switch input {
        case .none:
            try runAsr(model, cfg, audioFile: nil, channel: channel)
        case .some("mic"): try runAsrMic(model, cfg)
        case .some(let input):
            let audioFile = URL(fileURLWithPath: input)
            try runAsr(model, cfg, audioFile: audioFile, channel: channel)
        }
    }
}
