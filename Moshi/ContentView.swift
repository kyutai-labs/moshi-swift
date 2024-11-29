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

func downloadFromHub(id: String, filename: String) throws -> URL {
    var url: URL? = nil
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do {
            let downloadDir = FileManager.default.urls(
                for: .downloadsDirectory, in: .userDomainMask
            ).first!
            print(downloadDir)
            let api = HubApi(downloadBase: downloadDir)
            let repo = Hub.Repo(id: id)
            url = try await api.snapshot(from: repo, matching: filename)
        } catch {
            print("downloadFromHub", error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    return url!.appending(path: filename)
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

func makeMimi() throws -> Mimi {
    let cfg = MimiConfig.mimi_2024_07()
    let model = Mimi(cfg)

    let url = try downloadFromHub(
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

class Model {
    let moshi: LM
    let vocab: [Int: String]
    let mimi: Mimi
    let gen: LMGen

    init() throws {
        print("downloading model")
        let url = try downloadFromHub(
            id: "kyutai/moshiko-mlx-q4", filename: "model.q4.safetensors")
        let cfg = LmConfig.moshi_2024_07()
        self.moshi = try makeMoshi(url, cfg)
        self.mimi = try makeMimi()
        print("downloaded model")
        let maxSteps = cfg.transformer.maxSeqLen
        self.gen = LMGen(moshi, maxSteps: maxSteps, audioSampler: Sampler(), textSampler: Sampler())
        self.vocab = try loadVocab(cfg)
        print("warming up mimi")
        self.mimi.warmup()
        print("warming up moshi")
        self.moshi.warmup()
        print("done warming up")
    }
}

struct ContentView: View {
    @State private var model: Model? = nil
    @Environment(DeviceStat.self) private var deviceStat

    var body: some View {
        let buttonText = model == nil ? "Load Weights" : "Launch Model"
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("...")
            Button(buttonText) {
                do {
                    if self.model == nil {
                        // TODO: Async download with progress bar.
                        let m = try Model()
                        self.model = m
                    } else {
                        let model = self.model!
                        // TODO: Do not create a fresh audio input/output on each session.
                        let microphoneCapture = MicrophoneCapture()
                        microphoneCapture.startCapturing()
                        let player = AudioPlayer(sampleRate: 24000)
                        try player.startPlaying()
                        print("started the audio loops")
                        
                        // TODO: Async inference loop
                        while let pcm = microphoneCapture.receive() {
                            let pcm = MLXArray(pcm)[.newAxis, .newAxis]
                            let codes = model.mimi.encodeStep(StreamArray(pcm))
                            if let codes = codes.asArray() {
                                let (_, _, steps) = codes.shape3
                                for step in 0..<steps {
                                    if let textToken = model.gen.step(
                                        otherAudioTokens: codes[0..., 0..<8, step])
                                    {
                                        let textTokenI: Int = textToken[0].item()
                                        if textTokenI != 0 && textTokenI != 3 {
                                            if var v = model.vocab[textTokenI] {
                                                v.replace("▁", with: " ")
                                                print(v, terminator: "")
                                                fflush(stdout)
                                            }
                                        }
                                    }
                                    if let audioTokens = model.gen.lastAudioTokens() {
                                        let pcmOut = model.mimi.decodeStep(
                                            StreamArray(audioTokens[0..., 0..., .newAxis]))
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
                } catch {
                    print("error in callback", error)
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
}

#Preview {
    ContentView()
}
