//
//  ContentView.swift
//  moshi
//
//  Created by Laurent on 20/11/2024.
//

import SwiftUI
import MLX
import MLXNN
import MLXRandom
import Hub
import Metal
import MoshiLib

func downloadFromHub(id: String, filename: String) throws -> URL {
    var url: URL? = nil
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do {
            let downloadDir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!
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

struct ContentView: View {
    @State private var model: LM? = nil

    var body: some View {
        let buttonText = model == nil ? "Load Weights" : "Launch Model"
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("...")
            Button(buttonText) {
                do {
                    print("downloading model")
                    let url = try downloadFromHub(id: "kyutai/moshiko-mlx-q4", filename: "model.q4.safetensors")
                    print("downloaded model")
                    let cfg = LmConfig.moshi_2024_07()
                    let lm = try makeMoshi(url, cfg)
                    self.model = lm
                } catch {
                    print("error in callback", error)
                }
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
