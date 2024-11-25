// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import MLX
import MLXNN
import MoshiLib

func makeMoshi(_ filename: String) throws -> LM {
    let weights = try loadArrays(url: URL(fileURLWithPath: filename))
    let parameters = ModuleParameters.unflattened(weights)
    let cfg = LmConfig.asr1b()
    let model = LM(cfg)
    try model.update(parameters: parameters, verify: [.all])
    eval(model)
    return model
}

func runAsr(dir: String) throws {
    let mimi = try makeMimi(dir: dir)
    let moshi = try makeMoshi(dir + "/asr-1b-8d2516b9@150.safetensors")
    print("using device \(Device.defaultDevice().description)")

    let pcm = readAudioToPCMArray(
        fileURL: URL(fileURLWithPath: dir + "/bria-24khz.mp3"))!
    let chunkSize = 1920
    let sampler = Sampler()
    let codebooks = moshi.cfg.audioCodebooks
    var prevTextToken = moshi.cfg.textInVocabSize - 1
    do {
        let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
        let audioIds = (0..<16).map { _ in MLXArray([moshi.cfg.audioPaddingToken()]) }
        let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
        let (textToken, _) = sampler(logits: textLogits)
        let textTokenI: Int = textToken[0].item()
        print("sampled first", textTokenI)
        prevTextToken = textTokenI
    }
    for start in stride(from: 0, to: pcm.count, by: chunkSize) {
        let end = min(start + chunkSize, pcm.count)
        let pcmA = MLXArray(pcm[start..<end])[.newAxis, .newAxis]
        let codes = mimi.encodeStep(StreamArray(pcmA))
        if let codes = codes.asArray() {
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
                let audioIds = (0..<codebooks).map { codes[0..., $0, step] }
                let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
                let (textToken, _) = sampler(logits: textLogits)
                let textTokenI: Int = textToken[0].item()
                print("sampled", textTokenI)
                prevTextToken = textTokenI
            }
        }
    }
}
