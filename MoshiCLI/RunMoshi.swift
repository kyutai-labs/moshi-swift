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
    var prevTextToken = moshi.cfg.textInVocabSize - 1
    for start in stride(from: 0, to: pcm.count, by: chunkSize) {
        let end = min(start + chunkSize, pcm.count)
        let pcmA = MLXArray(pcm[start..<end])[.newAxis, .newAxis]
        let codes = mimi.encodeStep(StreamArray(pcmA))
        print("codes", codes.shape)
        if let codes = codes.asArray() {
            let (_, codebooks, steps) = codes.shape3
            for step in 0..<steps {
                let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
                let audioIds = (0..<codebooks).map { codes[0..., $0, step] }
                let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
                let textToken = sampler(logits: textLogits)
                print(textToken)
                // prevTextToken = textToken
            }
        }
    }
}
