// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import MoshiLib

public struct ASR {
    let moshi: LM
    let vocab: [Int: String]
    let mimi: Mimi
    var prevTextToken: Int = 0
    var cnt: Int = 0
    let sampler: Sampler = Sampler(temp: 0.0)
    let asrDelayInSteps: Int

    public init(_ moshi: LM, _ mimi: Mimi, vocab: [Int: String], asrDelayInSteps: Int = 25) {
        self.moshi = moshi
        self.mimi = mimi
        self.vocab = vocab
        self.asrDelayInSteps = asrDelayInSteps
    }

    mutating public func reset() {
        mimi.resetState()
        moshi.resetCache()
        prevTextToken = self.moshi.cfg.textInitToken()
        cnt = 0
        let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
        let audioIds = (0..<16).map { _ in MLXArray([moshi.cfg.audioPaddingToken()]) }
        let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
        let (textToken, _) = sampler(logits: textLogits)
        let textTokenI: Int = textToken[0].item()
        prevTextToken = textTokenI
    }

    mutating public func onPcmInput(_ pcm: MLXArray) -> [String] {
        var tokens: [String] = []
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
                        tokens.append(v)
                    }
                }
                prevTextToken = textTokenI
                cnt += 1
            }
        }
        return tokens
    }
}
