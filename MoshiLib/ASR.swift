// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import MoshiLib

public class ASR {
    let moshi: LM
    let vocab: [Int: String]
    let mimi: Mimi
    var prevTextToken: Int = 0
    let sampler: Sampler = Sampler(temp: 0.0)
    let cb: Callbacks

    public init(
        _ moshi: LM, _ mimi: Mimi, vocab: [Int: String], cb: Callbacks = EmptyCallbacks()
    ) {
        self.moshi = moshi
        self.mimi = mimi
        self.vocab = vocab
        self.cb = cb
    }

    public func reset() {
        mimi.resetState()
        moshi.resetCache()
        prevTextToken = self.moshi.cfg.textInitToken()
        let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
        let audioIds = (0..<self.moshi.cfg.audioCodebooks).map { _ in
            MLXArray([moshi.cfg.audioPaddingToken()])
        }
        let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
        let (textToken, _) = sampler(logits: textLogits)
        let textTokenI: Int = textToken[0].item()
        prevTextToken = textTokenI
        cb.onReset()
    }

    public func onPcmInput(_ pcm: MLXArray) -> [String] {
        var tokens: [String] = []
        let codebooks = moshi.cfg.audioCodebooks
        cb.onEvent(.beginEncode)
        let codes = mimi.encodeStep(StreamArray(pcm))
        codes.eval()
        cb.onEvent(.endEncode)
        if let codes = codes.asArray() {
            cb.onInputAudioTokens(codes)
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
                let audioIds = (0..<codebooks).map { codes[0..., $0, step].reshaped(1, 1) }
                cb.onEvent(.beginStep)
                let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
                eval(textLogits)
                cb.onEvent(.endStep)
                let (textToken, _) = sampler(logits: textLogits)
                let textTokenI: Int = textToken[0].item()
                cb.onOutputTextToken(textTokenI)
                if textTokenI != 0 && textTokenI != 3 {
                    if var v = vocab[textTokenI] {
                        v.replace("▁", with: " ")
                        tokens.append(v)
                    }
                }
                prevTextToken = textTokenI
            }
        }
        return tokens
    }
}
