// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import MLX
import MLXNN
import MoshiLib

func runAsr(dir: String) throws {
    let mimi = try makeMimi(dir: dir)
    print("using device \(Device.defaultDevice().description)")

    let pcm = readAudioToPCMArray(
        fileURL: URL(fileURLWithPath: dir + "/bria-24khz.mp3"))!
    let chunkSize = 1920
    for start in stride(from: 0, to: pcm.count, by: chunkSize) {
        let end = min(start + chunkSize, pcm.count)
        let pcmA = MLXArray(pcm[start..<end])[.newAxis, .newAxis]
        let codes = mimi.encodeStep(StreamArray(pcmA))
    }
}
