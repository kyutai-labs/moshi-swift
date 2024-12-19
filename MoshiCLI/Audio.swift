// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation

func readAudioToPCMArray(fileURL: URL, channel: Int = 0) -> [Float]? {
    do {
        let audioFile = try AVAudioFile(forReading: fileURL)
        let format = audioFile.processingFormat
        let frameCount = UInt32(audioFile.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            print("failed to create buffer")
            return nil
        }

        try audioFile.read(into: buffer)
        guard let channelData = buffer.floatChannelData else {
            print("failed to get channel data")
            return nil
        }
        let _ = Int(format.channelCount)
        let frameLength = Int(buffer.frameLength)
        var pcmData: [Float] = []
        let samples = channelData[channel]
        pcmData.append(contentsOf: UnsafeBufferPointer(start: samples, count: frameLength))
        return pcmData
    } catch {
        print("error reading audio file: \(error)")
        return nil
    }
}

func writeWAVFile(_ pcmData: [Float], sampleRate: Double, outputURL: URL) throws {
    let format = AVAudioFormat(
        standardFormatWithSampleRate: sampleRate, channels: AVAudioChannelCount(1))
    guard let format = format else {
        throw NSError(
            domain: "AudioFormat", code: -1,
            userInfo: [NSLocalizedDescriptionKey: "Failed to create audio format"])
    }
    let audioFile = try AVAudioFile(forWriting: outputURL, settings: format.settings)
    guard
        let buffer = AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(pcmData.count))
    else {
        throw NSError(
            domain: "PCMBuffer", code: -1,
            userInfo: [NSLocalizedDescriptionKey: "Failed to create PCM buffer"])
    }
    buffer.frameLength = AVAudioFrameCount(pcmData.count)
    let channelData = buffer.floatChannelData!
    channelData[0].update(from: Array(pcmData), count: pcmData.count)
    try audioFile.write(from: buffer)
}
