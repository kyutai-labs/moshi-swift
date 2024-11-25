import AVFoundation
import Foundation

class ThreadSafeChannel<T> {
    private var buffer: [T] = []
    private let queue = DispatchQueue(label: "tschannel", attributes: .concurrent)
    private let semaphore = DispatchSemaphore(value: 0)

    func send(_ value: T) {
        queue.async(flags: .barrier) {
            self.buffer.append(value)
            self.semaphore.signal()
        }
    }

    func receive() -> T? {
        semaphore.wait()
        return queue.sync {
            guard !buffer.isEmpty else { return nil }
            return buffer.removeFirst()
        }
    }
}

// The code below is probably macos specific and unlikely to work on ios.
class MicrophoneCapture {
    private let audioEngine: AVAudioEngine!
    private let channel: ThreadSafeChannel<[Float]>

    init() {
        audioEngine = AVAudioEngine()
        channel = ThreadSafeChannel()
    }

    func startCapturing() {
        let inputNode = audioEngine.inputNode

        // Desired format: 1 channel (mono), 24kHz, Float32
        let desiredSampleRate: Double = 24000.0
        let desiredChannelCount: AVAudioChannelCount = 1

        let inputFormat = inputNode.inputFormat(forBus: 0)

        // Create a custom audio format with the desired settings
        guard
            let mono24kHzFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: desiredSampleRate,
                channels: desiredChannelCount,
                interleaved: false)
        else {
            print("Could not create target format")
            return
        }

        // Install a tap to capture audio and resample to the target format
        inputNode.installTap(onBus: 0, bufferSize: 1920, format: inputFormat) { buffer, _ in
            // Resample the buffer to match the desired format
            let converter = AVAudioConverter(from: inputFormat, to: mono24kHzFormat)
            let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: mono24kHzFormat, frameCapacity: AVAudioFrameCount(buffer.frameCapacity))!

            var error: NSError? = nil
            let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }

            converter?.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)

            if let error = error {
                print("Conversion error: \(error)")
                return
            }

            self.processAudioBuffer(buffer: convertedBuffer)
        }

        // Start the audio engine
        do {
            audioEngine.prepare()
            try audioEngine.start()
            print("Microphone capturing started at 24kHz, mono")
        } catch {
            print("Error starting audio engine: \(error)")
        }
    }

    private func processAudioBuffer(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)

        let pcmData = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount)).map {
            $0
        }
        channel.send(pcmData)
    }

    func stopCapturing() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        print("Microphone capturing stopped")
    }

    func receive() -> [Float]? {
        channel.receive()
    }
}
