import AVFoundation
import Foundation
import Synchronization

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
    private let audioEngine: AVAudioEngine
    private let channel: ThreadSafeChannel<[Float]>
    private let bufferedLen: Atomic<Int> = .init(0)

    init() {
        audioEngine = AVAudioEngine()
        channel = ThreadSafeChannel()
    }

    func startCapturing() {
        let inputNode = audioEngine.inputNode
        // Setting the voice mode on macos causes weird hangs of the microphone
        // so we discard it for now.
        #if os(iOS)
            do {
            //    try inputNode.setVoiceProcessingEnabled(true)
            } catch {
                print("could not set voice processing on the input node")
            }
        #endif

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

        // Resample the buffer to match the desired format
        let converter = AVAudioConverter(from: inputFormat, to: mono24kHzFormat)

        // Install a tap to capture audio and resample to the target format
        inputNode.installTap(onBus: 0, bufferSize: 1920, format: inputFormat) { buffer, _ in
            let targetLen = Int(buffer.frameLength) * 24000 / Int(inputFormat.sampleRate)
            let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: mono24kHzFormat, frameCapacity: AVAudioFrameCount(targetLen))!
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
        bufferedLen.add(pcmData.count, ordering: .sequentiallyConsistent)
        channel.send(pcmData)
    }

    func stopCapturing() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        print("Microphone capturing stopped")
    }

    func receive() -> [Float]? {
        let data = channel.receive()
        if let data = data {
            bufferedLen.subtract(data.count, ordering: .sequentiallyConsistent)
        }
        return data
    }

    func bufferedDuration() -> Double {
        return Double(bufferedLen.load(ordering: .sequentiallyConsistent)) / 24000.0
    }
}

class FloatRingBuffer {
    private var buffer: [Float]
    private let capacity: Int
    private var readIndex = 0
    private var writeIndex = 0
    private var count = 0

    private let lock = NSLock()

    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = [Float](repeating: 0, count: capacity)
    }

    func currentCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func write(_ values: [Float]) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        if values.count + count > capacity {
            return false
        }
        for value in values {
            buffer[writeIndex] = value
            writeIndex = (writeIndex + 1) % capacity
            count += 1
        }
        return true
    }

    func read(maxCount: Int) -> [Float] {
        lock.lock()
        defer { lock.unlock() }

        var values: [Float] = []
        for _ in 0..<maxCount {
            if count == 0 {
                break
            }
            let value = buffer[readIndex]
            values.append(value)
            readIndex = (readIndex + 1) % capacity
            count -= 1
        }
        return values
    }
}

class AudioPlayer {
    private let audioEngine: AVAudioEngine
    private let ringBuffer: FloatRingBuffer
    private let sampleRate: Double

    init(sampleRate: Double) {
        audioEngine = AVAudioEngine()
        ringBuffer = FloatRingBuffer(capacity: Int(sampleRate * 4))
        self.sampleRate = sampleRate
    }

    func bufferedDuration() -> Double {
        Double(ringBuffer.currentCount()) / sampleRate
    }

    func startPlaying() throws {
        let audioFormat = AVAudioFormat(standardFormatWithSampleRate: self.sampleRate, channels: 1)!
        let sourceNode = AVAudioSourceNode(format: audioFormat) {
            _, _, frameCount, audioBufferList -> OSStatus in
            let audioBuffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
            guard let channelData = audioBuffers[0].mData?.assumingMemoryBound(to: Float.self)
            else {
                // TODO: Get a proper error here that would work on ios.
                return 1
            }
            let data = self.ringBuffer.read(maxCount: Int(frameCount))
            for i in 0..<Int(frameCount) {
                channelData[i] = i < data.count ? data[i] : 0
            }
            return noErr
        }

        let af = sourceNode.inputFormat(forBus: 0)
        print("playing audio-format \(af)")
        audioEngine.attach(sourceNode)
        audioEngine.connect(sourceNode, to: audioEngine.mainMixerNode, format: audioFormat)
        try audioEngine.start()
    }

    func send(_ values: [Float]) -> Bool {
        ringBuffer.write(values)
    }
}

func setDefaultToStd() {
    #if os(iOS)
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playAndRecord, mode: .default, options: [.allowAirPlay, .allowBluetooth])
            try audioSession.setActive(true)
            try audioSession.overrideOutputAudioPort(.none)
        } catch {
            print("failed to configure audio session: \(error.localizedDescription)")
        }
    #endif
}

func setDefaultToSpeaker() {
    #if os(iOS)
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playAndRecord, mode: .default, options: .defaultToSpeaker)
            try audioSession.setActive(true)
            try audioSession.overrideOutputAudioPort(.speaker)
        } catch {
            print("failed to configure audio session: \(error.localizedDescription)")
        }
    #endif
}
