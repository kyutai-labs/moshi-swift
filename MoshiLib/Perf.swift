// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
import MLX
import os.signpost

public enum EventKind {
    case beginStep
    case endStep
    case beginDepformer
    case endDepformer
    case beginDecode
    case endDecode
    case beginEncode
    case endEncode
}

public protocol Callbacks {
    func onReset()
    func onEvent(_ eventKind: EventKind)
    func onInputAudioTokens(_ codes: MLXArray)
    func onOutputTextToken(_ token: Int)
    func onOutputAudioTokens(_ codes: MLXArray)
}

public class EmptyCallbacks: Callbacks {
    public init() {}
    public func onReset() {}
    public func onEvent(_ eventKind: EventKind) {}
    public func onInputAudioTokens(_ codes: MLXArray) {}
    public func onOutputTextToken(_ token: Int) {}
    public func onOutputAudioTokens(_ codes: MLXArray) {}
}

public struct ChromeTraceEvent: Codable {
    let name: String
    let cat: String
    let ph: String
    let ts: Int
    let pid: Int
    let tid: Int
}

public struct StatsSummary {
    public struct Stats: Identifiable {
        public var min: Float = Float.infinity
        public var max: Float = 0.0
        public var sum: Float = 0.0
        public var cnt: Int = 0
        public let id = UUID()

        mutating func addValue(_ b: CFAbsoluteTime, _ e: CFAbsoluteTime) {
            let v = Float(e - b)
            self.min = Float.minimum(self.min, v)
            self.max = Float.maximum(self.max, v)
            self.sum += v
            self.cnt += 1
        }
    }
    public var encode: Stats = Stats()
    public var decode: Stats = Stats()
    public var step: Stats = Stats()
    public var depformer: Stats = Stats()

    public init() {
    }
}

public class PerfStats: Callbacks {
    private let log: OSLog
    private var events: [(CFAbsoluteTime, EventKind)] = []
    private var inputAudioTokens: [MLXArray] = []
    private var outputAudioTokens: [MLXArray] = []
    private var textTokens: [Int] = []

    public init() {
        self.log = OSLog(subsystem: "org.kyutai.moshi", category: "Performance")
    }

    func append(_ kind: EventKind) {
        events.append((CFAbsoluteTimeGetCurrent(), kind))
    }

    public func onReset() {
        events.removeAll()
        inputAudioTokens.removeAll()
        outputAudioTokens.removeAll()
        textTokens.removeAll()
    }

    public func onInputAudioTokens(_ codes: MLXArray) {
        codes.eval()
        inputAudioTokens.append(codes)
    }

    public func onOutputTextToken(_ token: Int) {
        textTokens.append(token)
    }

    public func onOutputAudioTokens(_ codes: MLXArray) {
        codes.eval()
        outputAudioTokens.append(codes)
    }

    public func getSummary(maxEvents: Int) -> StatsSummary {
        let startIdx = max(0, self.events.count - maxEvents)
        var summary = StatsSummary()
        var lastBeginEncode: CFAbsoluteTime? = nil
        var lastBeginDecode: CFAbsoluteTime? = nil
        var lastBeginStep: CFAbsoluteTime? = nil
        var lastBeginDepformer: CFAbsoluteTime? = nil
        for event in self.events[startIdx...] {
            let time = event.0
            switch event.1 {
            case .beginStep:
                lastBeginStep = time
            case .beginDecode:
                lastBeginDecode = time
            case .beginEncode:
                lastBeginEncode = time
            case .beginDepformer:
                lastBeginDepformer = time
            case .endStep:
                if let b = lastBeginStep {
                    lastBeginStep = nil
                    summary.step.addValue(b, time)
                }
            case .endDecode:
                if let b = lastBeginDecode {
                    lastBeginDecode = nil
                    summary.decode.addValue(b, time)
                }
            case .endEncode:
                if let b = lastBeginEncode {
                    lastBeginEncode = nil
                    summary.encode.addValue(b, time)
                }
            case .endDepformer:
                if let b = lastBeginDepformer {
                    lastBeginDepformer = nil
                    summary.depformer.addValue(b, time)
                }
            }
        }
        return summary
    }

    public func onEvent(_ kind: EventKind) {
        switch kind {
        case .beginStep:
            os_signpost(.begin, log: log, name: "step")
        case .endStep:
            os_signpost(.end, log: log, name: "step")
        case .beginDepformer:
            os_signpost(.begin, log: log, name: "depformer")
        case .endDepformer:
            os_signpost(.end, log: log, name: "depformer")
        case .beginEncode:
            os_signpost(.begin, log: log, name: "encode")
        case .endEncode:
            os_signpost(.end, log: log, name: "encode")
        case .beginDecode:
            os_signpost(.begin, log: log, name: "decode")
        case .endDecode:
            os_signpost(.end, log: log, name: "decode")
        }
        append(kind)
    }

    public func writeJSONTrace(url: URL) throws {
        let encoder = JSONEncoder()
        var traceEvents: [ChromeTraceEvent] = []
        for (time, kind) in events {
            let ts = Int((time - events[0].0) * 1e6)
            let (name, ph) =
                switch kind {
                case .beginStep: ("step", "B")
                case .endStep: ("step", "E")
                case .beginEncode: ("encode", "B")
                case .endEncode: ("encode", "E")
                case .beginDepformer: ("depformer", "B")
                case .endDepformer: ("depformer", "E")
                case .beginDecode: ("decode", "B")
                case .endDecode: ("decode", "E")
                }
            traceEvents.append(
                ChromeTraceEvent(name: name, cat: "", ph: ph, ts: ts, pid: 42, tid: 1))
        }
        let jsonData = try encoder.encode(traceEvents)
        try jsonData.write(to: url)
    }

    public func allInputAudioTokens() -> MLXArray? {
        inputAudioTokens.isEmpty ? nil : concatenated(inputAudioTokens, axis: 2)
    }

    public func allOutputAudioTokens() -> MLXArray? {
        outputAudioTokens.isEmpty ? nil : concatenated(outputAudioTokens, axis: 2)
    }

    public func allTextTokens() -> MLXArray? {
        textTokens.isEmpty ? nil : MLXArray(textTokens)
    }

    public func writeCodes(url: URL) throws {
        var arrays: [String: MLXArray] = [:]
        if let a = allInputAudioTokens() {
            arrays["input_audio_tokens"] = a
        }
        if let a = allOutputAudioTokens() {
            arrays["output_audio_tokens"] = a
        }
        if let a = allTextTokens() {
            arrays["text_tokens"] = a
        }
        try save(arrays: arrays, url: url)
    }
}
