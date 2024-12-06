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

public class PerfStats: Callbacks {
    private let log: OSLog
    private var events: [(CFAbsoluteTime, EventKind)] = []

    public init() {
        self.log = OSLog(subsystem: "org.kyutai.moshi", category: "Performance")
    }

    func append(_ kind: EventKind) {
        events.append((CFAbsoluteTimeGetCurrent(), kind))
    }

    public func onReset() {
    }

    public func onInputAudioTokens(_ codes: MLXArray) {
    }

    public func onOutputTextToken(_ token: Int) {
    }

    public func onOutputAudioTokens(_ codes: MLXArray) {
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
}
