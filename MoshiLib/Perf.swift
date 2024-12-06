// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

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

public struct ChromeTraceEvent: Codable {
    let name: String
    let cat: String
    let ph: String
    let ts: Int
    let pid: Int
    let tid: Int
}

public class PerfStats {
    private let log: OSLog
    private var events: [(CFAbsoluteTime, EventKind)] = []

    public init() {
        self.log = OSLog(subsystem: "org.kyutai.moshi", category: "Performance")
    }

    public func append(_ kind: EventKind) {
        events.append((CFAbsoluteTimeGetCurrent(), kind))
    }

    public func beginStep() {
        os_signpost(.begin, log: log, name: "step")
        append(.beginStep)
    }

    public func endStep() {
        os_signpost(.end, log: log, name: "step")
        append(.endStep)
    }

    public func beginDepformer() {
        os_signpost(.begin, log: log, name: "depformer")
        append(.beginDepformer)
    }

    public func endDepformer() {
        os_signpost(.end, log: log, name: "depformer")
        append(.endDepformer)
    }

    public func beginEncode() {
        os_signpost(.begin, log: log, name: "encode")
        append(.beginEncode)
    }

    public func endEncode() {
        os_signpost(.end, log: log, name: "encode")
        append(.endEncode)
    }

    public func beginDecode() {
        os_signpost(.begin, log: log, name: "decode")
        append(.beginDecode)
    }

    public func endDecode() {
        os_signpost(.end, log: log, name: "decode")
        append(.endDecode)
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
