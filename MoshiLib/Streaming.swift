// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXFast
import MLXNN
import MLXRandom

public class StreamArray {
    let inner: MLXArray?

    public init(_ x: MLXArray? = nil) {
        self.inner = x
    }

    public func dim(_ dim: Int) -> Int {
        self.inner?.dim(dim) ?? 0
    }

    public func eval() {
        self.inner?.eval()
    }

    public func cat2(_ rhs: StreamArray, axis: Int) -> StreamArray {
        switch (self.inner, rhs.inner) {
        case (.none, .none): StreamArray()
        case (.some(let lhs), .none): StreamArray(lhs)
        case (.none, .some(let rhs)): StreamArray(rhs)
        case (.some(let lhs), .some(let rhs)): StreamArray(concatenated([lhs, rhs], axis: axis))
        }
    }

    public var shape: [Int]? {
        self.inner?.shape
    }

    public func asArray() -> MLXArray? {
        self.inner
    }

    public func narrow(_ offset: Int, _ len: Int, axis: Int) -> StreamArray {
        if let inner = self.inner {
            let totalLen = inner.dim(axis)
            if len <= 0 {
                return StreamArray()
            } else {
                // Not sure if there exists something closer to pytorch narrow
                let t = inner.split(indices: [offset, min(totalLen, offset + len)], axis: axis)
                return StreamArray(t[1])
            }
        } else {
            return StreamArray()
        }
    }

    public func split(lhsLen: Int, axis: Int) -> (StreamArray, StreamArray) {
        if let t = self.inner {
            let len = t.dim(axis)
            let lhsLen = min(len, lhsLen)
            if lhsLen == 0 {
                return (StreamArray(), StreamArray(t))
            } else if lhsLen == len {
                return (StreamArray(t), StreamArray())
            } else {
                let split = t.split(indices: [lhsLen], axis: axis)
                return (StreamArray(split[0]), StreamArray(split[1]))
            }
        } else {
            return (StreamArray(), StreamArray())
        }
    }

    public func elu() -> StreamArray {
        self.map { MLXNN.elu($0, alpha: 1.0) }
    }

    public func map(_ f: (MLXArray) -> MLXArray) -> StreamArray {
        switch self.inner {
        case .none: StreamArray()
        case .some(let x): StreamArray(f(x))
        }
    }
}

public protocol StreamingLayer {
    func resetState()
    func step(_ x: StreamArray) -> StreamArray
}

public class StreamingBinOp {
    enum BinOp {
        case add
        case mul
        case sub
        case div
    }

    var prevLHS: StreamArray
    var prevRHS: StreamArray
    let op: BinOp
    let axis: Int

    init(_ op: BinOp, axis: Int) {
        self.prevLHS = StreamArray()
        self.prevRHS = StreamArray()
        self.op = op
        self.axis = axis
    }

    public func resetState() {
        self.prevLHS = StreamArray()
        self.prevRHS = StreamArray()
    }

    public func step(_ lhs: StreamArray, _ rhs: StreamArray) -> StreamArray {
        let lhs = self.prevLHS.cat2(lhs, axis: self.axis)
        let rhs = self.prevRHS.cat2(rhs, axis: self.axis)
        let lhsLen = lhs.dim(self.axis)
        let rhsLen = rhs.dim(self.axis)
        let commonLen = min(lhsLen, rhsLen)
        let (lhs_, prevLHS) = lhs.split(lhsLen: commonLen, axis: self.axis)
        let (rhs_, prevRHS) = rhs.split(lhsLen: commonLen, axis: self.axis)
        self.prevLHS = prevLHS
        self.prevRHS = prevRHS
        switch (lhs_.inner, rhs_.inner) {
        case (.some(let l), .some(let r)):
            var res: MLXArray
            switch self.op {
            case .add: res = l + r
            case .sub: res = l - r
            case .mul: res = l * r
            case .div: res = l / r
            }
            return StreamArray(res)
        case (.none, .none): return StreamArray()
        case _: fatalError("internal error")
        }
    }
}
