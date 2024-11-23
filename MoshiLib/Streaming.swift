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

    public func len(dim: Int) -> Int {
        self.inner?.dim(dim) ?? 0
    }

    public func cat2(rhs: StreamArray, dim: Int) -> StreamArray {
        switch (self.inner, rhs.inner) {
        case (.none, .none): StreamArray()
        case (.some(let lhs), .none): StreamArray(lhs)
        case (.none, .some(let rhs)): StreamArray(rhs)
        case (.some(let lhs), .some(let rhs)): StreamArray(concatenated([lhs, rhs], axis: dim))
        }
    }

    public func split(lhsLen: Int, dim: Int) -> (StreamArray, StreamArray) {
        if let t = self.inner {
            let len = t.dim(dim)
            let lhsLen = min(len, lhsLen)
            if lhsLen == 0 {
                return (StreamArray(), StreamArray(t))
            } else if lhsLen == len {
                return (StreamArray(t), StreamArray())
            } else {
                let split = t.split(indices: [lhsLen], axis: dim)
                return (StreamArray(split[0]), StreamArray(split[1]))
            }
        } else {
            return (StreamArray(), StreamArray())
        }
    }
}

public protocol StreamingModel {
    func resetState()
    func step(x: StreamArray) -> StreamArray
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
    let dim: Int

    init(_ op: BinOp, dim: Int) {
        self.prevLHS = StreamArray()
        self.prevRHS = StreamArray()
        self.op = op
        self.dim = dim
    }

    public func resetState() {
        self.prevLHS = StreamArray()
        self.prevRHS = StreamArray()
    }

    public func step(lhs: StreamArray, rhs: StreamArray) -> StreamArray {
        let lhs = self.prevLHS.cat2(rhs: lhs, dim: self.dim)
        let rhs = self.prevRHS.cat2(rhs: rhs, dim: self.dim)
        let lhsLen = lhs.len(dim: self.dim)
        let rhsLen = rhs.len(dim: self.dim)
        let commonLen = min(lhsLen, rhsLen)
        let (lhs_, prevLHS) = lhs.split(lhsLen: commonLen, dim: self.dim)
        let (rhs_, prevRHS) = rhs.split(lhsLen: commonLen, dim: self.dim)
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
