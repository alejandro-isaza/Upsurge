// Copyright © 2015 Venture Media Labs.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import Foundation


public class MatrixSlice<T: Value>: MutableQuadraticType, CustomStringConvertible, Equatable {
    public typealias Index = (Int, Int)
    public typealias Slice = MatrixSlice<Element>
    public typealias Element = T

    public var rows: Int
    public var columns: Int
    
    public var base: Matrix<Element>
    public var span: Span

    public func withUnsafeBufferPointer<R>(@noescape body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R {
        let index = linearIndex(span.startIndex)
        return try base.withUnsafeBufferPointer { pointer in
            let start = pointer.baseAddress + index
            return try body(UnsafeBufferPointer(start: start, count: pointer.count - index))
        }
    }

    public func withUnsafePointer<R>(@noescape body: (UnsafePointer<Element>) throws -> R) rethrows -> R {
        let index = linearIndex(span.startIndex)
        return try base.withUnsafePointer { pointer in
            return try body(pointer + index)
        }
    }

    public func withUnsafeMutableBufferPointer<R>(@noescape body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R {
        let index = linearIndex(span.startIndex)
        return try base.withUnsafeMutableBufferPointer { pointer in
            let start = pointer.baseAddress + index
            return try body(UnsafeMutableBufferPointer(start: start, count: pointer.count - index))
        }
    }

    public func withUnsafeMutablePointer<R>(@noescape body: (UnsafeMutablePointer<Element>) throws -> R) rethrows -> R {
        let index = linearIndex(span.startIndex)
        return try base.withUnsafeMutablePointer { pointer in
            return try body(pointer + index)
        }
    }
    
    public var arrangement: QuadraticArrangement {
        return .RowMajor
    }
    
    public var stride: Int {
        return base.dimensions[1]
    }
    
    public var step: Int {
        return base.elements.step
    }

    init(base: Matrix<Element>, span: Span) {
        assert(Span(zeroTo: base.dimensions).contains(span))
        self.base = base
        self.span = span
        
        rows = span.dimensions[0]
        columns = span.dimensions[1]
    }
    
    public subscript(indices: Int...) -> Element {
        get {
            return self[indices]
        }
        set {
            self[indices] = newValue
        }
    }
    
    public subscript(indices: [Int]) -> Element {
        get {
            assert(indices.count == 2)
            return base[indices]
        }
        set {
            assert(indices.count == 2)
            base[indices] = newValue
        }
    }
    
    private subscript(span: Span) -> Slice {
        get {
            assert(self.span.contains(span))
            return MatrixSlice(base: base, span: span)
        }
        set {
            assert(self.span.contains(span))
            assert(self.span ≅ newValue.span)
            for (lhsIndex, rhsIndex) in zip(span, newValue.span) {
                self[lhsIndex] = newValue[rhsIndex]
            }
        }
    }
    
    public subscript(intervals: IntervalType...) -> Slice {
        get {
            return self[intervals]
        }
        set {
            self[intervals] = newValue
        }
    }
    
    public subscript(intervals: [IntervalType]) -> Slice {
        get {
            let span = Span(base: self.span, intervals: intervals)
            return self[span]
        }
        set {
            let span = Span(base: self.span, intervals: intervals)
            self[span] = newValue
        }
    }
    
    public func indexIsValid(indices: [Int]) -> Bool {
        assert(indices.count == dimensions.count)
        for (i, index) in indices.enumerate() {
            if index < span[i].startIndex || span[i].endIndex <= index {
                return false
            }
        }
        return true
    }
    
    public var description: String {
        var description = ""
        
        for i in 0..<rows {
            let contents = (0..<columns).map{"\(self[Interval(integerLiteral: span.startIndex[0] + i), Interval(integerLiteral: span.startIndex[1] + $0)])"}.joinWithSeparator("\t")
            
            switch (i, rows) {
            case (0, 1):
                description += "(\t\(contents)\t)"
            case (0, _):
                description += "⎛\t\(contents)\t⎞"
            case (rows - 1, _):
                description += "⎝\t\(contents)\t⎠"
            default:
                description += "⎜\t\(contents)\t⎥"
            }
            
            description += "\n"
        }
        
        return description
    }
}

// MARK: - Equatable

public func ==<T>(lhs: MatrixSlice<T>, rhs: Matrix<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T>(lhs: MatrixSlice<T>, rhs: MatrixSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T>(lhs: MatrixSlice<T>, rhs: Tensor<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T>(lhs: MatrixSlice<T>, rhs: TensorSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T>(lhs: MatrixSlice<T>, rhs: TwoDimensionalTensorSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

