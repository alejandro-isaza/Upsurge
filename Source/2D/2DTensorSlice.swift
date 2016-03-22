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


public class TwoDimensionalTensorSlice<T: Value>: MutableQuadraticType, Equatable {
    public typealias Index = [Int]
    public typealias Slice = TwoDimensionalTensorSlice<Element>
    public typealias Element = T
    
    public var arrangement: QuadraticArrangement {
        return .RowMajor
    }
    
    public let rows: Int
    public let columns: Int
    public var stride: Int
    
    var base: Tensor<Element>
    public var span: Span

    public func withUnsafeBufferPointer<R>(@noescape body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R {
        return try base.withUnsafeBufferPointer(body)
    }

    public func withUnsafePointer<R>(@noescape body: (UnsafePointer<Element>) throws -> R) rethrows -> R {
        return try base.withUnsafePointer(body)
    }

    public func withUnsafeMutableBufferPointer<R>(@noescape body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R {
        return try base.withUnsafeMutableBufferPointer(body)
    }

    public func withUnsafeMutablePointer<R>(@noescape body: (UnsafeMutablePointer<Element>) throws -> R) rethrows -> R {
        return try base.withUnsafeMutablePointer(body)
    }
    
    public var step: Int {
        return base.elements.step
    }
    
    init(base: Tensor<Element>, span: Span) {
        assert(span.dimensions.count == base.dimensions.count)
        self.base = base
        self.span = span
        
        assert(base.spanIsValid(span))
        assert(span.dimensions.reduce(0){ $0.1 > 1 ? $0.0 + 1 : $0.0 } <= 2)
        assert(span.dimensions.last >= 1)
        
        var rowIndex: Int
        if let index = span.dimensions.indexOf({ $0 > 1 }) {
            rowIndex = index
        } else {
            rowIndex = span.dimensions.count - 2
        }
        rows = span.dimensions[rowIndex]
        columns = span.dimensions.last!
        
        stride = span.dimensions.suffixFrom(rowIndex + 1).reduce(1, combine: *)
    }
    
    public subscript(row: Int, column: Int) -> Element {
        get {
            return self[[row, column]]
        }
        set {
            self[[row, column]] = newValue
        }
    }
    
    public subscript(indices: Index) -> Element {
        get {
            var index = span.startIndex
            let indexReplacementRage: Range<Int> = span.startIndex.count - indices.count..<span.startIndex.count
            index.replaceRange(indexReplacementRage, with: indices)
            assert(indexIsValid(index))
            return base[index]
        }
        set {
            var index = span.startIndex
            let indexReplacementRage: Range<Int> = span.startIndex.count - indices.count..<span.startIndex.count
            index.replaceRange(indexReplacementRage, with: indices)
            assert(indexIsValid(index))
            base[index] = newValue
        }
    }
    
    public subscript(slice: [IntervalType]) -> Slice {
        get {
            let span = Span(base: self.span, intervals: slice)
            return self[span]
        }
        set {
            let span = Span(base: self.span, intervals: slice)
            assert(span ≅ newValue.span)
            self[span] = newValue
        }
    }
    
    public subscript(slice: IntervalType...) -> Slice {
        get {
            return self[slice]
        }
        set {
            self[slice] = newValue
        }
    }
    
    subscript(span: Span) -> Slice {
        get {
            assert(self.span.contains(span))
            return Slice(base: base, span: span)
        }
        set {
            assert(self.span.contains(span))
            assert(span ≅ newValue.span)
            for (lhsIndex, rhsIndex) in zip(span, newValue.span)  {
                base[lhsIndex] = newValue[rhsIndex]
            }
        }
    }
    
    public var isContiguous: Bool {
        let onesCount: Int
        if let index = dimensions.indexOf({ $0 != 1 }) {
            onesCount = index
        } else {
            onesCount = rank
        }
        
        let diff = (0..<rank).map({ dimensions[$0] - base.dimensions[$0] }).reverse()
        let fullCount: Int
        if let index = diff.indexOf({ $0 != 0 }) where index.base < count {
            fullCount = diff.startIndex.distanceTo(index)
        } else {
            fullCount = rank
        }
        
        return rank - fullCount - onesCount <= 1
    }
    
    public func indexIsValid(indices: [Int]) -> Bool {
        assert(indices.count == rank)
        for (i, index) in indices.enumerate() {
            if index < span[i].startIndex || span[i].endIndex <= index {
                return false
            }
        }
        return true
    }
}

// MARK: - Equatable

public func ==<T>(lhs: TwoDimensionalTensorSlice<T>, rhs: TwoDimensionalTensorSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T: Equatable>(lhs: TwoDimensionalTensorSlice<T>, rhs: TensorSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T: Equatable>(lhs: TwoDimensionalTensorSlice<T>, rhs: Tensor<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T: Equatable>(lhs: TwoDimensionalTensorSlice<T>, rhs: Matrix<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T: Equatable>(lhs: TwoDimensionalTensorSlice<T>, rhs: MatrixSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}


