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

import Accelerate

/// A `Tensor` is a multi-dimensional collection of values.
public class Tensor<Element: Value>: MutableTensorType, Equatable {
    public typealias Index = [Int]
    public typealias Slice = TensorSlice<Element>

    public var dimensions: [Int]
    public var elements: ValueArray<Element>

    public func withUnsafeBufferPointer<R>(@noescape body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R {
        return try elements.withUnsafeBufferPointer(body)
    }

    public func withUnsafePointer<R>(@noescape body: (UnsafePointer<Element>) throws -> R) rethrows -> R {
        return try elements.withUnsafePointer(body)
    }

    public func withUnsafeMutableBufferPointer<R>(@noescape body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R {
        return try elements.withUnsafeMutableBufferPointer(body)
    }

    public func withUnsafeMutablePointer<R>(@noescape body: (UnsafeMutablePointer<Element>) throws -> R) rethrows -> R {
        return try elements.withUnsafeMutablePointer(body)
    }
    
    public var span: Span {
        return Span(zeroTo: dimensions)
    }

    public init<M: LinearType where M.Element == Element>(dimensions: [Int], elements: M) {
        assert(dimensions.reduce(1, combine: *) == elements.count)
        self.dimensions = dimensions
        self.elements = ValueArray(elements)
    }

    public init(_ tensor: Tensor<Element>) {
        self.dimensions = tensor.dimensions
        self.elements = ValueArray<Element>(tensor.elements)
    }
    
    public convenience init(_ tensorSlice: TensorSlice<Element>) {
        self.init(dimensions: tensorSlice.dimensions)
        for index in Span(zeroTo: dimensions) {
            self[index] = tensorSlice[index]
        }
    }

    public init(_ matrix: Matrix<Element>) {
        self.dimensions = [matrix.rows, matrix.columns]
        self.elements = matrix.elements
    }

    public init(dimensions: [Int]) {
        self.dimensions = dimensions
        self.elements = ValueArray(count: dimensions.reduce(1, combine: *))
    }

    public init(dimensions: [Int], repeatedValue: Element) {
        self.dimensions = dimensions
        self.elements = ValueArray(count: dimensions.reduce(1, combine: *), repeatedValue: repeatedValue)
    }

    public init(dimensions: [Int], initializer: () -> Element) {
        self.dimensions = dimensions
        self.elements = ValueArray(count: dimensions.reduce(1, combine: *), initializer: initializer)
    }
    
    public subscript(indices: Int...) -> Element {
        get {
            return self[indices]
        }
        set {
            self[indices] = newValue
        }
    }

    public subscript(indices: Index) -> Element {
        get {
            var index = [Int](count: dimensions.count, repeatedValue: 0)
            let indexReplacementRage: Range<Int> = dimensions.count - indices.count..<dimensions.count
            index.replaceRange(indexReplacementRage, with: indices)
            assert(indexIsValid(index))
            let elementsIndex = linearIndex(index)
            return elements[elementsIndex]
        }
        set {
            var index = [Int](count: dimensions.count, repeatedValue: 0)
            let indexReplacementRage: Range<Int> = dimensions.count - indices.count..<dimensions.count
            index.replaceRange(indexReplacementRage, with: indices)
            assert(indexIsValid(index))
            let elementsIndex = linearIndex(index)
            elements[elementsIndex] = newValue
        }
    }

    public subscript(slice: IntervalType...) -> TensorSlice<Element> {
        get {
            return self[slice]
        }
        set {
            self[slice] = newValue
        }
    }

    public subscript(slice: [IntervalType]) -> TensorSlice<Element> {
        get {
            let subSpan = Span(base: span, intervals: slice)
            return self[subSpan]
        }
        set {
            let subSpan = Span(base: span, intervals: slice)
            self[subSpan] = newValue
        }
    }

    subscript(span: Span) -> TensorSlice<Element> {
        get {
            assert(spanIsValid(span))
            return TensorSlice(base: self, span: span)
        }
        set {
            assert(spanIsValid(span))
            assert(span ≅ newValue.span)
            let tensorSlice = TensorSlice(base: self, span: span)
            tensorSlice[tensorSlice.span] = newValue
        }
    }
    
    public func reshape(dimensions: Int...) {
        precondition(dimensions.reduce(1, combine: *) == count)
        self.dimensions = dimensions
    }

    public func copy() -> Tensor {
        return Tensor(self)
    }
    
    func spanIsValid(subSpan: Span) -> Bool {
        let span = Span(zeroTo: dimensions)
        return span.contains(subSpan)
    }
}

// MARK: - Matrix Extraction

extension Tensor {
    /**
     Extract a matrix from the tensor.
     
     - Precondition: All but the last two intervals must be a specific index, not a range. The last interval must either span the full dimension, or the second-last interval count must be 1.
     */
    func asMatrix(span: Span) -> TwoDimensionalTensorSlice<Element> {
        return TwoDimensionalTensorSlice(base: self, span: span)
    }
    
    public func asMatrix(intervals: IntervalType...) -> TwoDimensionalTensorSlice<Element> {
        let baseSpan = Span(zeroTo: dimensions)
        let matrixSpan = Span(base: baseSpan, intervals: intervals)
        return asMatrix(matrixSpan)
    }

}

// MARK: -

public func swap<T>(lhs: Tensor<T>, rhs: Tensor<T>) {
    swap(&lhs.dimensions, &rhs.dimensions)
    swap(&lhs.elements, &rhs.elements)
}

// MARK: - Equatable

public func ==<T>(lhs: Tensor<T>, rhs: Tensor<T>) -> Bool {
    return lhs.elements == rhs.elements
}

public func ==<T>(lhs: Tensor<T>, rhs: TensorSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T>(lhs: Tensor<T>, rhs: Matrix<T>) -> Bool {
    return lhs.elements == rhs.elements
}

public func ==<T>(lhs: Tensor<T>, rhs: MatrixSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T>(lhs: Tensor<T>, rhs: TwoDimensionalTensorSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}
