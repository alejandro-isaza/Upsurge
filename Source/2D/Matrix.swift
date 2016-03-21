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

public class Matrix<T: Value>: MutableQuadraticType, Equatable, CustomStringConvertible {
    public typealias Index = (Int, Int)
    public typealias Slice = MatrixSlice<Element>
    public typealias Element = T
    
    public var rows: Int
    public var columns: Int
    public var elements: ValueArray<Element>
    
    public var span: Span {
        return Span(zeroTo: [rows, columns])
    }

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

    public var arrangement: QuadraticArrangement {
        return .RowMajor
    }

    public var stride: Int {
        return columns
    }
    
    public var step: Int {
        return elements.step
    }

    /// Construct a Matrix from a `QuadraticType`
    public init<M: QuadraticType where M.Element == Element>(_ quad: M) {
        rows = quad.rows
        columns = quad.columns
        elements = ValueArray(count: rows * columns)
        quad.withUnsafeBufferPointer { pointer in
            for row in 0..<rows {
                let sourcePointer = UnsafeMutablePointer<Element>(pointer.baseAddress + (row * quad.stride))
                let destPointer = elements.mutablePointer + row * columns
                destPointer.assignFrom(sourcePointer, count: columns)
            }
        }
    }

    /// Construct a Matrix of `rows` by `columns` with every the given elements in row-major order
    public init<M: LinearType where M.Element == Element>(rows: Int, columns: Int, elements: M) {
        assert(rows * columns == elements.count)
        self.rows = rows
        self.columns = columns
        self.elements = ValueArray(elements)
    }

    /// Construct a Matrix of `rows` by `columns` with uninitialized elements
    public init(rows: Int, columns: Int) {
        self.rows = rows
        self.columns = columns
        self.elements = ValueArray(count: rows * columns)
    }

    /// Construct a Matrix of `rows` by `columns` with elements initialized to repeatedValue
    public init(rows: Int, columns: Int, repeatedValue: Element) {
        self.rows = rows
        self.columns = columns
        self.elements = ValueArray(count: rows * columns, repeatedValue: repeatedValue)
    }
    
    /// Construct a Matrix from an array of rows
    public convenience init(_ contents: [[Element]]) {
        let rows = contents.count
        let cols = contents[0].count

        self.init(rows: rows, columns: cols)

        for (i, row) in contents.enumerate() {
            elements.replaceRange(i*cols..<i*cols+min(cols, row.count), with: row)
        }
    }
    
    /// Construct a Matrix from an array of rows
    public init(rows: Int, columns: Int, initializer: () -> Element) {
        self.rows = rows
        self.columns = columns
        self.elements = ValueArray(count: rows * columns, initializer: initializer)
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
            assert(indexIsValidForRow(indices[0], column: indices[1]))
            return elements[(indices[0] * columns) + indices[1]]
        }
        set {
            assert(indices.count == 2)
            assert(indexIsValidForRow(indices[0], column: indices[1]))
            elements[(indices[0] * columns) + indices[1]] = newValue
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
            let span = Span(dimensions: dimensions, intervals: intervals)
            return self[span]
        }
        set {
            let span = Span(dimensions: dimensions, intervals: intervals)
            self[span] = newValue
        }
    }
    
    subscript(span: Span) -> Slice {
        get {
            return MatrixSlice(base: self, span: span)
        }
        set {
            assert(span ≅ newValue.span)
            for (lhsIndex, rhsIndex) in zip(span, newValue.span) {
                self[lhsIndex] = newValue[rhsIndex]
            }
        }
    }

    public func row(index: Int) -> ValueArraySlice<Element> {
        return ValueArraySlice<Element>(base: elements, startIndex: index * columns, endIndex: (index + 1) * columns, step: 1)
    }

    public func column(index: Int) -> ValueArraySlice<Element> {
        return ValueArraySlice<Element>(base: elements, startIndex: index, endIndex: rows * columns - columns + index + 1, step: columns)
    }

    public func copy() -> Matrix {
        let copy = elements.copy()
        return Matrix(rows: rows, columns: columns, elements: copy)
    }

    public func indexIsValidForRow(row: Int, column: Int) -> Bool {
        return row >= 0 && row < rows && column >= 0 && column < columns
    }
    
    public var description: String {
        var description = ""

        for i in 0..<rows {
            let contents = (0..<columns).map{"\(self[i, $0])"}.joinWithSeparator("\t")

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

public func ==<T>(lhs: Matrix<T>, rhs: Matrix<T>) -> Bool {
    return lhs.elements == rhs.elements
}

public func ==<T>(lhs: Matrix<T>, rhs: MatrixSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T>(lhs: Matrix<T>, rhs: Tensor<T>) -> Bool {
    return lhs.elements == rhs.elements
}

public func ==<T>(lhs: Matrix<T>, rhs: TensorSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T>(lhs: Matrix<T>, rhs: TwoDimensionalTensorSlice<T>) -> Bool {
    assert(lhs.span ≅ rhs.span)
    for (lhsIndex, rhsIndex) in zip(lhs.span, rhs.span) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

// MARK: -

public func swap<T>(lhs: Matrix<T>, rhs: Matrix<T>) {
    swap(&lhs.rows, &rhs.rows)
    swap(&lhs.columns, &rhs.columns)
    swap(&lhs.elements, &rhs.elements)
}
