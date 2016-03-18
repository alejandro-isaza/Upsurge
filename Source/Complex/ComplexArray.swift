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

public class ComplexArray<T: Real>: MutableLinearType, ArrayLiteralConvertible  {
    public typealias Index = Int
    public typealias Element = Complex<T>
    public typealias Slice = ComplexArraySlice<T>

    var elements: ValueArray<Complex<T>>

    public var count: Int {
        get {
            return elements.count
        }
        set {
            elements.count = newValue
        }
    }

    public var capacity: Int {
        return elements.capacity
    }

    public var startIndex: Index {
        return 0
    }

    public var endIndex: Index {
        return count
    }

    public var step: Index {
        return 1
    }
    
    public var span: Span {
        return Span(zeroTo: [endIndex])
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

    public var reals: ComplexArrayRealSlice<T> {
        get {
            return ComplexArrayRealSlice<T>(base: self, startIndex: startIndex, endIndex: 2*endIndex - 1, step: 2)
        }
        set {
            precondition(newValue.count == reals.count)
            for i in 0..<newValue.count {
                self.reals[i] = newValue[i]
            }
        }
    }

    public var imags: ComplexArrayRealSlice<T> {
        get {
            return ComplexArrayRealSlice<T>(base: self, startIndex: startIndex + 1, endIndex: 2*endIndex, step: 2)
        }
        set {
            precondition(newValue.count == imags.count)
            for i in 0..<newValue.count {
                self.imags[i] = newValue[i]
            }
        }
    }

    /// Construct an uninitialized ComplexArray with the given capacity
    public required init(capacity: Int) {
        elements = ValueArray<Complex<T>>(capacity: capacity)
    }

    /// Construct an uninitialized ComplexArray with the given size
    public required init(count: Int) {
        elements = ValueArray<Complex<T>>(count: count)
    }

    /// Construct a ComplexArray from an array literal
    public required init(arrayLiteral elements: Element...) {
        self.elements = ValueArray<Complex<T>>(count: elements.count)
        self.elements.mutablePointer.initializeFrom(elements)
    }

    /// Construct a ComplexArray from contiguous memory
    public required init<C : LinearType where C.Element == Element>(_ values: C) {
        elements = ValueArray<Complex<T>>(values)
    }

    /// Construct a ComplexArray of `count` elements, each initialized to `repeatedValue`.
    public required init(count: Int, repeatedValue: Element) {
        elements = ValueArray<Complex<T>>(count: count, repeatedValue: repeatedValue)
    }

    public subscript(index: Index) -> Element {
        get {
            precondition(0 <= index && index < capacity)
            assert(index < count)
            return elements[index]
        }
        set {
            precondition(0 <= index && index < capacity)
            assert(index < count)
            elements[index] = newValue
        }
    }
    
    public subscript(indices: [Int]) -> Element {
        get {
            assert(indices.count == 1)
            return self[indices[0]]
        }
        set {
            assert(indices.count == 1)
            self[indices[0]] = newValue
        }
    }
    
    public subscript(intervals: [IntervalType]) -> Slice {
        get {
            assert(intervals.count == 1)
            let start = intervals[0].start ?? startIndex
            let end = intervals[0].end ?? endIndex
            return Slice(base: self, startIndex: start, endIndex: end, step: step)
        }
        set {
            assert(intervals.count == 1)
            let start = intervals[0].start ?? startIndex
            let end = intervals[0].end ?? endIndex
            assert(startIndex <= start && end <= endIndex)
            for i in start..<end {
                self[i] = newValue[i - start]
            }
        }
    }

    public func copy() -> ComplexArray {
        return ComplexArray(elements)
    }

    public func append(value: Element) {
        elements.append(value)
    }

    public func appendContentsOf<C : CollectionType where C.Generator.Element == Element>(values: C) {
        elements.appendContentsOf(values)
    }

    public func replaceRange<C: CollectionType where C.Generator.Element == Element>(subRange: Range<Index>, with newElements: C) {
        elements.replaceRange(subRange, with: newElements)
    }

    public func toRowMatrix() -> Matrix<Element> {
        return Matrix(rows: 1, columns: count, elements: self)
    }

    public func toColumnMatrix() -> Matrix<Element> {
        return Matrix(rows: count, columns: 1, elements: self)
    }
}

public func ==<T: Real>(lhs: ComplexArray<T>, rhs: ComplexArray<T>) -> Bool {
    if lhs.count != rhs.count {
        return false
    }

    for i in 0..<lhs.count {
        if lhs[i] != rhs[i] {
            return false
        }
    }
    return true
}
