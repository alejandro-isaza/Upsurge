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

/// A `ValueArray` is similar to an `Array` but it's a `class` instead of a `struct` and it has a fixed size. As opposed to an `Array`, assigning a `ValueArray` to a new variable will not create a copy, it only creates a new reference. If any reference is modified all other references will reflect the change. To copy a `ValueArray` you have to explicitly call `copy()`.
open class ValueArray<Element: Value>: MutableLinearType, ExpressibleByArrayLiteral, CustomStringConvertible, Equatable {
    public typealias Index = Int
    public typealias IndexDistance = Int
    public typealias Slice = ValueArraySlice<Element>

    internal(set) var mutablePointer: UnsafeMutablePointer<Element>
    open internal(set) var capacity: IndexDistance
    open internal(set) var count: IndexDistance

    open var startIndex: Index {
        return 0
    }

    open var endIndex: Index {
        return count
    }

    open var step: IndexDistance {
        return 1
    }

    open var span: Span {
        return Span(zeroTo: [endIndex])
    }

    open func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R {
        return try body(UnsafeBufferPointer(start: mutablePointer, count: count))
    }

    open func withUnsafePointer<R>(_ body: (UnsafePointer<Element>) throws -> R) rethrows -> R {
        return try body(mutablePointer)
    }

    open func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R {
        return try body(UnsafeMutableBufferPointer(start: mutablePointer, count: count))
    }

    open func withUnsafeMutablePointer<R>(_ body: (UnsafeMutablePointer<Element>) throws -> R) rethrows -> R {
        return try body(mutablePointer)
    }

    open var pointer: UnsafePointer<Element> {
        return UnsafePointer(mutablePointer)
    }

    /// Construct an uninitialized ValueArray with the given capacity
    public required init(capacity: IndexDistance) {
        mutablePointer = UnsafeMutablePointer.allocate(capacity: capacity)
        self.capacity = capacity
        self.count = 0
    }

    /// Construct an uninitialized ValueArray with the given size
    public required init(count: IndexDistance) {
        mutablePointer = UnsafeMutablePointer.allocate(capacity: count)
        self.capacity = count
        self.count = count
    }

    /// Construct a ValueArray from an array literal
    public required init(arrayLiteral elements: Element...) {
        mutablePointer = UnsafeMutablePointer.allocate(capacity: elements.count)
        self.capacity = elements.count
        self.count = elements.count
        mutablePointer.initialize(from: elements)
    }

    /// Construct a ValueArray from contiguous memory
    public required init<C: LinearType>(_ values: C) where C.Element == Element {
        mutablePointer = UnsafeMutablePointer.allocate(capacity: values.count)
        capacity = values.count
        count = values.count
        values.withUnsafeBufferPointer { pointer in
            for i in 0..<count {
                mutablePointer[i] = pointer[values.startIndex + i * values.step]
            }
        }
    }

    /// Construct a ValueArray of `count` elements, each initialized to `repeatedValue`.
    public required convenience init(count: IndexDistance, repeatedValue: Element) {
        self.init(count: count) { repeatedValue }
    }

    /// Construct a ValueArray of `count` elements, each initialized with `initializer`.
    public required init(count: IndexDistance, initializer: () -> Element) {
        mutablePointer = UnsafeMutablePointer.allocate(capacity: count)
        capacity = count
        self.count = count
        for i in 0..<count {
            self[i] = initializer()
        }
    }

    deinit {
        mutablePointer.deallocate(capacity: capacity)
    }

    open subscript(index: Index) -> Element {
        get {
            assert(indexIsValid(index))
            return pointer[index * step]
        }
        set {
            assert(indexIsValid(index))
            mutablePointer[index * step] = newValue
        }
    }

    open subscript(intervals: [IntervalType]) -> Slice {
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
                self[i] = newValue[newValue.startIndex + i - start]
            }
        }
    }

    open subscript(intervals: IntervalType...) -> Slice {
        get {
            return self[intervals]
        }
        set {
            self[intervals] = newValue
        }
    }

    open subscript(intervals: [Int]) -> Element {
        get {
            assert(intervals.count == 1)
            return self[intervals[0]]
        }
        set {
            assert(intervals.count == 1)
            self[intervals[0]] = newValue
        }
    }

    open func index(after i: Index) -> Index {
        return i + 1
    }

    open func formIndex(after i: inout Index) {
        i += 1
    }

    open func copy() -> ValueArray {
        let copy = ValueArray(count: capacity)
        copy.mutablePointer.initialize(from: mutablePointer, count: count)
        return copy
    }

    open func append(_ value: Element) {
        precondition(count + 1 <= capacity)
        mutablePointer[count] = value
        count += 1
    }

    open func appendContentsOf<C: Collection>(_ values: C) where C.Iterator.Element == Element {
        precondition(count + Int(values.count.toIntMax()) <= capacity)
        let endPointer = mutablePointer + count
        endPointer.initialize(from: values)
        count += Int(values.count.toIntMax())
    }

    open func replaceRange<C: Collection>(_ subRange: Range<Index>, with newElements: C) where C.Iterator.Element == Element {
        assert(subRange.lowerBound >= startIndex && subRange.upperBound <= endIndex)
        (mutablePointer + subRange.lowerBound).initialize(from: newElements)
    }

    open func toRowMatrix() -> Matrix<Element> {
        return Matrix(rows: 1, columns: count, elements: self)
    }

    open func toColumnMatrix() -> Matrix<Element> {
        return Matrix(rows: count, columns: 1, elements: self)
    }

    open var description: String {
        return "[\(map { "\($0)" }.joined(separator: ", "))]"
    }

    open var debugDescription: String {
        return description
    }

    // MARK: - Equatable

    public static func ==(lhs: ValueArray, rhs: ValueArray) -> Bool {
        return lhs.count == rhs.count && lhs.elementsEqual(rhs)
    }

    public static func ==(lhs: ValueArray, rhs: Slice) -> Bool {
        return lhs.count == rhs.count && lhs.elementsEqual(rhs)
    }
}

// MARK: -

public func swap<T>(_ lhs: inout ValueArray<T>, rhs: inout ValueArray<T>) {
    swap(&lhs.mutablePointer, &rhs.mutablePointer)
    swap(&lhs.capacity, &rhs.capacity)
    swap(&lhs.count, &rhs.count)
}
