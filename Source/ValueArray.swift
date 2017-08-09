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

/// A `ValueArray` is similar to an `Array` but it's a `class` instead of a `struct` and it has a fixed size. As opposed to an `Array`, assigning a `ValueArray` to a new variable will not create a copy, it only creates a new reference. If any reference is modified all other references will reflect the change. To copy a `ValueArray` you have to explicitly call `copy()`.
open class ValueArray<Element: Value>: MutableLinearType, ExpressibleByArrayLiteral, CustomStringConvertible, Equatable {
    public typealias Index = Int
    public typealias IndexDistance = Int
    public typealias Slice = ValueArraySlice<Element>
    
    public static func calculatePageAlignedCapacity(
        _ capacity: Int,
        extraPage: Bool = true)
        -> (pagedCapacity: Int, bytesCapacity: Int)
    {
        let pageSize = 4096//Int(getpagesize())
        let elementSize = MemoryLayout<Element>.size
        let dataSize = capacity * elementSize
        let pageCount = (dataSize + (pageSize - 1)) / pageSize
        let bytesCapacity = extraPage ? (pageCount + 1) * pageSize : pageCount * pageSize
        let pagedCapacity = bytesCapacity / elementSize
        return (pagedCapacity: pagedCapacity, bytesCapacity: bytesCapacity)
    }
    
    public static func forcePageAlignedCapacity(_ capacity: Int) -> (pagedPointer: UnsafeMutablePointer<Element>, pagedCapacity: Int, bytesCapacity: Int) {
        
        let pageSize = 4096//Int(getpagesize())
        let (pagedCapacity, bytesCapacity) = calculatePageAlignedCapacity(capacity)
        
        var mutablePointer: UnsafeMutableRawPointer? = nil
        
        let ret = posix_memalign(&mutablePointer, pageSize, bytesCapacity)
        
        if ret != noErr {
            let err = String(validatingUTF8: strerror(ret)) ?? "unknown error"
            fatalError("Unable to allocate aligned memory: \(err).")
        }
        
        return (pagedPointer: mutablePointer!.assumingMemoryBound(to: Element.self), pagedCapacity: pagedCapacity, bytesCapacity: bytesCapacity)
    }

    convenience required public init() {
      self.init(count: 0)
    }

    public internal(set) var mutablePointer: UnsafeMutablePointer<Element>
    private var unownedPointer: Bool = false

    open internal(set) var bytesCapacity: Int
    
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
        let (pagedPointer, pagedCapacity, bytesCapacity) = ValueArray.forcePageAlignedCapacity(capacity)
        self.mutablePointer = pagedPointer
        self.bytesCapacity = bytesCapacity
        self.capacity = pagedCapacity
        self.count = 0
    }

    /// Construct an uninitialized ValueArray with the given size
    public required init(count: IndexDistance) {
        let (pagedPointer, pagedCapacity, bytesCapacity) = ValueArray.forcePageAlignedCapacity(count)
        self.mutablePointer = pagedPointer
        self.bytesCapacity = bytesCapacity
        self.capacity = pagedCapacity
        self.count = count
    }

    /// Construct a ValueArray from an array literal
    public required init(arrayLiteral elements: Element...) {
        let (pagedPointer, pagedCapacity, bytesCapacity) = ValueArray.forcePageAlignedCapacity(elements.count)
        self.mutablePointer = pagedPointer
        self.bytesCapacity = bytesCapacity
        self.capacity = pagedCapacity
        self.count = elements.count
        _ = UnsafeMutableBufferPointer(start: mutablePointer, count: count).initialize(from: elements)
    }

    /// Construct a ValueArray from contiguous memory
    public required init<C: LinearType>(_ values: C) where C.Element == Element {
        let (pagedPointer, pagedCapacity, bytesCapacity) = ValueArray.forcePageAlignedCapacity(values.count)
        self.mutablePointer = pagedPointer
        self.bytesCapacity = bytesCapacity
        self.capacity = pagedCapacity
        self.count = values.count
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
    
    /// Construct a ValueArray of `count` elements, each initialized to `repeatedValue`.
    public required convenience init(repeating repeatedValue: Element, count: Int) {
        self.init(count: count) { repeatedValue }
    }
    
    
    
    public required init(
        unownedMutablePointer: UnsafeMutablePointer<Element>,
        count: Int)
    {
        self.mutablePointer = unownedMutablePointer
        self.capacity = count
        self.bytesCapacity = ValueArray.calculatePageAlignedCapacity(count, extraPage: false).bytesCapacity
        self.count = count
        self.unownedPointer = true
    }
    
    public required init(
        repeating repeatedValue: Element,
        unownedMutablePointer: UnsafeMutablePointer<Element>,
        count: Int)
    {
        self.mutablePointer = unownedMutablePointer
        self.capacity = count
        self.bytesCapacity = ValueArray.calculatePageAlignedCapacity(count, extraPage: false).bytesCapacity
        self.count = count
        self.unownedPointer = true
        
        for index in 0 ..< count {
            mutablePointer[index] = repeatedValue
        }
    }

    /// Construct a ValueArray of `count` elements, each initialized with `initializer`.
    public required init(count: IndexDistance, initializer: () -> Element) {
        let (pagedPointer, pagedCapacity, bytesCapacity) = ValueArray.forcePageAlignedCapacity(count)
        self.mutablePointer = pagedPointer
        self.bytesCapacity = bytesCapacity
        self.capacity = pagedCapacity
        self.count = count
        for i in 0..<count {
            self[i] = initializer()
        }
    }

    deinit {
        if !unownedPointer {
            mutablePointer.deallocate(capacity: capacity)
        }
    }

    open subscript(index: Index) -> Element {
        get {
            assert(indexIsValid(index))
            return pointer[index]
        }
        set {
            assert(indexIsValid(index))
            mutablePointer[index] = newValue
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

    open func copy() -> ValueArray {
        let copy = ValueArray(count: capacity)
        copy.mutablePointer.initialize(from: mutablePointer, count: count)
        return copy
    }

    open func append(_ newElement: Element) {
        precondition(count + 1 <= capacity)
        mutablePointer[count] = newElement
        count += 1
    }

    open func append<S: Sequence>(contentsOf newElements: S) where S.Iterator.Element == Element {
        let a = Array(newElements)
        precondition(count + a.count <= capacity)
        let endPointer = mutablePointer + count
        _ = UnsafeMutableBufferPointer(start: endPointer, count: capacity - count).initialize(from: a)
        count += a.count
    }

    open func replaceSubrange<C: Collection>(_ subrange: Range<Index>, with newElements: C) where C.Iterator.Element == Element {
        assert(subrange.lowerBound >= startIndex && subrange.upperBound <= endIndex)
        _ = UnsafeMutableBufferPointer(start: mutablePointer + subrange.lowerBound, count: capacity - subrange.lowerBound).initialize(from: newElements)
    }

    open func toRowMatrix() -> Matrix<Element> {
        return Matrix(rows: 1, columns: count, elements: self)
    }

    open func toColumnMatrix() -> Matrix<Element> {
        return Matrix(rows: count, columns: 1, elements: self)
    }

    open func toMatrix(rows: Int, columns: Int) -> Matrix<Element> {
        precondition(rows*columns == count, "Element count must equal rows*columns")
        return Matrix(rows: rows, columns: columns, elements: self)
    }

    open func tile(_ m: Int, _ n: Int) -> Matrix<Element> {
        // Construct a block matrix of size m by n, with a copy of source as each element.
        // m:  Specifies the number of times to copy along the vertical axis.
        // n:  Specifies the number of times to copy along the horizontal axis.
        precondition(m > 0 && n > 0, "Minimum of 1 repeat in each direction is required")
        let results = ValueArray(count: m*n*count)
        let typeMemorySize = MemoryLayout<Element>.size
        let bytesInSource = count*typeMemorySize
        for i in 0..<m*n {
            memcpy(results.mutablePointer+(i*count), pointer, bytesInSource)
        }
        return results.toMatrix(rows: m, columns: n*count)
    }

    // MARK: - Equatable

    public static func == (lhs: ValueArray, rhs: ValueArray) -> Bool {
        return lhs.count == rhs.count && lhs.elementsEqual(rhs)
    }

    public static func == (lhs: ValueArray, rhs: Slice) -> Bool {
        return lhs.count == rhs.count && lhs.elementsEqual(rhs)
    }
}

// MARK: -

public func swap<T>(_ lhs: inout ValueArray<T>, rhs: inout ValueArray<T>) {
    swap(&lhs.mutablePointer, &rhs.mutablePointer)
    swap(&lhs.capacity, &rhs.capacity)
    swap(&lhs.count, &rhs.count)
}
