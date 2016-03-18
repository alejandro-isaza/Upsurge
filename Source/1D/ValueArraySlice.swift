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

/// A slice of a `ValueArray`. Slices not only specify start and end indexes, they also specify a step size.
public struct ValueArraySlice<Element: Value>: MutableLinearType, CustomStringConvertible, Equatable {
    public typealias Index = Int
    public typealias Slice = ValueArraySlice<Element>

    var base: ValueArray<Element>
    public var startIndex: Int
    public var endIndex: Int
    public var step: Int
    
    public var span: Span {
        return Span(ranges: [startIndex..<endIndex])
    }

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

    public init(base: ValueArray<Element>, startIndex: Int, endIndex: Int, step: Int) {
        assert(base.startIndex <= startIndex && endIndex <= base.endIndex)
        self.base = base
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.step = step
    }
    
    public subscript(index: Index) -> Element {
        get {
            assert(indexIsValid(index))
            return base[index * step]
        }
        set {
            assert(indexIsValid(index))
            base[index * step] = newValue
        }
    }
        
    public subscript(intervals: [IntervalType]) -> Slice {
        get {
            assert(self.span.contains(intervals))
            assert(intervals.count == 1)
            let start = intervals[0].start ?? startIndex
            let end = intervals[0].end ?? endIndex
            return Slice(base: base, startIndex: start, endIndex: end, step: step)
        }
        set {
            assert(self.span.contains(intervals))
            assert(intervals.count == 1)
            let start = intervals[0].start ?? startIndex
            let end = intervals[0].end ?? endIndex
            for i in start..<end {
                self[i] = newValue[i - start]
            }
        }
    }
    
    public subscript(intervals: [Int]) -> Element {
        get {
            assert(intervals.count == 1)
            return self[intervals[0]]
        }
        set {
            assert(intervals.count == 1)
            self[intervals[0]] = newValue
        }
    }
    
    public var description: String {
        var string = "["
        for i in startIndex.stride(to: endIndex, by: step) {
            string += "\(base[i]), "
        }
        if string.startIndex.distanceTo(string.endIndex) > 1 {
            let range = string.endIndex.advancedBy(-2)..<string.endIndex
            string.replaceRange(range, with: "]")
        } else {
            string += "]"
        }
        return string
    }
}

// MARK: - Equatable

public func ==<T>(lhs: ValueArraySlice<T>, rhs: ValueArray<T>) -> Bool {
    if lhs.count != rhs.count {
        return false
    }
    
    for (lhsIndex, rhsIndex) in zip(lhs.startIndex..<lhs.endIndex, 0..<rhs.count) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}

public func ==<T>(lhs: ValueArraySlice<T>, rhs: ValueArraySlice<T>) -> Bool {
    if lhs.count != rhs.count {
        return false
    }
    
    for (lhsIndex, rhsIndex) in zip(lhs.startIndex..<lhs.endIndex, rhs.startIndex..<rhs.endIndex) {
        if lhs[lhsIndex] != rhs[rhsIndex] {
            return false
        }
    }
    return true
}
