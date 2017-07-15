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

/// Span is a collection of Ranges to specify a multi-dimensional slice of a Tensor.
public struct Span: ExpressibleByArrayLiteral, Sequence {
    public typealias Element = [Int]

    private var ranges: [Element]

    public var startIndex: [Int] {
        return ranges.map { $0.first! }
    }

    public var endIndex: [Int] {
        return ranges.map { $0.last! + 1 }
    }

    public var count: Int {
        return dimensions.reduce(1, *)
    }

    public var rank: Int {
        return ranges.count
    }

    public var dimensions: [Int] {
        return ranges.map { $0.count }
    }

    public init(ranges: [Element]) {
        self.ranges = ranges
    }

    public init(arrayLiteral elements: Element...) {
        self.init(ranges: elements)
    }

    public init(base: Span, intervals: [IntervalType]) {
        assert(base.contains(intervals))
        var ranges = [Element]()

        for (i, interval) in intervals.enumerated() {
            let start = interval.start ?? base[i].first!
            let end = interval.end ?? base[i].last! + 1
            assert(base[i].first! <= start && end <= base[i].last! + 1)
            ranges.append((start ... end - 1).map { $0 })
        }
        self.init(ranges: ranges)
    }

    public init(dimensions: [Int], intervals: [IntervalType]) {
        var ranges = [Element]()
        for i in 0..<intervals.count {
            let start = intervals[i].start ?? 0
            let end = intervals[i].end ?? dimensions[i]
            assert(0 <= start && end <= dimensions[i])
            ranges.append((start ... end - 1).map { $0 })
        }
        self.init(ranges: ranges)
    }

    public init(zeroTo dimensions: [Int]) {
        let start = [Int](repeating: 0, count: dimensions.count)
        self.init(start: start, end: dimensions)
    }

    public init(start: [Int], end: [Int]) {
        ranges = zip(start, end).map { ($0...$1 - 1).map { $0 } }
    }

    public init(start: [Int], length: [Int]) {
        let end = zip(start, length).map { $0 + $1 }
        self.init(start: start, end: end)
    }

    public func makeIterator() -> SpanGenerator {
        return SpanGenerator(span: self)
    }

    public subscript(index: Int) -> Element {
        return ranges[index]
    }

    public subscript(range: ClosedRange<Int>) -> ArraySlice<Element> {
        return ranges[range]
    }

    public subscript(range: Range<Int>) -> ArraySlice<Element> {
        return ranges[range]
    }

    public func contains(_ other: Span) -> Bool {
        return (0..<dimensions.count).all {
            self[$0].startIndex <= other[$0].startIndex && other[$0].endIndex <= self[$0].endIndex
        }
    }

    public func contains(_ intervals: [IntervalType]) -> Bool {
        assert(dimensions.count == intervals.count)
        for i in 0..<dimensions.count {
            let start = intervals[i].start ?? self[i].first!
            let end = intervals[i].end ?? self[i].last! + 1
            if start < self[i].first! || self[i].last! + 1 < end {
                return false
            }
        }
        return true
    }
}

public class SpanGenerator: IteratorProtocol {
    private var span: Span
    private var presentIndex: [Int]
    private var first = true

    public init(span: Span) {
        self.span = span
        self.presentIndex = span.startIndex.map { $0 }
    }

    public func next() -> [Int]? {
        if presentIndex.isEmpty {
            return nil
        }
        if first {
            first = false
            return presentIndex
        }
        if !incrementIndex(presentIndex.count - 1) {
            return nil
        }
        return presentIndex
    }

    private func incrementIndex(_ position: Int) -> Bool {
        if position < 0 || span.count <= position {
            return false
        }

        if presentIndex[position] < span[position].last! {
            presentIndex[position] += 1
        } else {
            if !incrementIndex(position - 1) {
                return false
            }
            presentIndex[position] = span[position].first!
        }

        return true
    }
}

// MARK: - Dimensional Congruency

infix operator ≅ : ComparisonPrecedence
public func ≅(lhs: Span, rhs: Span) -> Bool {
    if lhs.dimensions == rhs.dimensions {
        return true
    }

    let (max, min) = lhs.dimensions.count > rhs.dimensions.count ? (lhs, rhs) : (rhs, lhs)
    let diff = max.dimensions.count - min.dimensions.count
    return max.dimensions[0..<diff].reduce(1, *) == 1 && Array(max.dimensions[diff..<max.dimensions.count]) == min.dimensions
}
