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
    public typealias Element = CountableClosedRange<Int>

    private var ranges: [Element]

    var startIndex: [Int] {
        return ranges.map{ $0.lowerBound }
    }

    var endIndex: [Int] {
        return ranges.map{ $0.upperBound + 1 }
    }
    
    var count: Int {
        return dimensions.reduce(1, *)
    }
    
    var rank: Int {
        return ranges.count
    }
    
    var dimensions: [Int] {
        return ranges.map{ $0.count }
    }
    
    init(ranges: [Element]) {
        self.ranges = ranges
    }
    
    public init(arrayLiteral elements: Element...) {
        self.init(ranges: elements)
    }
    
    init(base: Span, intervals: [IntervalType]) {
        assert(base.contains(intervals))
        var ranges = [Element]()

        for (i,interval) in intervals.enumerated() {
            let start = interval.start ?? base[i].lowerBound
            let end = interval.end ?? base[i].upperBound + 1
            assert(base[i].lowerBound <= start && end <= base[i].upperBound + 1)
            ranges.append(start ... end - 1)
        }
        self.init(ranges: ranges)
    }
    
    init(dimensions: [Int], intervals: [IntervalType]) {
        var ranges = [Element]()
        for i in 0..<intervals.count {
            let start = intervals[i].start ?? 0
            let end = intervals[i].end ?? dimensions[i]
            assert(0 <= start && end <= dimensions[i])
            ranges.append(start ... end - 1)
        }
        self.init(ranges: ranges)
    }
    
    init(zeroTo dimensions: [Int]) {
        let start = [Int](repeating: 0, count: dimensions.count)
        self.init(start: start, end: dimensions)
    }
    
    init(start: [Int], end: [Int]) {
        ranges = zip(start, end).map{ $0...$1 - 1 }
    }
    
    init(start: [Int], length: [Int]) {
        let end = zip(start, length).map{ $0 + $1 }
        self.init(start: start, end: end)
    }
    
    public func makeIterator() -> SpanGenerator {
        return SpanGenerator(span: self)
    }
    
    subscript(index: Int) -> Element {
        return ranges[index]
    }

    subscript(range: ClosedRange<Int>) -> ArraySlice<Element> {
        return ranges[range]
    }

    subscript(range: Range<Int>) -> ArraySlice<Element> {
        return ranges[range]
    }
    
    func contains(_ other: Span) -> Bool {
        for i in 0..<dimensions.count {
            if other[i].startIndex < self[i].startIndex || self[i].endIndex < other[i].endIndex {
                return false
            }
        }
        return true
    }
    
    func contains(_ intervals: [IntervalType]) -> Bool {
        assert(dimensions.count == intervals.count)
        for i in 0..<dimensions.count {
            let start = intervals[i].start ?? self[i].lowerBound
            let end = intervals[i].end ?? self[i].upperBound + 1
            if start < self[i].lowerBound || self[i].upperBound + 1 < end {
                return false
            }
        }
        return true
    }
}

open class SpanGenerator: IteratorProtocol {
    fileprivate var span: Span
    fileprivate var presentIndex: [Int]
    fileprivate var kill = false
    
    init(span: Span) {
        self.span = span
        self.presentIndex = span.startIndex.map{ $0 }
    }
    
    open func next() -> [Int]? {
        return incrementIndex(presentIndex.count - 1)
    }
    
    func incrementIndex(_ position: Int) -> [Int]? {
        if position < 0 || span.count <= position || kill {
            return nil
        } else if presentIndex[position] < span[position].upperBound {
            let result = presentIndex
            presentIndex[position] += 1
            return result
        } else {
            guard let result = incrementIndex(position - 1) else {
                kill = true
                return presentIndex
            }
            presentIndex[position] = span[position].lowerBound
            return result
        }
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
