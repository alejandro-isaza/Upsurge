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
public struct Span: ArrayLiteralConvertible, SequenceType {
    public typealias Element = Range<Int>
    
    var ranges: [Element]

    var startIndex: [Int] {
        return ranges.map{ $0.startIndex }
    }

    var endIndex: [Int] {
        return ranges.map{ $0.endIndex }
    }
    
    var count: Int {
        return dimensions.reduce(1, combine: *)
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
        for i in 0..<intervals.count {
            let start = intervals[i].start ?? base[i].startIndex
            let end = intervals[i].end ?? base[i].endIndex
            assert(base[i].startIndex <= start && end <= base[i].endIndex)
            ranges.append(start..<end)
        }
        self.init(ranges: ranges)
    }
    
    init(dimensions: [Int], intervals: [IntervalType]) {
        var ranges = [Element]()
        for i in 0..<intervals.count {
            let start = intervals[i].start ?? 0
            let end = intervals[i].end ?? dimensions[i]
            assert(0 <= start && end <= dimensions[i])
            ranges.append(start..<end)
        }
        self.init(ranges: ranges)
    }
    
    init(zeroTo dimensions: [Int]) {
        let start = [Int](count: dimensions.count, repeatedValue: 0)
        self.init(start: start, end: dimensions)
    }
    
    init(start: [Int], end: [Int]) {
        ranges = zip(start, end).map{ $0..<$1 }
    }
    
    init(start: [Int], length: [Int]) {
        let end = zip(start, length).map{ $0 + $1 }
        self.init(start: start, end: end)
    }
    
    public func generate() -> SpanGenerator {
        return SpanGenerator(span: self)
    }
    
    subscript(index: Int) -> Element {
        return self.ranges[index]
    }
    
    func contains(other: Span) -> Bool {
        for i in 0..<dimensions.count {
            if other[i].startIndex < self[i].startIndex || self[i].endIndex < other[i].endIndex {
                return false
            }
        }
        return true
    }
    
    func contains(intervals: [IntervalType]) -> Bool {
        assert(dimensions.count == intervals.count)
        for i in 0..<dimensions.count {
            let start = intervals[i].start ?? self[i].startIndex
            let end = intervals[i].end ?? self[i].endIndex
            if start < self[i].startIndex || self[i].endIndex < end {
                return false
            }
        }
        return true
    }
}

public class SpanGenerator: GeneratorType {
    private var span: Span
    private var presentIndex: [Int]
    private var kill = false
    
    init(span: Span) {
        self.span = span
        self.presentIndex = span.startIndex.map{ $0 }
    }
    
    public func next() -> [Int]? {
        return incrementIndex(presentIndex.count - 1)
    }
    
    func incrementIndex(position: Int) -> [Int]? {
        if position < 0 || span.count <= position || kill {
            return nil
        } else if presentIndex[position] + 1 < span[position].endIndex {
            let result = presentIndex
            presentIndex[position] += 1
            return result
        } else {
            guard let result = incrementIndex(position - 1) else {
                kill = true
                return presentIndex
            }
            presentIndex[position] = span[position].startIndex
            return result
        }
    }
}

// MARK: - Dimensional Congruency

infix operator ≅ { precedence 130 }
public func ≅(lhs: Span, rhs: Span) -> Bool {
    if lhs.dimensions == rhs.dimensions {
        return true
    }

    let (max, min) = lhs.dimensions.count > rhs.dimensions.count ? (lhs, rhs) : (rhs, lhs)
    let diff = max.dimensions.count - min.dimensions.count
    return max.dimensions[0..<diff].reduce(1, combine: *) == 1 && Array(max.dimensions[diff..<max.dimensions.count]) == min.dimensions
}
