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

public protocol TensorType {
    associatedtype Element
    associatedtype Slice
    
    /// A description of the dimensions over which the TensorType spans
    var span: Span { get }
    
    subscript(intervals: [IntervalType]) -> Slice { get }
    subscript(intervals: [Int]) -> Element { get }

    /// Call `body(pointer)` with the buffer pointer to the beginning of the memory block
    func withUnsafeBufferPointer<R>(@noescape body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R

    /// Call `body(pointer)` with the pointer to the beginning of the memory block
    func withUnsafePointer<R>(@noescape body: (UnsafePointer<Element>) throws -> R) rethrows -> R
}

public extension TensorType {
    /// The size of each dimension
    public var dimensions: [Int] {
        return span.dimensions
    }
    
    /// The number of valid element in the memory block, taking into account the step size.
    public var count: Int {
        return span.count
    }
    
    /// The number of dimensions
    public var rank: Int {
        return span.rank
    }
    
    /// Convert a high-dimensional index into an integer index for a LinearType
    public func linearIndex(indices: [Int]) -> Int {
        precondition(indexIsValid(indices))
        var index = indices[0]
        for (i, dim) in span.dimensions[1..<rank].enumerate() {
            index = (dim * index) + indices[i+1]
        }
        return index
    }
    
    /// Check that an index falls within the span
    public func indexIsValid(indices: [Int]) -> Bool {
        if indices.count != rank {
            return false
        }
        
        for (i, index) in indices.enumerate() {
            if index < span[i].startIndex || span[i].endIndex <= index {
                return false
            }
        }
        return true
    }
}

public protocol MutableTensorType: TensorType {
    subscript(intervals: [IntervalType]) -> Slice { get set }
    subscript(intervals: [Int]) -> Element { get set }

    /// Call `body(pointer)` with the mutable buffer pointer to the beginning of the memory block
    mutating func withUnsafeMutableBufferPointer<R>(@noescape body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R

    /// Call `body(pointer)` with the mutable pointer to the beginning of the memory block
    mutating func withUnsafeMutablePointer<R>(@noescape body: (UnsafeMutablePointer<Element>) throws -> R) rethrows -> R
}


public extension MutableTensorType {
    /// Assign all values of a TensorType to this tensor.
    ///
    /// - precondition: The available space on `self` is greater than or equal to the number of elements on `lhs`
    mutating func assignFrom<T: TensorType where T.Element == Element>(rhs: T) {
        precondition(rhs.count <= count)
        withPointers(&self, rhs) { lhsp, rhsp in
            lhsp.assignFrom(UnsafeMutablePointer(rhsp), count: count)
        }
    }
}
