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


public struct TensorSlice<ElementType where ElementType: CustomStringConvertible, ElementType: Equatable> : Equatable {
    public typealias Index = [Int]
    public typealias Element = ElementType
    
    var base: Tensor<Element>
    let dimensions: [Int]
    
    public var slice: RangedIndex

    public var pointer: UnsafePointer<Element> {
        return base.pointer
    }
    
    public var mutablePointer: UnsafeMutablePointer<Element> {
        return base.mutablePointer
    }
    
    public init(base: Tensor<Element>, slice: RangedIndex) {
        assert(slice.dimensions.count == base.dimensions.count)
        self.base = base
        self.slice = slice
        self.dimensions = slice.dimensions
    }
    
    public subscript(indices: Int...) -> Element {
        get {
            let index = zip(slice.startIndex, indices).map{ $0 + $1 }
            assert(indexIsValid(index))
            return base[index]
        }
        set {
            let index = indices.enumerate().map{ $1 + dimensions[$0] }
            assert(indexIsValid(index))
            base[index] = newValue
        }
    }
    
    public subscript(indices: Index) -> Element {
        get {
            let index = indices.enumerate().map{ $1 + slice.startIndex[$0] }
            assert(indexIsValid(index))
            return base[index]
        }
        set {
            let index = indices.enumerate().map{ $1 + slice.startIndex[$0] }
            assert(indexIsValid(index))
            base[index] = newValue
        }
    }
    
    public subscript(slice: RangedIndex) -> TensorSlice<Element> {
        get {
            assert(slice.count == self.slice.count)
            assert(rangedIndexIsValid(slice))
            let baseSlice = RangedIndex(index: zip(slice.startIndex, slice.index).map{
                return $0 + $1.startIndex..<$0 + $1.endIndex
            })
            return TensorSlice(base: base, slice: baseSlice)
        }
        set {
            for index in slice  {
                let baseIndex = zip(self.slice.startIndex, index).map{ $0 + $1 }
                base[baseIndex] = newValue[index]
            }
        }
    }
    
    public subscript(slice: RangedDimension...) -> TensorSlice<Element> {
        get {
            assert(slice.count == self.slice.count)
            assert(rangedIndexIsValid(RangedIndex(index: slice)))
            let newSlice = RangedIndex(index: zip(self.slice.startIndex, slice).map{
                return $0 + $1.startIndex..<$0 + $1.endIndex
            })
            return TensorSlice(base: base, slice: newSlice)
        }
        set {
            for index in RangedIndex(index: slice)  {
                let baseIndex = zip(self.slice.startIndex, index).map{ $0 + $1 }
                base[baseIndex] = newValue[index]
            }
        }
    }
    
    public func indexIsValid(indices: [Int]) -> Bool {
        assert(indices.count == dimensions.count)
        for (i, index) in indices.enumerate() {
            if index < 0 && dimensions[i] <= index {
                return false
            }
        }
        return true
    }
    
    public func rangedIndexIsValid(indices: RangedIndex) -> Bool {
        assert(indices.dimensions.count == dimensions.count)
        for (i, range) in indices.enumerate() {
            if range.startIndex < 0 && dimensions[i] <= range.endIndex {
                return false
            }
        }
        return true
    }
}

// MARK: - Equatable

public func ==<T: Equatable>(lhs: TensorSlice<T>, rhs: TensorSlice<T>) -> Bool {
    assert(lhs.dimensions == rhs.dimensions)
    let slice = RangedIndex(index: lhs.dimensions.map{ 0..<$0 })
    for index in slice {
        if lhs[index] != rhs[index] {
            return false
        }
    }
    return true
}