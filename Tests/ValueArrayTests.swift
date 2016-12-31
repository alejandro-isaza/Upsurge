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

import XCTest
import Upsurge

class RealArrayTests: XCTestCase {
    func testDescription() {
        let emptyRealArray: ValueArray<Double> = []
        XCTAssertEqual(emptyRealArray.description, "[]")

        let array: ValueArray<Double> = [1.0, 2.0, 3.0]
        XCTAssertEqual(array.description, "[1.0, 2.0, 3.0]")
    }

    func testCopy() {
        let a: ValueArray = [1.0, 2.0, 3.0]
        let b = a.copy()
        b[0] = 4
        XCTAssertEqual(a[0], 1.0)
        XCTAssertEqual(b[0], 4.0)
    }

    func testSwap() {
        var a: ValueArray<Double> = [1, 2, 3]
        var b: ValueArray<Double> = [4]
        swap(&a, &b)

        XCTAssertEqual(a.count, 1)
        XCTAssertEqual(b.count, 3)
        XCTAssertEqual(a[0], 4.0)
        XCTAssertEqual(b[0], 1.0)
    }

    func testAppendContentsOf() {
        let a = ValueArray<Double>(capacity: 3)
        a.append(contentsOf: [2, 3])
        XCTAssertEqual(a.count, 2)
        XCTAssertEqual(a[0], 2.0)
        XCTAssertEqual(a[1], 3.0)
    }

    func testAddSlice() {
        let a: ValueArray<Double> = [1.0, 2.0, 3.0, 4.0]
        let b = a[0...1] + a[2...3]

        XCTAssertEqual(b.count, 2)
        XCTAssertEqual(b[0], 4.0)
        XCTAssertEqual(b[1], 6.0)
    }

    func testAppend() {
        let n = ValueArray<Double>(capacity: 6)
        n.append(1.0)
        n.append(2.0)

        XCTAssertEqual(n.capacity, 6)
        XCTAssertEqual(n.count, 2)
        XCTAssertEqual(n[0], 1.0)
        XCTAssertEqual(n[1], 2.0)
    }

    func testSlice() {
        let array = ValueArray<Double>((0..<10).map({ Double($0) }))
        let slice = array[5...8]

        XCTAssertEqual(slice.count, 4)
        XCTAssertEqual(slice.startIndex, 5)
        XCTAssertEqual(slice.endIndex, 9)
        XCTAssert(slice == ValueArray<Double>([5.0, 6.0, 7.0, 8.0]))
    }

    func testAssignSlice() {
        let array = ValueArray<Double>((0..<10).map({ Double($0) }))
        let matrix = Matrix<Double>(rows: 5, columns: 10, elements: (0..<50).map({ Double($0) }))
        let expected = ValueArray(matrix.row(2))

        array[0..<10] = matrix.row(2)
        XCTAssertEqual(array, expected)
    }
}
