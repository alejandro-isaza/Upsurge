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

import Foundation
import Upsurge
import XCTest

class RealMatrixTests: XCTestCase {
    func testAdd() {
        var a = Matrix<Double>(rows: 2, columns: 2, elements: [1, 2, 3, 4] as ValueArray<Double>)
        let b = Matrix<Double>(rows: 2, columns: 2, elements: [2, 3, 4, 1] as ValueArray<Double>)
        let c = a + b
        a += b

        let d = Matrix<Double>(rows: 2, columns: 2, elements: [3, 5, 7, 5] as ValueArray<Double>)

        XCTAssertEqual(a, c)
        XCTAssertEqual(a, d)
    }

    func testSub() {
        var a = Matrix<Double>(rows: 2, columns: 2, elements: [1, 2, 3, 4] as ValueArray<Double>)
        let b = Matrix<Double>(rows: 2, columns: 2, elements: [2, 3, 4, 1] as ValueArray<Double>)
        let c = a - b
        a -= b

        let d = Matrix<Double>(rows: 2, columns: 2, elements: [-1, -1, -1, 3] as ValueArray<Double>)

        XCTAssertEqual(a, c)
        XCTAssertEqual(a, d)
    }

    func testMult() {
        var a = Matrix<Double>(rows: 2, columns: 2, elements: [1, 2, 3, 4] as ValueArray<Double>)
        let b = Matrix<Double>(rows: 2, columns: 2, elements: [2, 3, 4, 5] as ValueArray<Double>)
        let c = a * b

        let d = Matrix<Double>(rows: 2, columns: 2, elements: [10, 13, 22, 29] as ValueArray<Double>)

        XCTAssertEqual(c, d)

        a *= b
        XCTAssertEqual(a, d)
    }

    func testMultiplyWithColumn() {
        let v = ValueArray<Double>([2, 3, 4, 5, 6] as ValueArray<Double>)
        let m = Matrix<Double>([[1, 2, 3, 4, 5]])
        let c = v.toColumnMatrix() * m

        let d = Matrix<Double>([[2, 4, 6, 8, 10], [3, 6, 9, 12, 15], [4, 8, 12, 16, 20], [5, 10, 15, 20, 25], [6, 12, 18, 24, 30]])

        XCTAssertEqual(c, d)
    }

    func testMultiplyWithRow() {
        let m = Matrix<Double>([[1], [2], [3], [4], [5]])
        let v = ValueArray<Double>([2, 3, 4, 5, 6] as ValueArray<Double>)
        let c = v.toRowMatrix() * m

        let d = Matrix<Double>([[70]])

        XCTAssertEqual(c, d)
    }

    func testPostMultiplyWithColumn() {
        let m = Matrix<Double>([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        let v = ValueArray<Double>([2, 3, 4, 5, 6] as ValueArray<Double>)
        let c = m * v.toColumnMatrix()

        let d = Matrix<Double>([[70], [170]])

        XCTAssertEqual(c, d)
    }

    func testPostMultiplyWithRow() {
        let m = Matrix<Double>([[1], [2], [3], [4], [5]])
        let v = ValueArray<Double>([2, 3, 4, 5, 6] as ValueArray<Double>)
        let c = m * v.toRowMatrix()

        let d = Matrix<Double>([[2, 3, 4, 5, 6], [4, 6, 8, 10, 12], [6, 9, 12, 15, 18], [8, 12, 16, 20, 24], [10, 15, 20, 25, 30]])

        XCTAssertEqual(c, d)
    }

    func testTranspose() {
        let a = Matrix<Double>(rows: 2, columns: 2, elements: [1, 2, 3, 4] as ValueArray<Double>)
        let b = transpose(a)
        let c = transpose(b)

        XCTAssertEqual(a, c)
    }

    func testNormalization() {
        let a = Matrix<Double>([[3, 5, 8], [4, 12, 15]])
        let b = normalize(a)
        let c = Matrix<Double>([[0.6, 0.8, 0.384615384615385], [0.923076923076923, 0.470588235294118, 0.882352941176471]])

        XCTAssertEqualWithAccuracy(b.elements[0], c.elements[0], accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(b.elements[1], c.elements[1], accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(b.elements[2], c.elements[2], accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(b.elements[3], c.elements[3], accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(b.elements[4], c.elements[4], accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(b.elements[5], c.elements[5], accuracy: 0.00001)
    }

    func testInvert() {
        let a = Matrix<Double>(rows: 2, columns: 2, elements: [2, 6, -2, 4] as ValueArray<Double>)
        let b = inv(a)

        let c = Matrix<Double>(rows: 2, columns: 2, elements: [0.2, -0.3, 0.1, 0.1] as ValueArray<Double>)

        XCTAssertEqualWithAccuracy(c.elements[0], b.elements[0], accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(c.elements[1], b.elements[1], accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(c.elements[2], b.elements[2], accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(c.elements[3], b.elements[3], accuracy: 0.00001)
    }

    func testSwap() {
        var a = Matrix<Double>([[1, 2, 3], [4, 5, 6]])
        var b = Matrix<Double>([[4, 3], [2, 1]])
        swap(&a, &b)

        XCTAssertEqual(a.columns, 2)
        XCTAssertEqual(a[0, 0], 4)
        XCTAssertEqual(b.columns, 3)
        XCTAssertEqual(b[0, 0], 1)
    }

    func testAddColumnRow() {
        let m = Matrix<Double>([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        let r = m.column(1) + m.row(1)

        XCTAssertEqual(r.count, 3)
        XCTAssertEqual(r[0], 6)
        XCTAssertEqual(r[1], 10)
        XCTAssertEqual(r[2], 14)
    }

    func testAddToColumn() {
        let m = Matrix<Double>([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
            ])
        var col = m.column(1)
        col += 1

        XCTAssertEqual(m[0, 1], 3)
        XCTAssertEqual(m[1, 1], 6)
        XCTAssertEqual(m[2, 1], 9)
    }
}
