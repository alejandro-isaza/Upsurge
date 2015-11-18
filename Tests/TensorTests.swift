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

class TensorTests: XCTestCase {
    var threeDimensional5IdentityTensor: Tensor<Real>!
    var fourDimensional2IndentityTensor: Tensor<Real>!
    
    override func setUp() {
        super.setUp()
        /*
        matrix 0:
        ⎛1, 0, 0, 0, 0⎞
        ⎜0, 0, 0, 0, 0⎟
        ⎜0, 0, 0, 0, 0⎟
        ⎜0, 0, 0, 0, 0⎟
        ⎝0, 0, 0, 0, 0⎠
        matrix 1:
        ⎛0, 0, 0, 0, 0⎞
        ⎜0, 1, 0, 0, 0⎟
        ⎜0, 0, 0, 0, 0⎟
        ⎜0, 0, 0, 0, 0⎟
        ⎝0, 0, 0, 0, 0⎠
        matrix 2:
        ⎛0, 0, 0, 0, 0⎞
        ⎜0, 0, 0, 0, 0⎟
        ⎜0, 0, 1, 0, 0⎟
        ⎜0, 0, 0, 0, 0⎟
        ⎝0, 0, 0, 0, 0⎠
        matrix 3:
        ⎛0, 0, 0, 0, 0⎞
        ⎜0, 0, 0, 0, 0⎟
        ⎜0, 0, 0, 0, 0⎟
        ⎜0, 0, 0, 1, 0⎟
        ⎝0, 0, 0, 0, 0⎠
        matrix 4:
        ⎛0, 0, 0, 0, 0⎞
        ⎜0, 0, 0, 0, 0⎟
        ⎜0, 0, 0, 0, 0⎟
        ⎜0, 0, 0, 0, 0⎟
        ⎝0, 0, 0, 0, 1⎠
        */
        threeDimensional5IdentityTensor = Tensor(dimensions: [5, 5, 5], repeatedValue: 0)
        threeDimensional5IdentityTensor[0, 0, 0] = 1
        threeDimensional5IdentityTensor[1, 1, 1] = 1
        threeDimensional5IdentityTensor[2, 2, 2] = 1
        threeDimensional5IdentityTensor[3, 3, 3] = 1
        threeDimensional5IdentityTensor[4, 4, 4] = 1

        /*
        Cube 1:
        in front:
        ⎛1, 0⎞
        ⎝0, 0⎠
        behind:
        ⎛0, 0⎞
        ⎝0, 0⎠
        
        Cube 2:
        in front:
        ⎛0, 0⎞
        ⎝0, 0⎠
        behind:
        ⎛0, 0⎞
        ⎝0, 1⎠
        */
        fourDimensional2IndentityTensor = Tensor(dimensions: [2, 2, 2, 2], repeatedValue: 0)
        fourDimensional2IndentityTensor[0, 0, 0, 0] = 1
        fourDimensional2IndentityTensor[1, 1, 1, 1] = 1
    }
    
    func testSliceAndSubscript() {
        let spanIndex: Span = [3, 2...3, 2...3]
        let slice1 = threeDimensional5IdentityTensor[spanIndex]
        
        let intervalIndex: [Interval] = [1, 1, .All, .All]
        let slice2 = fourDimensional2IndentityTensor[intervalIndex]
        
        let slice3 = fourDimensional2IndentityTensor.extractMatrix(1, 1, 0...1, 0...1)

        XCTAssertEqual(slice1, slice2)
        XCTAssert(slice1 == slice3)
        XCTAssert(slice3 == slice2)
        XCTAssertEqual(slice1[0, 1, 1], 1)
    }
    
    func testSliceAndValueAssignment() {
        threeDimensional5IdentityTensor[0, 1, 1] = 16
        let expected = Tensor<Real>(dimensions: [5, 5, 5], repeatedValue: 0)
        expected[0, 0, 0] = 1
        expected[1, 1, 1] = 1
        expected[2, 2, 2] = 1
        expected[3, 3, 3] = 1
        expected[4, 4, 4] = 1
        expected[0, 1, 1] = 16
        XCTAssertEqual(threeDimensional5IdentityTensor, expected)
        XCTAssertEqual(expected[0, 1, 1], 16)
        XCTAssertEqual(expected[0, 0, 1], 0)
    }
    
    func testSliceValueAssignment() {
        let fourDimensionalTensor = Tensor(dimensions: [2, 2, 2, 2], elements: [6.4, 2.4, 8.6, 0.2, 6.4, 1.5, 7.3, 1.1, 6.0, 1.4, 7.8, 9.2, 4.2, 6.1, 8.7, 3.6])
        fourDimensional2IndentityTensor[1, .All, 0...1, 0...1] = fourDimensionalTensor[0, 0...1, .All, 0...1]
        let expected = Tensor(dimensions: [2, 2, 2, 2], elements: [1, 0, 0, 0, 0, 0, 0, 0, 6.4, 2.4, 8.6, 0.2, 6.4, 1.5, 7.3, 1.1])
        XCTAssertEqual(fourDimensional2IndentityTensor, expected)
    }
    
    func testMatrixExtraction() {
        var m = fourDimensional2IndentityTensor.extractMatrix(1, 1, 0...1, 0...1)
        var expected = RealMatrix([[0, 0], [0, 1]])
        XCTAssertEqual(m, expected)
        
        m = fourDimensional2IndentityTensor.extractMatrix(0, 0, 0, 0)
        expected = RealMatrix([[1]])
        XCTAssertEqual(m, expected)
    }
}
