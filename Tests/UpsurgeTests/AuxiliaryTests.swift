// Copyright (c) 2014–2015 Mattt Thompson (http://mattt.me)
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

class AuxiliaryTests: XCTestCase {
    let n = 10000

    func testCopysign() {
        let signs = ValueArray<Double>((0..<n).map({ $0 % 2 == 0 ? 1.0 : -1.0 }))

        let magnitudes = ValueArray<Double>(count: n)
        for i in 0..<n {
            magnitudes[i] = Double(arc4random_uniform(10))
        }

        let expected = ValueArray<Double>(count: n)
        for i in 0..<signs.count {
            expected[i] = signs[i] * abs(magnitudes[i])
        }

        var actual: ValueArray<Double> = []
        self.measure {
            actual = copysign(signs, magnitude: magnitudes)
        }

        XCTAssertEqual(actual, expected)
    }

    func testThreshold() {
        let signs = (0..<n).map {$0 % 2 == 0 ? 1.0 : -1.0}
        let clip = threshold(signs, low: 0.0)

        XCTAssertEqual(min(clip), 0.0)
    }

    func testRound() {
        let n = 100
        let increment = (0..<n).map { Double($0) * 0.2 }
        let rounded = round(increment)

        let expected = ValueArray<Double>(increment.map { round($0) })
        XCTAssertEqual(rounded, expected)
    }

}
