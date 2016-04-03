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

import Accelerate

// MARK: - Double

public func +=<ML: MutableLinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(inout lhs: ML, rhs: MR) {
    assert(lhs.count >= rhs.count)
    let count = min(lhs.count, rhs.count)
    withPointers(rhs, &lhs) { rhsp, lhsp in
        vDSP_vaddD(lhsp + lhs.startIndex, lhs.step, rhsp + rhs.startIndex, rhs.step, lhsp, lhs.step, vDSP_Length(count))
    }
}

public func +<ML: LinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(lhs: ML, rhs: MR) -> ValueArray<Double> {
    let count = min(lhs.count, rhs.count)
    let results = ValueArray<Double>(count: count)
    withPointers(lhs, rhs) { lhsp, rhsp in
        vDSP_vaddD(lhsp + lhs.startIndex, lhs.step, rhsp + rhs.startIndex, rhs.step, results.mutablePointer, results.step, vDSP_Length(count))
    }
    return results
}

public func +=<ML: MutableLinearType where ML.Element == Double>(inout lhs: ML, rhs: Double) {
    withPointer(&lhs) { lhsp in
        var scalar = rhs
        vDSP_vsaddD(lhsp + lhs.startIndex, lhs.step, &scalar, lhsp + lhs.startIndex, lhs.step, vDSP_Length(lhs.count))
    }
}

public func +<ML: LinearType where ML.Element == Double>(lhs: ML, rhs: Double) -> ValueArray<Double> {
    var rhs = rhs
    let results = ValueArray<Double>(count: lhs.count)
    withPointer(lhs) { lhsp in
        vDSP_vsaddD(lhsp + lhs.startIndex, lhs.step, &rhs, results.mutablePointer + results.startIndex, results.step, vDSP_Length(lhs.count))
    }
    return results
}

public func +<ML: LinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(lhs: Double, rhs: MR) -> ValueArray<Double> {
    return rhs + lhs
}

public func -=<ML: MutableLinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(inout lhs: ML, rhs: MR) {
    let count = min(lhs.count, rhs.count)
    withPointers(rhs, &lhs) { rhsp, lhsp in
    vDSP_vsubD(rhsp + rhs.startIndex, rhs.step, lhsp + lhs.startIndex, lhs.step, lhsp + lhs.startIndex, lhs.step, vDSP_Length(count))
    }
}

public func -<ML: LinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(lhs: ML, rhs: MR) -> ValueArray<Double> {
    let count = min(lhs.count, rhs.count)
    let results = ValueArray<Double>(count: count)
    withPointers(lhs, rhs) { lhsp, rhsp in
        vDSP_vsubD(rhsp + rhs.startIndex, rhs.step, lhsp + lhs.startIndex, lhs.step, results.mutablePointer + results.startIndex, results.step, vDSP_Length(count))
    }
    return results
}

public func -=<ML: MutableLinearType where ML.Element == Double>(inout lhs: ML, rhs: Double) {
    var scalar: Double = -rhs
    withPointer(&lhs) { lhsp in
        vDSP_vsaddD(lhsp + lhs.startIndex, lhs.step, &scalar, lhsp + lhs.startIndex, lhs.step, vDSP_Length(lhs.count))
    }
}

public func -<ML: LinearType where ML.Element == Double>(lhs: ML, rhs: Double) -> ValueArray<Double> {
    let results = ValueArray<Double>(count: lhs.count)
    var scalar: Double = -rhs
    withPointer(lhs) { lhsp in
        vDSP_vsaddD(lhsp + lhs.startIndex, lhs.step, &scalar, results.mutablePointer + results.startIndex, results.step, vDSP_Length(lhs.count))
    }
    return results
}

public func -<ML: LinearType where ML.Element == Double>(lhs: Double, rhs: ML) -> ValueArray<Double> {
    let results = ValueArray<Double>(count: rhs.count)
    withPointer(rhs) { rhsp in
        var scalarm: Double = -1
        var scalara: Double = 0
        vDSP_vsmsaD(rhsp + rhs.startIndex, rhs.step, &scalarm, &scalara, results.mutablePointer + results.startIndex, results.step, vDSP_Length(rhs.count))
    }
    return results
}

public func /=<ML: MutableLinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(inout lhs: ML, rhs: MR) {
    let count = min(lhs.count, rhs.count)
    withPointers(rhs, &lhs) { rhsp, lhsp in
        vDSP_vdivD(rhsp + rhs.startIndex, rhs.step, lhsp + lhs.startIndex, lhs.step, lhsp + lhs.startIndex, lhs.step, vDSP_Length(count))
    }
}

public func /<ML: LinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(lhs: ML, rhs: MR) -> ValueArray<Double> {
    let count = min(lhs.count, rhs.count)
    let results = ValueArray<Double>(count: lhs.count)
    withPointers(lhs, rhs) { lhsp, rhsp in
        vDSP_vdivD(rhsp + rhs.startIndex, rhs.step, lhsp + lhs.startIndex, lhs.step, results.mutablePointer + results.startIndex, results.step, vDSP_Length(count))
    }
    return results
}

public func /=<ML: MutableLinearType where ML.Element == Double>(inout lhs: ML, rhs: Double) {
    withPointer(&lhs) { lhsp in
        var scalar = rhs
        vDSP_vsdivD(lhsp + lhs.startIndex, lhs.step, &scalar, lhsp + lhs.startIndex, lhs.step, vDSP_Length(lhs.count))
    }
}

public func /<ML: LinearType where ML.Element == Double>(lhs: ML, rhs: Double) -> ValueArray<Double> {
    let results = ValueArray<Double>(count: lhs.count)
    withPointer(lhs) { lhsp in
        var scalar = rhs
        vDSP_vsdivD(lhsp + lhs.startIndex, lhs.step, &scalar, results.mutablePointer + results.startIndex, results.step, vDSP_Length(lhs.count))
    }
    return results
}

public func /<ML: LinearType where ML.Element == Double>(lhs: Double, rhs: ML) -> ValueArray<Double> {
    var lhs = lhs
    let results = ValueArray<Double>(count: rhs.count)
    withPointer(rhs) { rhsp in
        vDSP_svdivD(&lhs, rhsp + rhs.startIndex, rhs.step, results.mutablePointer + results.startIndex, results.step, vDSP_Length(rhs.count))
    }
    return results
}

public func *=<ML: MutableLinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(inout lhs: ML, rhs: MR) {
    withPointers(rhs, &lhs) { rhsp, lhsp in
        vDSP_vmulD(lhsp + lhs.startIndex, lhs.step, rhsp + rhs.startIndex, rhs.step, lhsp + lhs.startIndex, lhs.step, vDSP_Length(lhs.count))
    }
}

public func *<ML: LinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(lhs: ML, rhs: MR) -> ValueArray<Double> {
    let results = ValueArray<Double>(count: lhs.count)
    withPointers(lhs, rhs) { lhsp, rhsp in
        vDSP_vmulD(lhsp + lhs.startIndex, lhs.step, rhsp + rhs.startIndex, rhs.step, results.mutablePointer + results.startIndex, results.step, vDSP_Length(lhs.count))
    }
    return results
}

public func *=<ML: MutableLinearType where ML.Element == Double>(inout lhs: ML, rhs: Double) {
    var rhs = rhs
    withPointer(&lhs) { lhsp in
        vDSP_vsmulD(lhsp + lhs.startIndex, lhs.step, &rhs, lhsp + lhs.startIndex, lhs.step, vDSP_Length(lhs.count))
    }
}

public func *<ML: LinearType where ML.Element == Double>(lhs: ML, rhs: Double) -> ValueArray<Double> {
    var rhs = rhs
    let results = ValueArray<Double>(count: lhs.count)
    withPointer(lhs) { lhsp in
        vDSP_vsmulD(lhsp + lhs.startIndex, lhs.step, &rhs, results.mutablePointer + results.startIndex, results.step, vDSP_Length(lhs.count))
    }
    return results
}

public func *<ML: LinearType where ML.Element == Double>(lhs: Double, rhs: ML) -> ValueArray<Double> {
    return rhs * lhs
}

public func %<ML: LinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(lhs: ML, rhs: MR) -> ValueArray<Double> {
    return mod(lhs, rhs)
}

public func %<ML: LinearType where ML.Element == Double>(lhs: ML, rhs: Double) -> ValueArray<Double> {
    return mod(lhs, ValueArray<Double>(count: lhs.count, repeatedValue: rhs))
}

infix operator • {}
public func •<ML: LinearType, MR: LinearType where ML.Element == Double, MR.Element == Double>(lhs: ML, rhs: MR) -> Double {
    return dot(lhs, rhs)
}


// MARK: - Float

public func +=<ML: MutableLinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(inout lhs: ML, rhs: MR) {
    assert(lhs.count >= rhs.count)
    let count = min(lhs.count, rhs.count)
    withPointers(rhs, &lhs) { rhsp, lhsp in
        vDSP_vadd(lhsp + lhs.startIndex, lhs.step, rhsp + rhs.startIndex, rhs.step, lhsp, lhs.step, vDSP_Length(count))
    }
}

public func +<ML: LinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(lhs: ML, rhs: MR) -> ValueArray<Float> {
    let count = min(lhs.count, rhs.count)
    let results = ValueArray<Float>(count: count)
    withPointers(lhs, rhs) { lhsp, rhsp in
        vDSP_vadd(lhsp + lhs.startIndex, lhs.step, rhsp + rhs.startIndex, rhs.step, results.mutablePointer, results.step, vDSP_Length(count))
    }
    return results
}

public func +=<ML: MutableLinearType where ML.Element == Float>(inout lhs: ML, rhs: Float) {
    withPointer(&lhs) { lhsp in
        var scalar = rhs
        vDSP_vsadd(lhsp + lhs.startIndex, lhs.step, &scalar, lhsp + lhs.startIndex, lhs.step, vDSP_Length(lhs.count))
    }
}

public func +<ML: LinearType where ML.Element == Float>(lhs: ML, rhs: Float) -> ValueArray<Float> {
    let results = ValueArray<Float>(count: lhs.count)
    withPointer(lhs) { lhsp in
        var scalar = rhs
        vDSP_vsadd(lhsp + lhs.startIndex, lhs.step, &scalar, results.mutablePointer + results.startIndex, results.step, vDSP_Length(lhs.count))
    }
    return results
}

public func +<ML: LinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(lhs: Float, rhs: MR) -> ValueArray<Float> {
    return rhs + lhs
}

public func -=<ML: MutableLinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(inout lhs: ML, rhs: MR) {
    let count = min(lhs.count, rhs.count)
    withPointers(rhs, &lhs) { rhsp, lhsp in
        vDSP_vsub(rhsp + rhs.startIndex, rhs.step, lhsp + lhs.startIndex, lhs.step, lhsp + lhs.startIndex, lhs.step, vDSP_Length(count))
    }
}

public func -<ML: LinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(lhs: ML, rhs: MR) -> ValueArray<Float> {
    let count = min(lhs.count, rhs.count)
    let results = ValueArray<Float>(count: count)
    withPointers(lhs, rhs) { lhsp, rhsp in
        vDSP_vsub(rhsp + rhs.startIndex, rhs.step, lhsp + lhs.startIndex, lhs.step, results.mutablePointer + results.startIndex, results.step, vDSP_Length(count))
    }
    return results
}

public func -=<ML: MutableLinearType where ML.Element == Float>(inout lhs: ML, rhs: Float) {
    withPointer(&lhs) { lhsp in
        var scalar: Float = -rhs
        vDSP_vsadd(lhsp + lhs.startIndex, lhs.step, &scalar, lhsp + lhs.startIndex, lhs.step, vDSP_Length(lhs.count))
    }
}

public func -<ML: LinearType where ML.Element == Float>(lhs: ML, rhs: Float) -> ValueArray<Float> {
    let results = ValueArray<Float>(count: lhs.count)
    withPointer(lhs) { lhsp in
        var scalar: Float = -rhs
        vDSP_vsadd(lhsp + lhs.startIndex, lhs.step, &scalar, results.mutablePointer + results.startIndex, results.step, vDSP_Length(lhs.count))
    }
    return results
}

public func -<ML: LinearType where ML.Element == Float>(lhs: Float, rhs: ML) -> ValueArray<Float> {
    let results = ValueArray<Float>(count: rhs.count)
    withPointer(rhs) { rhsp in
        var scalarm: Float = -1
        var scalara: Float = 0
        vDSP_vsmsa(rhsp + rhs.startIndex, rhs.step, &scalarm, &scalara, results.mutablePointer + results.startIndex, results.step, vDSP_Length(rhs.count))
    }
    return results
}

public func /=<ML: MutableLinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(inout lhs: ML, rhs: MR) {
    let count = min(lhs.count, rhs.count)
    withPointers(rhs, &lhs) { rhsp, lhsp in
        vDSP_vdiv(rhsp + rhs.startIndex, rhs.step, lhsp + lhs.startIndex, lhs.step, lhsp + lhs.startIndex, lhs.step, vDSP_Length(count))
    }
}

public func /<ML: LinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(lhs: ML, rhs: MR) -> ValueArray<Float> {
    let count = min(lhs.count, rhs.count)
    let results = ValueArray<Float>(count: lhs.count)
    withPointers(lhs, rhs) { lhsp, rhsp in
        vDSP_vdiv(rhsp + rhs.startIndex, rhs.step, lhsp + lhs.startIndex, lhs.step, results.mutablePointer + results.startIndex, results.step, vDSP_Length(count))
    }
    return results
}

public func /=<ML: MutableLinearType where ML.Element == Float>(inout lhs: ML, rhs: Float) {
    withPointer(&lhs) { lhsp in
        var scalar = rhs
        vDSP_vsdiv(lhsp + lhs.startIndex, lhs.step, &scalar, lhsp + lhs.startIndex, lhs.step, vDSP_Length(lhs.count))
    }
}

public func /<ML: LinearType where ML.Element == Float>(lhs: ML, rhs: Float) -> ValueArray<Float> {
    let results = ValueArray<Float>(count: lhs.count)
    withPointer(lhs) { lhsp in
        var scalar = rhs
        vDSP_vsdiv(lhsp + lhs.startIndex, lhs.step, &scalar, results.mutablePointer + results.startIndex, results.step, vDSP_Length(lhs.count))
    }
    return results
}

public func /<ML: LinearType where ML.Element == Float>(lhs: Float, rhs: ML) -> ValueArray<Float> {
    let results = ValueArray<Float>(count: rhs.count)
    withPointer(rhs) { rhsp in
        var scalar = lhs
        vDSP_svdiv(&scalar, rhsp + rhs.startIndex, rhs.step, results.mutablePointer + results.startIndex, results.step, vDSP_Length(rhs.count))
    }
    return results
}

public func *=<ML: MutableLinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(inout lhs: ML, rhs: MR) {
    withPointers(rhs, &lhs) { rhsp, lhsp in
        vDSP_vmul(lhsp + lhs.startIndex, lhs.step, rhsp + rhs.startIndex, rhs.step, lhsp + lhs.startIndex, lhs.step, vDSP_Length(lhs.count))
        }
}

public func *<ML: LinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(lhs: ML, rhs: MR) -> ValueArray<Float> {
    let results = ValueArray<Float>(count: lhs.count)
    withPointers(lhs, rhs) { lhsp, rhsp in
        vDSP_vmul(lhsp + lhs.startIndex, lhs.step, rhsp + rhs.startIndex, rhs.step, results.mutablePointer + results.startIndex, results.step, vDSP_Length(lhs.count))
    }
    return results
}

public func *=<ML: MutableLinearType where ML.Element == Float>(inout lhs: ML, rhs: Float) {
    withPointer(&lhs) { lhsp in
        var scalar = rhs
        vDSP_vsmul(lhsp + lhs.startIndex, lhs.step, &scalar, lhsp + lhs.startIndex, lhs.step, vDSP_Length(lhs.count))
    }
}

public func *<ML: LinearType where ML.Element == Float>(lhs: ML, rhs: Float) -> ValueArray<Float> {
    let results = ValueArray<Float>(count: lhs.count)
    withPointer(lhs) { lhsp in
        var scalar = rhs
        vDSP_vsmul(lhsp + lhs.startIndex, lhs.step, &scalar, results.mutablePointer + results.startIndex, results.step, vDSP_Length(lhs.count))
    }
    return results
}

public func *<ML: LinearType where ML.Element == Float>(lhs: Float, rhs: ML) -> ValueArray<Float> {
    return rhs * lhs
}

public func %<ML: LinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(lhs: ML, rhs: MR) -> ValueArray<Float> {
    return mod(lhs, rhs)
}

public func %<ML: LinearType where ML.Element == Float>(lhs: ML, rhs: Float) -> ValueArray<Float> {
    return mod(lhs, ValueArray<Float>(count: lhs.count, repeatedValue: rhs))
}

public func •<ML: LinearType, MR: LinearType where ML.Element == Float, MR.Element == Float>(lhs: ML, rhs: MR) -> Float {
    return dot(lhs, rhs)
}
