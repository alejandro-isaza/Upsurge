# Upsurge

Upsurge is a math utilities library. It provides support for linear operations on vectors and matrices, and slicing of higher-dimensional tensors. It relies on [Accelerate](https://developer.apple.com/library/mac/documentation/Accelerate/Reference/AccelerateFWRef/index.html#//apple_ref/doc/uid/TP40009465), which is a framework that provides high-performance functions for matrix math, digital signal processing, and image manipulation by harnessing [SIMD](http://en.wikipedia.org/wiki/SIMD) instructions available in modern CPUs.

Upsurge is a fork of [Surge](https://github.com/mattt/Surge) which was abandoned for a while. Upsurge supports tensors and has better support for matrices and arrays. It provides a custom `ValueArray` class as an alternative to Swift's built-in `Array`. It being a `class` instead of a `struct` means that you can manage when and if it gets copied, making memory management more explicit. This also allows defining the `+=` operator to mean addition instead of concatenation.


## Features

- [x] Tensor and tensor slicing: `tensor.asMatrix(1, 1, 0...4, 0...4)`
- [x] Matrix and matrix operations: `let result = A * B′`
- [x] ValueArrays with explicit copying and numeric operators: `let result = A • B`
- [x] Accelerate functions: `let conv = convolution(signal: signal, kernel: kernel)`


## Installation

Upsurge supports both CocoaPods (`pod 'Upsurge'`) and Carthage (`github "aleph7/Upsurge"`). 


## Usage

### Arrays and vector operations

All of Upsurge's linear (1-dimensional) operations can be performed on anything that conforms to `LinearType`. Swift's built-in arrays and array slices conform to `LinearType`, of course. But Upsurge also defines the `ValueArray` class to store a one-dimensional collection of values. `ValueArray` is very similar to Swift's `Array` but it is optimized to reduce unnecessary memory allocation. These are the most important differences:
* Its instances have a fixed size defined on creation. When you create a `ValueArray` you can define a capacity `var a = ValueArray<Double>(capacity: 100)` and then append elements up to that capacity. **Or** you can create it with specific elements `var a: ValueArray = [1.0, 2.0, 3.0]` but then you can't add any more elements after.
* It is a class. That means that creating a new variable will only create a reference and modifying the reference will also modify the original. For instance doing `var a: ValueArray = [1, 2, 3]; var b = a` and then `b[0] = 5` will result in `a` being `[5, 2, 3]`. If you want to create a copy you need to do `var b = ValueArray(a)` or `var b = a.copy()`.
* You **can** create an uninitialized `ValueArray` by doing `var a = ValueArray<Double>(capacity: n)` or `var a = ValueArray<Doube>(count: n)`. This is good for when you are going to fill up the array yourself. But you can also use `var a = ValueArray(count: n, repeatedValue: 0.0)` if you do want to initialize all the values.

#### Creating arrays

Create a `ValueArray` with specific literal elements when you know ahead of time what the contents are, and you don't need to add more elements at a later time:
```swift
let a: ValueArray = [1.0, 3.0, 5.0, 7.0]
```

Create a `ValueArray` with a capacity and then fill it in when you are loading the contents from an external source or have a very large array:
```swift
let a = ValueArray<Double>(capacity: 100)
for v in intputSource {
    a.append(v)
}
```

Finally there is a way of initializing both the capacity and the count of a `ValueArray`. You should rarely need this but it's there for when you are doing operations on existing arrays using low-level APIs that take pointers:
```swift
func operation(a: ValueArray<Double>) {
    let N = a.count
    let b = ValueArray<Double>(count: N)
    // ...
}
```

#### Vector arithmetic

You can perform operations on `ValueArray` in an intuitive manner:
```swift
let a: ValueArray = [1.0, 3.0, 5.0, 7.0]
let b: ValueArray = [2.0, 4.0, 6.0, 8.0]
let addition = a + b // [3.0, 7.0, 11.0, 15.0]
let product  = a • b // 100.0

```


### Matrix operations

```swift
import Upsurge

let A = Matrix<Double>([
    [1,  1],
    [1, -1]
])
let C = Matrix<Double>([
    [3],
    [1]
])

// find B such that A*B=C
let B = inv(A) * C // [2.0, 1.0]′

// Verify result
let r = A*B - C    // zero   
```

### Tiling
A block  `Matrix`  can be formed by repeating a 1-D `ValueArray` or 2-D `Matrix`  **mxn** times.

```swift
import Upsurge

let a = ValueArray = [1.0, 2.0]
// Tile source array 2 times in each directon,
// returning a 2X4 block matrix
let A = a.tile(m: 2, n:2)

let B = Matrix<Double>([
    [1.0,  2.0],
    [3.0,  4.0]
)]
// Tile source matrix 2 times in each directon,
// returning a 4x4 block matrix
let r = B.tile(m: 2, n: 2)
```

### Tensors

The `Tensor` class makes it easy to manipulate multi-dimensional data. You can easily slice or flatten a tensor to get matrices and vectors that you can operate on.


---

## License

Upsurge is available under the MIT license. See the LICENSE file for more info.
