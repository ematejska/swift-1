# ShapedArraySlice

A contiguous slice of a `ShapedArray` or `ShapedArraySlice` instance.

``` swift
@frozen public struct ShapedArraySlice<Scalar>: _ShapedArrayProtocol
```

`ShapedArraySlice` enables fast, efficient operations on contiguous slices of `ShapedArray`
instances. `ShapedArraySlice` instances do not have their own storage. Instead, they provides a
view onto the storage of their base `ShapedArray`. `ShapedArraySlice` can represent two
different kinds of slices: element arrays and subarrays.

Element arrays are subdimensional elements of a `ShapedArray`: their rank is one less than that
of their base. Element array slices are obtained by indexing a `ShapedArray` instance with a
singular `Int32` index.

For example:

``` 
    var matrix = ShapedArray(shape: [2, 2], scalars: [0, 1, 2, 3])
    // `matrix` represents [[0, 1], [2, 3]].

    let element = matrix[0]
    // `element` is a `ShapedArraySlice` with shape [2]. It is an element
    // array, specifically the first element in `matrix`: [0, 1].

    matrix[1] = ShapedArraySlice(shape: [2], scalars: [4, 8])
    // The second element in `matrix` has been mutated.
    // `matrix` now represents [[0, 1, 4, 8]].
```

Subarrays are a contiguous range of the elements in a `ShapedArray`. The rank of a subarray is
the same as that of its base, but its leading dimension is the count of the slice range.
Subarray slices are obtained by indexing a `ShapedArray` with a `Range<Int32>` that represents a
range of elements (in the leading dimension). Methods like `prefix(:)` and `suffix(:)` that
internally index with a range also produce subarray.

For example:

``` 
    let zeros = ShapedArray(repeating: 0, shape: [3, 2])
    var matrix = ShapedArray(shape: [3, 2], scalars: Array(0..<6))
    // `zeros` represents [[0, 0], [0, 0], [0, 0]].
    // `matrix` represents [[0, 1], [2, 3], [4, 5]].

    let subarray = matrix.prefix(2)
    // `subarray` is a `ShapedArraySlice` with shape [2, 2]. It is a slice
    // of the first 2 elements in `matrix` and represents [[0, 1], [2, 3]].

    matrix[0..<2] = zeros.prefix(2)
    // The first 2 elements in `matrix` have been mutated.
    // `matrix` now represents [[0, 0], [0, 0], [4, 5]].
```

## Inheritance

`Codable`, `CustomPlaygroundDisplayConvertible`, `CustomReflectable`, `CustomStringConvertible`, `Equatable`, `ExpressibleByArrayLiteral`, `Hashable`, `MutableCollection`, `RandomAccessCollection`, [`_ShapedArrayProtocol`](/_ShapedArrayProtocol)

## Nested Type Aliases

### `Index`

``` swift
public typealias Index = Int
```

### `Element`

``` swift
public typealias Element = ShapedArraySlice
```

### `SubSequence`

``` swift
public typealias SubSequence = ShapedArraySlice
```

## Initializers

### `init(base:baseIndices:bounds:)`

Creates a `ShapedArraySlice` from a base `ShapedArray`, with the specified subdimensional
indices and subarray bounds.

``` swift
@inlinable internal init(base: __owned ShapedArray<Scalar>, baseIndices indices: __owned [Int] = [], bounds: Range<Int>? = nil)
```

### `init(shape:scalars:)`

Creates a `ShapedArraySlice` with the specified shape and contiguous scalars in row-major
order.

``` swift
public init(shape: __owned [Int], scalars: __owned [Scalar])
```

> Precondition: The number of scalars must equal the product of the dimensions of the shape.

### `init(shape:scalars:)`

Creates an `ShapedArraySlice` with the specified shape and sequence of scalars in row-major
order.

``` swift
public init<S: Sequence>(shape: __owned [Int], scalars: __shared S) where S.Element == Scalar
```

> Precondition: The number of scalars must equal the product of the dimensions of the shape.

### `init(_:)`

Creates a `ShapedArraySlice` from a scalar value.

``` swift
public init(_ scalar: __owned Scalar)
```

### `init(shape:repeating:)`

Creates a `ShapedArraySlice` with the specified shape and a single, repeated scalar value.

``` swift
@inlinable @available(*, deprecated, renamed: "init(repeating:shape:)") public init(shape: __owned [Int], repeating repeatedValue: __owned Scalar)
```

#### Parameters

  - repeatedValue: - repeatedValue: The scalar value to repeat.
  - shape: - shape: The shape of the `ShapedArraySlice`.

### `init(repeating:shape:)`

Creates a `ShapedArraySlice` with the specified shape and a single, repeated scalar value.

``` swift
public init(repeating repeatedValue: __owned Scalar, shape: __owned [Int])
```

#### Parameters

  - repeatedValue: - repeatedValue: The scalar value to repeat.
  - shape: - shape: The shape of the `ShapedArraySlice`.

## Properties

### `base`

The underlying `ShapedArray` of the slice.

``` swift
var base: ShapedArray<Scalar>
```

### `baseIndices`

The subdimensional indices of a slice.

``` swift
var baseIndices: [Int]
```

### `bounds`

The subarray bounds of a slice.

``` swift
var bounds: Range<Int>?
```

### `indexingDepth`

Indexing depth of this slice, i.e. the difference in rank between the base and the slice.

``` swift
var indexingDepth: Int
```

### `rank`

The number of dimensions of the array.

``` swift
var rank: Int
```

### `shape`

The shape of the array.

``` swift
var shape: [Int]
```

### `scalarCount`

The total number of scalars in the array.

``` swift
var scalarCount: Int
```

### `scalarRange`

The range of scalars from the base `ShapedArray` represented by a `ShapedArraySlice`.

``` swift
var scalarRange: Range<Int>
```

### `indices`

``` swift
var indices: Range<Int>
```

### `startIndex`

``` swift
var startIndex: Int
```

### `endIndex`

``` swift
var endIndex: Int
```

### `description`

A textual representation of this `ShapedArraySlice`.

``` swift
var description: String
```

> Note: use \`fullDescription\` for a non-pretty-printed representation showing all scalars.

### `playgroundDescription`

``` swift
var playgroundDescription: Any
```

### `customMirror`

``` swift
var customMirror: Mirror
```

## Methods

### `withUnsafeBufferPointer(_:)`

Calls a closure with a pointer to the `ShapedArraySlice`’s contiguous storage.

``` swift
public func withUnsafeBufferPointer<Result>(_ body: (UnsafeBufferPointer<Scalar>) throws -> Result) rethrows -> Result
```

#### Parameters

  - body: - body: A closure with an `UnsafeBufferPointer` parameter that points to the contiguous storage for the `ShapedArraySlice`. If no such storage exists, it is created. If body has a return value, that value is also used as the return value for the `withUnsafeBufferPointer(_:)` method. The pointer argument is valid only for the duration of the method's execution.

### `withUnsafeMutableBufferPointer(_:)`

Calls the given closure with a pointer to the `ShapedArraySlice`'s mutable contiguous
storage.

``` swift
public mutating func withUnsafeMutableBufferPointer<Result>(_ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> Result) rethrows -> Result
```

#### Parameters

  - body: - body: A closure with an `UnsafeMutableBufferPointer` parameter that points to the contiguous storage for the `ShapedArraySlice`. If no such storage exists, it is created. If body has a return value, that value is also used as the return value for the `withUnsafeMutableBufferPointer(_:)` method. The pointer argument is valid only for the duration of the method’s execution.
