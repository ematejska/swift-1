# ShapedArray

`ShapedArray` is a multi-dimensional array. It has a shape, which has type `[Int]` and defines
the array dimensions, and uses a `TensorBuffer` internally as storage.

``` swift
@frozen public struct ShapedArray<Scalar>: _ShapedArrayProtocol
```

## Inheritance

`Codable`, [`ConvertibleFromNumpyArray`](/ConvertibleFromNumpyArray), `CustomPlaygroundDisplayConvertible`, `CustomReflectable`, `CustomStringConvertible`, `Equatable`, `ExpressibleByArrayLiteral`, `Hashable`, `MutableCollection`, `RandomAccessCollection`, [`_ShapedArrayProtocol`](/_ShapedArrayProtocol)

## Nested Type Aliases

### `Index`

``` swift
public typealias Index = Int
```

### `Element`

``` swift
public typealias Element = ShapedArraySlice<Scalar>
```

### `SubSequence`

``` swift
public typealias SubSequence = ShapedArraySlice<Scalar>
```

## Initializers

### `init(buffer:shape:)`

Creates a `ShapedArray` from a `TensorBuffer` and a shape.

``` swift
internal init(buffer: __owned TensorBuffer<Scalar>, shape: __owned [Int])
```

### `init(_:)`

Creates a `ShapedArray` with the same shape and scalars as the specified instance.

``` swift
public init(_ other: ShapedArray)
```

### `init(shape:scalars:)`

Creates a `ShapedArray` with the specified shape and contiguous scalars in row-major order.

``` swift
public init(shape: __owned [Int], scalars: __owned [Scalar])
```

> Precondition: The number of scalars must equal the product of the dimensions of the shape.

### `init(shape:scalars:)`

Creates a `ShapedArray` with the specified shape and sequence of scalars in row-major order.

``` swift
public init<S: Sequence>(shape: __owned [Int], scalars: __shared S) where S.Element == Scalar
```

> Precondition: The number of scalars must equal the product of the dimensions of the shape.

### `init(_:)`

Creates a `ShapedArray` from a scalar value.

``` swift
public init(_ scalar: __owned Scalar)
```

### `init(shape:repeating:)`

Creates a `ShapedArray` with the specified shape and a single, repeated scalar value.

``` swift
@inlinable @available(*, deprecated, renamed: "init(repeating:shape:)") public init(shape: __owned [Int], repeating repeatedValue: __owned Scalar)
```

#### Parameters

  - shape: - shape: The shape of the `ShapedArray`.
  - repeatedValue: - repeatedValue: The scalar value to repeat.

### `init(repeating:shape:)`

Creates a `ShapedArray` with the specified shape and a single, repeated scalar value.

``` swift
public init(repeating repeatedValue: __owned Scalar, shape: __owned [Int])
```

#### Parameters

  - repeatedValue: - repeatedValue: The scalar value to repeat.
  - shape: - shape: The shape of the `ShapedArray`.

## Properties

### `buffer`

Contiguous memory storing scalars.

``` swift
var buffer: TensorBuffer<Scalar>
```

### `shape`

The dimensions of the array.

``` swift
var shape: [Int]
```

### `rank`

The number of dimensions of the array.

``` swift
var rank: Int
```

### `scalarCount`

The total number of scalars in the array.

``` swift
var scalarCount: Int
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

A textual representation of this `ShapedArray`.

``` swift
var description: String
```

> Note: use \`fullDescription\` for a non-pretty-printed description showing all scalars.

### `playgroundDescription`

``` swift
var playgroundDescription: Any
```

### `customMirror`

``` swift
var customMirror: Mirror
```

## Methods

### `ensureUniquelyReferenced()`

``` swift
fileprivate mutating func ensureUniquelyReferenced()
```

### `withUnsafeBufferPointer(_:)`

Calls a closure with a pointer to the array’s contiguous storage.

``` swift
public func withUnsafeBufferPointer<Result>(_ body: (UnsafeBufferPointer<Scalar>) throws -> Result) rethrows -> Result
```

#### Parameters

  - body: - body: A closure with an `UnsafeBufferPointer` parameter that points to the contiguous storage for the array. If no such storage exists, it is created. If body has a return value, that value is also used as the return value for the `withUnsafeBufferPointer(_:)` method. The pointer argument is valid only for the duration of the method's execution.

### `withUnsafeMutableBufferPointer(_:)`

Calls the given closure with a pointer to the array’s mutable contiguous storage.

``` swift
public mutating func withUnsafeMutableBufferPointer<Result>(_ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> Result) rethrows -> Result
```

#### Parameters

  - body: - body: A closure with an `UnsafeMutableBufferPointer` parameter that points to the contiguous storage for the array. If no such storage exists, it is created. If body has a return value, that value is also used as the return value for the `withUnsafeMutableBufferPointer(_:)` method. The pointer argument is valid only for the duration of the method's execution.
