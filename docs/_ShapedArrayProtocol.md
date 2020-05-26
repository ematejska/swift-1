# \_ShapedArrayProtocol

``` swift
public protocol _ShapedArrayProtocol: RandomAccessCollection, MutableCollection
```

## Inheritance

`MutableCollection`, `RandomAccessCollection`

## Requirements

## Scalar

``` swift
associatedtype Scalar
```

## rank

The number of dimensions of the array.

``` swift
var rank: Int
```

## shape

The shape of the array.

``` swift
var shape: [Int]
```

## scalarCount

The total number of scalars in the array.

``` swift
var scalarCount: Int
```

## init(shape:scalars:)

Creates an array with the specified shape and contiguous scalars in row-major order.

``` swift
init(shape: [Int], scalars: [Scalar])
```

> Precondition: The number of scalars must equal the product of the dimensions of the shape.

## init(shape:scalars:)

Creates an array with the specified shape and sequence of scalars in row-major order.

``` swift
init<S: Sequence>(shape: [Int], scalars: S) where S.Element == Scalar
```

> Precondition: The number of scalars must equal the product of the dimensions of the shape.

## withUnsafeBufferPointer(\_:)

Calls a closure with a pointer to the array’s contiguous storage.

``` swift
func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Scalar>) throws -> R) rethrows -> R
```

### Parameters

  - body: - body: A closure with an `UnsafeBufferPointer` parameter that points to the contiguous storage for the array. If no such storage exists, it is created. If body has a return value, that value is also used as the return value for the `withUnsafeBufferPointer(_:)` method. The pointer argument is valid only for the duration of the method's execution.

## withUnsafeMutableBufferPointer(\_:)

Calls the given closure with a pointer to the array’s mutable contiguous storage.

``` swift
mutating func withUnsafeMutableBufferPointer<R>(_ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> R) rethrows -> R
```

### Parameters

  - body: - body: A closure with an `UnsafeMutableBufferPointer` parameter that points to the contiguous storage for the array. If no such storage exists, it is created. If body has a return value, that value is also used as the return value for the `withUnsafeMutableBufferPointer(_:)` method. The pointer argument is valid only for the duration of the method's execution.
