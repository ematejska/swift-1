# TensorShape

A struct representing the shape of a tensor.

``` swift
@frozen public struct TensorShape: ExpressibleByArrayLiteral
```

`TensorShape` is a thin wrapper around an array of integers that represent shape dimensions. All
tensor types use `TensorShape` to represent their shape.

## Inheritance

`Codable`, `Collection`, `CustomStringConvertible`, `Equatable`, `ExpressibleByArrayLiteral`, `MutableCollection`, [`PythonConvertible`](/PythonConvertible), `RandomAccessCollection`, `RangeReplaceableCollection`

## Nested Type Aliases

### `Element`

``` swift
public typealias Element = Int
```

### `Index`

``` swift
public typealias Index = Int
```

### `Indices`

``` swift
public typealias Indices = Range<Int>
```

### `SubSequence`

``` swift
public typealias SubSequence = Self
```

## Initializers

### `init(_:)`

Initialize with an array of dimensions. The rank of the tensor is the length of the array.

``` swift
@inlinable public init(_ dimensions: [Int])
```

#### Parameters

  - dimensions: - dimensions: The shape dimensions.

### `init(_:)`

Initialize with a collection of dimensions. The rank of the tensor is the length of the
collection.

``` swift
@inlinable public init<C: Collection>(_ dimensions: C) where C.Element == Int
```

#### Parameters

  - dimensions: - dimensions: The shape dimensions.

### `init(arrayLiteral:)`

Initialize with an array literal representing the shape dimensions. The rank of the tensor
is the number of dimensions.

``` swift
@inlinable public init(arrayLiteral elements: Int)
```

#### Parameters

  - dimensions: - dimensions: The shape dimensions.

### `init(_:)`

Initialize with variadic elements representing the shape dimensions. The rank of the tensor
is the number of elements.

``` swift
@inlinable public init(_ elements: Int)
```

#### Parameters

  - dimensions: - dimensions: The shape dimensions.

### `init(repeating:count:)`

``` swift
@inlinable public init(repeating repeatedValue: Int, count: Int)
```

### `init()`

``` swift
@inlinable public init()
```

### `init(from:)`

``` swift
@inlinable public init(from decoder: Decoder) throws
```

## Properties

### `pythonObject`

<dl>
<dt><code>canImport(Python) || canImport(PythonKit)</code></dt>
<dd>

``` swift
var pythonObject: PythonObject
```

</dd>
</dl>

### `dimensions`

The dimensions of the shape.

``` swift
var dimensions: [Int]
```

### `rank`

The rank of the shape (i.e. the number of dimensions).

``` swift
var rank: Int
```

### `contiguousSize`

The size of the shape as a contiguously stored array.

``` swift
var contiguousSize: Int
```

### `count`

The rank of the shape (i.e. the number of dimensions).

``` swift
var count: Int
```

### `indices`

``` swift
var indices: Indices
```

### `startIndex`

``` swift
var startIndex: Index
```

### `endIndex`

``` swift
var endIndex: Index
```

### `description`

``` swift
var description: String
```

## Methods

### `index(after:)`

``` swift
@inlinable public func index(after i: Index) -> Index
```

### `index(_:offsetBy:)`

``` swift
@inlinable public func index(_ i: Int, offsetBy distance: Int) -> Int
```

### `distance(from:to:)`

``` swift
@inlinable public func distance(from start: Int, to end: Int) -> Int
```

### `append(_:)`

``` swift
@inlinable public mutating func append(_ newElement: Element)
```

### `append(contentsOf:)`

``` swift
@inlinable public mutating func append(contentsOf newElements: TensorShape)
```

### `append(contentsOf:)`

``` swift
@inlinable public mutating func append<S: Sequence>(contentsOf newElements: S) where Element == S.Element
```

### `replaceSubrange(_:with:)`

``` swift
@inlinable public mutating func replaceSubrange<C>(_ subrange: Range<Index>, with newElements: C) where C: Collection, Element == C.Element
```

### `==(lhs:rhs:)`

``` swift
@inlinable public static func ==(lhs: TensorShape, rhs: TensorShape) -> Bool
```

### `encode(to:)`

``` swift
@inlinable public func encode(to encoder: Encoder) throws
```

### `fans()`

``` swift
fileprivate func fans() -> (in: Int, out: Int)
```
