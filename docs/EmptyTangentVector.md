# EmptyTangentVector

An empty struct representing empty `TangentVector`s for parameterless layers.

``` swift
public struct EmptyTangentVector: EuclideanDifferentiable, VectorProtocol, ElementaryFunctions, PointwiseMultiplicative, KeyPathIterable
```

## Inheritance

`ElementaryFunctions`, `KeyPathIterable`, [`PointwiseMultiplicative`](/PointwiseMultiplicative), [`VectorProtocol`](/VectorProtocol), [`EuclideanDifferentiable`](/EuclideanDifferentiable)

## Nested Type Aliases

### `VectorSpaceScalar`

``` swift
public typealias VectorSpaceScalar = Float
```

### `TangentVector`

``` swift
public typealias TangentVector = Self
```

## Initializers

### `init()`

``` swift
public init()
```

## Methods

### `adding(_:)`

``` swift
public func adding(_ x: Float) -> EmptyTangentVector
```

### `add(_:)`

``` swift
public mutating func add(_ x: Float)
```

### `subtracting(_:)`

``` swift
public func subtracting(_ x: Float) -> EmptyTangentVector
```

### `subtract(_:)`

``` swift
public mutating func subtract(_ x: Float)
```

### `scaled(by:)`

``` swift
public func scaled(by scalar: Float) -> EmptyTangentVector
```

### `scale(by:)`

``` swift
public mutating func scale(by scalar: Float)
```
