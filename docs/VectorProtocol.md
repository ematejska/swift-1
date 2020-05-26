# VectorProtocol

A type that represents an unranked vector space. Values of this type are
elements in this vector space and have either no shape or a static shape.

``` swift
public protocol VectorProtocol: AdditiveArithmetic
```

## Inheritance

`AdditiveArithmetic`

## Requirements

## VectorSpaceScalar

The type of scalars in the vector space.

``` swift
associatedtype VectorSpaceScalar
```

## adding(\_:)

``` swift
func adding(_ x: VectorSpaceScalar) -> Self
```

## add(\_:)

``` swift
mutating func add(_ x: VectorSpaceScalar)
```

## subtracting(\_:)

``` swift
func subtracting(_ x: VectorSpaceScalar) -> Self
```

## subtract(\_:)

``` swift
mutating func subtract(_ x: VectorSpaceScalar)
```

## scaled(by:)

Returns `self` multiplied by the given scalar.

``` swift
func scaled(by scalar: VectorSpaceScalar) -> Self
```

## scale(by:)

Multiplies `self` by the given scalar.

``` swift
mutating func scale(by scalar: VectorSpaceScalar)
```
