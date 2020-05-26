# AnyRandomNumberGenerator

A type-erased random number generator.

``` swift
public struct AnyRandomNumberGenerator: RandomNumberGenerator
```

The `AnyRandomNumberGenerator` type forwards random number generating operations to an
underlying random number generator, hiding its specific underlying type.

## Inheritance

`RandomNumberGenerator`

## Initializers

### `init(_:)`

``` swift
@inlinable public init(_ rng: RandomNumberGenerator)
```

#### Parameters

  - rng: - rng: A random number generator.

## Properties

### `_rng`

``` swift
var _rng: RandomNumberGenerator
```

## Methods

### `next()`

``` swift
@inlinable public mutating func next() -> UInt64
```
