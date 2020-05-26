# UniformFloatingPointDistribution

``` swift
@frozen public struct UniformFloatingPointDistribution<T: BinaryFloatingPoint>: RandomDistribution where T.RawSignificand: FixedWidthInteger
```

## Inheritance

[`RandomDistribution`](/RandomDistribution)

## Initializers

### `init(lowerBound:upperBound:)`

``` swift
public init(lowerBound: T = 0, upperBound: T = 1)
```

## Properties

### `lowerBound`

``` swift
let lowerBound: T
```

### `upperBound`

``` swift
let upperBound: T
```

## Methods

### `next(using:)`

``` swift
public func next<G: RandomNumberGenerator>(using rng: inout G) -> T
```
