# NormalDistribution

``` swift
@frozen public struct NormalDistribution<T: BinaryFloatingPoint>: RandomDistribution where T.RawSignificand: FixedWidthInteger
```

## Inheritance

[`RandomDistribution`](/RandomDistribution)

## Initializers

### `init(mean:standardDeviation:)`

``` swift
public init(mean: T = 0, standardDeviation: T = 1)
```

## Properties

### `mean`

``` swift
let mean: T
```

### `standardDeviation`

``` swift
let standardDeviation: T
```

### `uniformDist`

``` swift
let uniformDist
```

## Methods

### `next(using:)`

``` swift
public func next<G: RandomNumberGenerator>(using rng: inout G) -> T
```
