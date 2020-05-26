# UniformIntegerDistribution

``` swift
@frozen public struct UniformIntegerDistribution<T: FixedWidthInteger>: RandomDistribution
```

## Inheritance

[`RandomDistribution`](/RandomDistribution)

## Initializers

### `init(lowerBound:upperBound:)`

``` swift
public init(lowerBound: T = T.self.min, upperBound: T = T.self.max)
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
