# BetaDistribution

``` swift
@frozen public struct BetaDistribution: RandomDistribution
```

## Inheritance

[`RandomDistribution`](/RandomDistribution)

## Initializers

### `init(alpha:beta:)`

``` swift
public init(alpha: Float = 0, beta: Float = 1)
```

## Properties

### `alpha`

``` swift
let alpha: Float
```

### `beta`

``` swift
let beta: Float
```

### `uniformDistribution`

``` swift
let uniformDistribution
```

## Methods

### `next(using:)`

``` swift
public func next<G: RandomNumberGenerator>(using rng: inout G) -> Float
```

### `chengsAlgorithmBB(_:_:_:using:)`

Returns one sample from a Beta(alpha, beta) distribution using Cheng's BB
algorithm, when both alpha and beta are greater than 1.

``` swift
private static func chengsAlgorithmBB<G: RandomNumberGenerator>(_ alpha0: Float, _ a: Float, _ b: Float, using rng: inout G) -> Float
```

#### Parameters

  - alpha: - alpha: First Beta distribution shape parameter.
  - a: - a: `min(alpha, beta)`.
  - b: - b: `max(alpha, beta)`.
  - rng: - rng: Random number generator.

#### Returns

Sample obtained using Cheng's BB algorithm.

### `chengsAlgorithmBC(_:_:_:using:)`

Returns one sample from a Beta(alpha, beta) distribution using Cheng's BC
algorithm, when at least one of alpha and beta is less than 1.

``` swift
private static func chengsAlgorithmBC<G: RandomNumberGenerator>(_ alpha0: Float, _ a: Float, _ b: Float, using rng: inout G) -> Float
```

#### Parameters

  - alpha: - alpha: First Beta distribution shape parameter.
  - a: - a: `max(alpha, beta)`.
  - b: - b: `min(alpha, beta)`.
  - rng: - rng: Random number generator.

#### Returns

Sample obtained using Cheng's BB algorithm.
