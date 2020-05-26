# SeedableRandomNumberGenerator

A type that provides seedable deterministic pseudo-random data.

``` swift
public protocol SeedableRandomNumberGenerator: RandomNumberGenerator
```

A SeedableRandomNumberGenerator can be used anywhere where a
RandomNumberGenerator would be used. It is useful when the pseudo-random
data needs to be reproducible across runs.

# Conforming to the SeedableRandomNumberGenerator Protocol

To make a custom type conform to the `SeedableRandomNumberGenerator`
protocol, implement the `init(seed: [UInt8])` initializer, as well as the
requirements for `RandomNumberGenerator`. The values returned by `next()`
must form a deterministic sequence that depends only on the seed provided
upon initialization.

## Inheritance

`RandomNumberGenerator`

## Requirements

## init(seed:)

``` swift
init(seed: [UInt8])
```

## init(seed:)

``` swift
init<T: BinaryInteger>(seed: T)
```
