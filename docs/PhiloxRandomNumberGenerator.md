# PhiloxRandomNumberGenerator

An implementation of `SeedableRandomNumberGenerator` using Philox.
Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
http://www.thesalmons.org/john/random123/papers/random123sc11.pdf

``` swift
public struct PhiloxRandomNumberGenerator: SeedableRandomNumberGenerator
```

This struct implements a 10-round Philox4x32 PRNG. It must be seeded with
a 64-bit value.

An individual generator is not thread-safe, but distinct generators do not
share state. The random data generated is of high-quality, but is not
suitable for cryptographic applications.

## Inheritance

[`SeedableRandomNumberGenerator`](/SeedableRandomNumberGenerator)

## Initializers

### `init(uint64Seed:)`

``` swift
public init(uint64Seed seed: UInt64)
```

### `init(seed:)`

``` swift
public init(seed: [UInt8])
```

## Properties

### `global`

``` swift
var global
```

### `ctr`

``` swift
var ctr: UInt64
```

### `key`

``` swift
let key: UInt32x2
```

### `useNextValue`

``` swift
var useNextValue
```

### `nextValue`

``` swift
var nextValue: UInt64
```

## Methods

### `bump(key:)`

``` swift
private func bump(key: UInt32x2) -> UInt32x2
```

### `round(ctr:key:)`

``` swift
private func round(ctr: UInt32x4, key: UInt32x2) -> UInt32x4
```

### `random(forCtr:key:)`

``` swift
private func random(forCtr initialCtr: UInt32x4, key initialKey: UInt32x2) -> UInt32x4
```

### `next()`

``` swift
public mutating func next() -> UInt64
```
