# ThreefryRandomNumberGenerator

An implementation of `SeedableRandomNumberGenerator` using Threefry.
Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
http://www.thesalmons.org/john/random123/papers/random123sc11.pdf

``` swift
public struct ThreefryRandomNumberGenerator: SeedableRandomNumberGenerator
```

This struct implements a 20-round Threefry2x32 PRNG. It must be seeded with
a 64-bit value.

An individual generator is not thread-safe, but distinct generators do not
share state. The random data generated is of high-quality, but is not
suitable for cryptographic applications.

## Inheritance

[`SeedableRandomNumberGenerator`](/SeedableRandomNumberGenerator)

## Initializers

### `init(uint64Seed:)`

``` swift
internal init(uint64Seed seed: UInt64)
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

### `rot`

``` swift
let rot: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
```

### `ctr`

``` swift
var ctr: UInt64
```

### `key`

``` swift
let key: UInt32x2
```

## Methods

### `rotl32(value:n:)`

``` swift
private func rotl32(value: UInt32, n: UInt32) -> UInt32
```

### `random(forCtr:key:)`

``` swift
private func random(forCtr ctr: UInt32x2, key: UInt32x2) -> UInt32x2
```

### `next()`

``` swift
public mutating func next() -> UInt64
```
