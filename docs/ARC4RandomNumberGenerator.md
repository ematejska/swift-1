# ARC4RandomNumberGenerator

An implementation of `SeedableRandomNumberGenerator` using ARC4.

``` swift
@frozen public struct ARC4RandomNumberGenerator: SeedableRandomNumberGenerator
```

ARC4 is a stream cipher that generates a pseudo-random stream of bytes. This
PRNG uses the seed as its key.

ARC4 is described in Schneier, B., "Applied Cryptography: Protocols,
Algorithms, and Source Code in C", 2nd Edition, 1996.

An individual generator is not thread-safe, but distinct generators do not
share state. The random data generated is of high-quality, but is not
suitable for cryptographic applications.

## Inheritance

[`SeedableRandomNumberGenerator`](/SeedableRandomNumberGenerator)

## Initializers

### `init(seed:)`

Initialize ARC4RandomNumberGenerator using an array of UInt8. The array
must have length between 1 and 256 inclusive.

``` swift
public init(seed: [UInt8])
```

## Properties

### `global`

``` swift
var global
```

### `state`

``` swift
var state: [UInt8]
```

### `iPos`

``` swift
var iPos: UInt8
```

### `jPos`

``` swift
var jPos: UInt8
```

## Methods

### `next()`

``` swift
public mutating func next() -> UInt64
```

### `S(_:)`

``` swift
private func S(_ index: UInt8) -> UInt8
```

### `swapAt(_:_:)`

``` swift
private mutating func swapAt(_ i: UInt8, _ j: UInt8)
```

### `nextByte()`

``` swift
private mutating func nextByte() -> UInt8
```
