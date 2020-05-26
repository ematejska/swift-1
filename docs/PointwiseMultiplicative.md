# PointwiseMultiplicative

A type with values that support pointwise multiplication.

``` swift
public protocol PointwiseMultiplicative: AdditiveArithmetic
```

## Inheritance

`AdditiveArithmetic`

## Requirements

## one

The one value.

``` swift
var one: Self
```

One is the identity element for multiplication. For any value,
`x .* .one == x` and `.one .* x == x`.

## reciprocal

The multiplicative inverse of self.

``` swift
var reciprocal: Self
```

For any value, `x .* x.reciprocal == .one` and
`x.reciprocal .* x == .one`.

## .\*(lhs:rhs:)

Multiplies two values and produces their product.

``` swift
static func .*(lhs: Self, rhs: Self) -> Self
```

### Parameters

  - lhs: - lhs: The first value to multiply.
  - rhs: - rhs: The second value to multiply.

## .\*=(lhs:rhs:)

Multiplies two values and produces their product.

``` swift
static func .*=(lhs: inout Self, rhs: Self)
```

### Parameters

  - lhs: - lhs: The first value to multiply.
  - rhs: - rhs: The second value to multiply.
