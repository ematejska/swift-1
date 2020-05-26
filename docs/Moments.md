# Moments

Pair of first and second moments (i.e., mean and variance).

``` swift
public struct Moments<Scalar: TensorFlowFloatingPoint>: Differentiable
```

> Note: This is needed because tuple types are not differentiable.

## Inheritance

[`Differentiable`](/Differentiable)

## Initializers

### `init(mean:variance:)`

``` swift
@differentiable public init(mean: Tensor<Scalar>, variance: Tensor<Scalar>)
```

## Properties

### `mean`

``` swift
var mean: Tensor<Scalar>
```

### `variance`

``` swift
var variance: Tensor<Scalar>
```
