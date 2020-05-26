# LSTMCell.State

``` swift
public struct State: Equatable, Differentiable, VectorProtocol, KeyPathIterable
```

## Inheritance

[`Differentiable`](/Differentiable), `Equatable`, `KeyPathIterable`, [`VectorProtocol`](/VectorProtocol)

## Initializers

### `init(cell:hidden:)`

``` swift
@differentiable public init(cell: Tensor<Scalar>, hidden: Tensor<Scalar>)
```

## Properties

### `cell`

``` swift
var cell: Tensor<Scalar>
```

### `hidden`

``` swift
var hidden: Tensor<Scalar>
```
