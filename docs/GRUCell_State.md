# GRUCell.State

``` swift
public struct State: Equatable, Differentiable, VectorProtocol, KeyPathIterable
```

## Inheritance

[`Differentiable`](/Differentiable), `Equatable`, `KeyPathIterable`, [`VectorProtocol`](/VectorProtocol)

## Initializers

### `init(hidden:)`

``` swift
@differentiable public init(hidden: Tensor<Scalar>)
```

## Properties

### `hidden`

``` swift
var hidden: Tensor<Scalar>
```
