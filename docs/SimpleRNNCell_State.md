# SimpleRNNCell.State

``` swift
public struct State: Equatable, Differentiable, VectorProtocol, KeyPathIterable
```

## Inheritance

[`Differentiable`](/Differentiable), `Equatable`, `KeyPathIterable`, [`VectorProtocol`](/VectorProtocol)

## Initializers

### `init(_:)`

``` swift
public init(_ value: Tensor<Scalar>)
```

## Properties

### `value`

``` swift
var value: Tensor<Scalar>
```
