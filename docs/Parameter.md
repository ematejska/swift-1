# Parameter

A mutable, shareable, owning reference to a tensor.

``` swift
public final class Parameter<Scalar: TensorFlowScalar>
```

## Inheritance

[`CopyableToDevice`](/CopyableToDevice)

## Initializers

### `init(copying:to:)`

Creates a copy of `other` on the given `Device`.

``` swift
public convenience init(copying other: Parameter, to device: Device)
```

### `init(_:)`

``` swift
public init(_ value: Tensor<Scalar>)
```

## Properties

### `value`

``` swift
var value: Tensor<Scalar>
```
