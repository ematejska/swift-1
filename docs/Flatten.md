# Flatten

A flatten layer.

``` swift
@frozen public struct Flatten<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

A flatten layer flattens the input when applied without affecting the batch size.

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init()`

Creates a flatten layer.

``` swift
public init()
```

## Methods

### `callAsFunction(_:)`

Returns the output obtained from applying the layer to the given input.

``` swift
@differentiable public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar>
```

#### Parameters

  - input: - input: The input to the layer.

#### Returns

The output.
