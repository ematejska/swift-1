# GlobalAvgPool1D

A global average pooling layer for temporal data.

``` swift
@frozen public struct GlobalAvgPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init()`

Creates a global average pooling layer.

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
