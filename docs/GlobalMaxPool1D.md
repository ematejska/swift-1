# GlobalMaxPool1D

A global max pooling layer for temporal data.

``` swift
@frozen public struct GlobalMaxPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init()`

Creates a global max pooling layer.

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
  - context: - context: The contextual information for the layer application, e.g. the current learning phase.

#### Returns

The output.
