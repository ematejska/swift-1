# MaxPool1D

A max pooling layer for temporal data.

``` swift
@frozen public struct MaxPool1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init(poolSize:stride:padding:)`

Creates a max pooling layer.

``` swift
public init(poolSize: Int, stride: Int, padding: Padding)
```

#### Parameters

  - poolSize: - poolSize: The size of the sliding reduction window for pooling.
  - stride: - stride: The stride of the sliding window for temporal dimension.
  - padding: - padding: The padding algorithm for pooling.

## Properties

### `poolSize`

The size of the sliding reduction window for pooling.

``` swift
let poolSize: Int
```

### `stride`

The stride of the sliding window for temporal dimension.

``` swift
let stride: Int
```

### `padding`

The padding algorithm for pooling.

``` swift
let padding: Padding
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
