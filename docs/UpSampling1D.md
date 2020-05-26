# UpSampling1D

An upsampling layer for 1-D inputs.

``` swift
@frozen public struct UpSampling1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init(size:)`

Creates an upsampling layer.

``` swift
public init(size: Int)
```

#### Parameters

  - size: - size: The upsampling factor for timesteps.

## Properties

### `size`

``` swift
let size: Int
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
