# AvgPool3D

An average pooling layer for spatial or spatio-temporal data.

``` swift
@frozen public struct AvgPool3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init(poolSize:strides:padding:)`

Creates an average pooling layer.

``` swift
public init(poolSize: (Int, Int, Int, Int, Int), strides: (Int, Int, Int, Int, Int), padding: Padding)
```

### `init(poolSize:strides:padding:)`

Creates an average pooling layer.

``` swift
public init(poolSize: (Int, Int, Int), strides: (Int, Int, Int), padding: Padding = .valid)
```

#### Parameters

  - poolSize: - poolSize: Vertical and horizontal factors by which to downscale.
  - strides: - strides: The strides.
  - padding: - padding: The padding.

### `init(poolSize:strides:padding:)`

Creates an average pooling layer with the specified pooling window size and stride. All
pooling sizes and strides are the same.

``` swift
public init(poolSize: Int, strides: Int, padding: Padding = .valid)
```

## Properties

### `poolSize`

The size of the sliding reduction window for pooling.

``` swift
let poolSize: (Int, Int, Int, Int, Int)
```

### `strides`

The strides of the sliding window for each dimension of a 5-D input.
Strides in non-spatial dimensions must be `1`.

``` swift
let strides: (Int, Int, Int, Int, Int)
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
