# SeparableConv2D

A 2-D Separable convolution layer.

``` swift
@frozen public struct SeparableConv2D<Scalar: TensorFlowFloatingPoint>: Layer
```

This layer performs a depthwise convolution that acts separately on channels followed by
a pointwise convolution that mixes channels.

## Inheritance

[`Layer`](/Layer)

## Nested Type Aliases

### `Activation`

The element-wise activation function type.

``` swift
public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
```

## Initializers

### `init(depthwiseFilter:pointwiseFilter:bias:activation:strides:padding:)`

Creates a `SeparableConv2D` layer with the specified depthwise and pointwise filter,
bias, activation function, strides, and padding.

``` swift
public init(depthwiseFilter: Tensor<Scalar>, pointwiseFilter: Tensor<Scalar>, bias: Tensor<Scalar>? = nil, activation: @escaping Activation = identity, strides: (Int, Int) = (1, 1), padding: Padding = .valid)
```

#### Parameters

  - depthwiseFilter: - depthwiseFilter: The 4-D depthwise convolution kernel `[filter height, filter width, input channels count, channel multiplier]`.
  - pointwiseFilter: - pointwiseFilter: The 4-D pointwise convolution kernel `[1, 1, channel multiplier * input channels count, output channels count]`.
  - bias: - bias: The bias vector.
  - activation: - activation: The element-wise activation function.
  - strides: - strides: The strides of the sliding window for spatial dimensions.
  - padding: - padding: The padding algorithm for convolution.

### `init(depthwiseFilterShape:pointwiseFilterShape:strides:padding:activation:useBias:depthwiseFilterInitializer:pointwiseFilterInitializer:biasInitializer:)`

Creates a `SeparableConv2D` layer with the specified depthwise and pointwise filter shape,
strides, padding, and element-wise activation function.

``` swift
public init(depthwiseFilterShape: (Int, Int, Int, Int), pointwiseFilterShape: (Int, Int, Int, Int), strides: (Int, Int) = (1, 1), padding: Padding = .valid, activation: @escaping Activation = identity, useBias: Bool = true, depthwiseFilterInitializer: ParameterInitializer<Scalar> = glorotUniform(), pointwiseFilterInitializer: ParameterInitializer<Scalar> = glorotUniform(), biasInitializer: ParameterInitializer<Scalar> = zeros())
```

#### Parameters

  - depthwiseFilterShape: - depthwiseFilterShape: The shape of the 4-D depthwise convolution kernel.
  - pointwiseFilterShape: - pointwiseFilterShape: The shape of the 4-D pointwise convolution kernel.
  - strides: - strides: The strides of the sliding window for spatial/spatio-temporal dimensions.
  - padding: - padding: The padding algorithm for convolution.
  - activation: - activation: The element-wise activation function.
  - filterInitializer: - filterInitializer: Initializer to use for the filter parameters.
  - biasInitializer: - biasInitializer: Initializer to use for the bias parameters.

## Properties

### `depthwiseFilter`

The 4-D depthwise convolution kernel.

``` swift
var depthwiseFilter: Tensor<Scalar>
```

### `pointwiseFilter`

The 4-D pointwise convolution kernel.

``` swift
var pointwiseFilter: Tensor<Scalar>
```

### `bias`

The bias vector.

``` swift
var bias: Tensor<Scalar>
```

### `activation`

The element-wise activation function.

``` swift
let activation: Activation
```

### `strides`

The strides of the sliding window for spatial dimensions.

``` swift
let strides: (Int, Int)
```

### `padding`

The padding algorithm for convolution.

``` swift
let padding: Padding
```

### `useBias`

Note: `useBias` is a workaround for TF-1153: optional differentiation support.

``` swift
let useBias: Bool
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
