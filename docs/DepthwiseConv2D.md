# DepthwiseConv2D

A 2-D depthwise convolution layer.

``` swift
@frozen public struct DepthwiseConv2D<Scalar: TensorFlowFloatingPoint>: Layer
```

This layer creates seperable convolution filters that are convolved with the layer input to produce a
tensor of outputs.

## Inheritance

[`Layer`](/Layer)

## Nested Type Aliases

### `Activation`

The element-wise activation function type.

``` swift
public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
```

## Initializers

### `init(filter:bias:activation:strides:padding:)`

Creates a `DepthwiseConv2D` layer with the specified filter, bias, activation function,
strides, and padding.

``` swift
public init(filter: Tensor<Scalar>, bias: Tensor<Scalar>? = nil, activation: @escaping Activation = identity, strides: (Int, Int) = (1, 1), padding: Padding = .valid)
```

#### Parameters

  - filter: - filter: The 4-D convolution kernel.
  - bias: - bias: The bias vector.
  - activation: - activation: The element-wise activation function.
  - strides: - strides: The strides of the sliding window for spatial dimensions.
  - padding: - padding: The padding algorithm for convolution.

### `init(filterShape:strides:padding:activation:useBias:filterInitializer:biasInitializer:)`

Creates a `DepthwiseConv2D` layer with the specified filter shape, strides, padding, and
element-wise activation function.

``` swift
public init(filterShape: (Int, Int, Int, Int), strides: (Int, Int) = (1, 1), padding: Padding = .valid, activation: @escaping Activation = identity, useBias: Bool = true, filterInitializer: ParameterInitializer<Scalar> = glorotUniform(), biasInitializer: ParameterInitializer<Scalar> = zeros())
```

#### Parameters

  - filterShape: - filterShape: The shape of the 4-D convolution kernel with form, \[filter width, filter height, input channel count, channel multiplier\].
  - strides: - strides: The strides of the sliding window for spatial/spatio-temporal dimensions.
  - padding: - padding: The padding algorithm for convolution.
  - activation: - activation: The element-wise activation function.
  - filterInitializer: - filterInitializer: Initializer to use for the filter parameters.
  - biasInitializer: - biasInitializer: Initializer to use for the bias parameters.

## Properties

### `filter`

The 4-D convolution kernel.

``` swift
var filter: Tensor<Scalar>
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

  - input: - input: The input to the layer of shape, \[batch count, input height, input width, input channel count\]

#### Returns

The output of shape, \[batch count, output height, output width, input channel count \* channel multiplier\]
