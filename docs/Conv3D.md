# Conv3D

A 3-D convolution layer for spatial/spatio-temporal convolution over images.

``` swift
@frozen public struct Conv3D<Scalar: TensorFlowFloatingPoint>: Layer
```

This layer creates a convolution filter that is convolved with the layer input to produce a
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

### `init(filter:bias:activation:strides:padding:dilations:)`

Creates a `Conv3D` layer with the specified filter, bias, activation function, strides, and
padding.

``` swift
public init(filter: Tensor<Scalar>, bias: Tensor<Scalar>? = nil, activation: @escaping Activation = identity, strides: (Int, Int, Int) = (1, 1, 1), padding: Padding = .valid, dilations: (Int, Int, Int) = (1, 1, 1))
```

#### Parameters

  - filter: - filter: The 5-D convolution filter of shape \[filter depth, filter height, filter width, input channel count, output channel count\].
  - bias: - bias: The bias vector of shape \[output channel count\].
  - activation: - activation: The element-wise activation function.
  - strides: - strides: The strides of the sliding window for spatial dimensions, i.e. (stride depth, stride height, stride width)
  - padding: - padding: The padding algorithm for convolution.
  - dilations: - dilations: The dilation factor for spatial/spatio-temporal dimensions.

### `init(filterShape:strides:padding:dilations:activation:useBias:filterInitializer:biasInitializer:)`

Creates a `Conv3D` layer with the specified filter shape, strides, padding, dilations and
element-wise activation function. The filter tensor is initialized using Glorot uniform
initialization with the specified seed. The bias vector is initialized with zeros.

``` swift
public init(filterShape: (Int, Int, Int, Int, Int), strides: (Int, Int, Int) = (1, 1, 1), padding: Padding = .valid, dilations: (Int, Int, Int) = (1, 1, 1), activation: @escaping Activation = identity, useBias: Bool = true, filterInitializer: ParameterInitializer<Scalar> = glorotUniform(), biasInitializer: ParameterInitializer<Scalar> = zeros())
```

#### Parameters

  - filterShape: - filterShape: The shape of the 5-D convolution filter, representing (filter depth, filter height, filter width, input channel count, output channel count).
  - strides: - strides: The strides of the sliding window for spatial dimensions, i.e. (stride depth, stride height, stride width)
  - padding: - padding: The padding algorithm for convolution.
  - dilations: - dilations: The dilation factor for spatial/spatio-temporal dimensions.
  - activation: - activation: The element-wise activation function.
  - filterInitializer: - filterInitializer: Initializer to use for the filter parameters.
  - biasInitializer: - biasInitializer: Initializer to use for the bias parameters.

## Properties

### `filter`

The 5-D convolution filter.

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
let strides: (Int, Int, Int)
```

### `padding`

The padding algorithm for convolution.

``` swift
let padding: Padding
```

### `dilations`

The dilation factor for spatial/spatio temporal dimensions.

``` swift
let dilations: (Int, Int, Int)
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

The output spatial dimensions are computed as:

output depth =
(input depth + 2 \* padding depth - (dilation depth \* (filter depth - 1) + 1))
/ stride depth + 1

output height =
(input height + 2 \* padding height - (dilation height \* (filter height - 1) + 1))
/ stride height + 1

output width =
(input width + 2 \* padding width - (dilation width \* (filter width - 1) + 1))
/ stride width + 1

and padding sizes are determined by the padding scheme.

> Note: Padding size equals zero when using \`.valid\`.

#### Parameters

  - input: - input: The input to the layer of shape \[batch count, input depth, input height, input width, input channel count\].

#### Returns

The output of shape \[batch count, output depth, output height, output width, output channel count\].
