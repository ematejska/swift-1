# Conv1D

A 1-D convolution layer (e.g. temporal convolution over a time-series).

``` swift
@frozen public struct Conv1D<Scalar: TensorFlowFloatingPoint>: Layer
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

### `init(filter:bias:activation:stride:padding:dilation:)`

Creates a `Conv1D` layer with the specified filter, bias, activation function, stride,
dilation and padding.

``` swift
public init(filter: Tensor<Scalar>, bias: Tensor<Scalar>? = nil, activation: @escaping Activation = identity, stride: Int = 1, padding: Padding = .valid, dilation: Int = 1)
```

#### Parameters

  - filter: - filter: The 3-D convolution filter of shape \[filter width, input channel count, output channel count\].
  - bias: - bias: The bias vector of shape \[output channel count\].
  - activation: - activation: The element-wise activation function.
  - stride: - stride: The stride of the sliding window for the temporal dimension.
  - padding: - padding: The padding algorithm for convolution.
  - dilation: - dilation: The dilation factor for the temporal dimension.

## Properties

### `filter`

The 3-D convolution filter.

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

### `stride`

The stride of the sliding window for the temporal dimension.

``` swift
let stride: Int
```

### `padding`

The padding algorithm for convolution.

``` swift
let padding: Padding
```

### `dilation`

The dilation factor for the temporal dimension.

``` swift
let dilation: Int
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

The output width is computed as:

output width =
(input width + 2 \* padding size - (dilation \* (filter width - 1) + 1)) / stride + 1

and padding size is determined by the padding scheme.

> Note: Padding size equals zero when using \`.valid\`.

#### Parameters

  - input: - input: The input to the layer \[batch size, input width, input channel count\].

#### Returns

The output of shape \[batch size, output width, output channel count\].
