# TransposedConv3D

A 3-D transposed convolution layer (e.g. spatial transposed convolution over images).

``` swift
@frozen public struct TransposedConv3D<Scalar: TensorFlowFloatingPoint>: Layer
```

This layer creates a convolution filter that is transpose-convolved with the layer input
to produce a tensor of outputs.

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

Creates a `TransposedConv3D` layer with the specified filter, bias,
activation function, strides, and padding.

``` swift
public init(filter: Tensor<Scalar>, bias: Tensor<Scalar>? = nil, activation: @escaping Activation = identity, strides: (Int, Int, Int) = (1, 1, 1), padding: Padding = .valid)
```

#### Parameters

  - filter: - filter: The 5-D convolution kernel.
  - bias: - bias: The bias vector.
  - activation: - activation: The element-wise activation function.
  - strides: - strides: The strides of the sliding window for spatial dimensions.
  - padding: - padding: The padding algorithm for convolution.

### `init(filterShape:strides:padding:activation:useBias:filterInitializer:biasInitializer:)`

Creates a `TransposedConv3D` layer with the specified filter shape, strides, padding, and
element-wise activation function. The filter tensor is initialized using Glorot uniform
initialization with the specified generator. The bias vector is initialized with zeros.

``` swift
public init(filterShape: (Int, Int, Int, Int, Int), strides: (Int, Int, Int) = (1, 1, 1), padding: Padding = .valid, activation: @escaping Activation = identity, useBias: Bool = true, filterInitializer: ParameterInitializer<Scalar> = glorotUniform(), biasInitializer: ParameterInitializer<Scalar> = zeros())
```

#### Parameters

  - filterShape: - filterShape: The shape of the 5-D convolution kernel.
  - strides: - strides: The strides of the sliding window for spatial dimensions.
  - padding: - padding: The padding algorithm for convolution.
  - activation: - activation: The element-wise activation function.
  - generator: - generator: The random number generator for initialization.

## Properties

### `filter`

The 5-D convolution kernel.

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

### `paddingIndex`

The paddingIndex property allows us to handle computation based on padding.

``` swift
let paddingIndex: Int
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
