# LayerNorm

A layer that applies layer normalization over a mini-batch of inputs.

``` swift
@frozen public struct LayerNorm<Scalar: TensorFlowFloatingPoint>: Layer
```

Reference: [Layer Normalization](https://arxiv.org/abs/1607.06450).

## Inheritance

[`Layer`](/Layer)

## Initializers

### `init(offset:scale:axis:epsilon:)`

Creates a layer normalization layer.

``` swift
public init(offset: Tensor<Scalar>, scale: Tensor<Scalar>, axis: Int, epsilon: Scalar)
```

### `init(featureCount:axis:epsilon:)`

Creates a layer normalization layer.

``` swift
public init(featureCount: Int, axis: Int, epsilon: Scalar = 0.001)
```

#### Parameters

  - featureCount: - featureCount: The number of features.
  - axis: - axis: The axis that should be normalized.
  - epsilon: - epsilon: The small scalar added to variance.

## Properties

### `offset`

The offset value, also known as beta.

``` swift
var offset: Tensor<Scalar>
```

### `scale`

The scale value, also known as gamma.

``` swift
var scale: Tensor<Scalar>
```

### `axis`

The axis.

``` swift
let axis: Int
```

### `epsilon`

The variance epsilon value.

``` swift
let epsilon: Scalar
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
