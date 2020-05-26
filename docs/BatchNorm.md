# BatchNorm

A batch normalization layer.

``` swift
@frozen public struct BatchNorm<Scalar: TensorFlowFloatingPoint>: Layer
```

Normalizes the activations of the previous layer at each batch, i.e. applies a transformation
that maintains the mean activation close to `0` and the activation standard deviation close to
`1`.

Reference: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift](https://arxiv.org/abs/1502.03167).

## Inheritance

[`Layer`](/Layer)

## Initializers

### `init(axis:momentum:offset:scale:epsilon:runningMean:runningVariance:)`

Creates a batch normalization layer.

``` swift
public init(axis: Int, momentum: Scalar, offset: Tensor<Scalar>, scale: Tensor<Scalar>, epsilon: Scalar, runningMean: Tensor<Scalar>, runningVariance: Tensor<Scalar>)
```

#### Parameters

  - axis: - axis: The axis that should not be normalized (typically the feature axis).
  - momentum: - momentum: The momentum for the moving average.
  - offset: - offset: The offset to be added to the normalized tensor.
  - scale: - scale: The scale to multiply the normalized tensor by.
  - epsilon: - epsilon: A small scalar added to the denominator to improve numerical stability.
  - runningMean: - runningMean: The running mean.
  - runningVariance: - runningVariance: The running variance.

### `init(featureCount:axis:momentum:epsilon:)`

Creates a batch normalization layer.

``` swift
public init(featureCount: Int, axis: Int = -1, momentum: Scalar = 0.99, epsilon: Scalar = 0.001)
```

#### Parameters

  - featureCount: - featureCount: The number of features.
  - axis: - axis: The axis that should be normalized (typically the features axis).
  - momentum: - momentum: The momentum for the moving average.
  - epsilon: - epsilon: A small scalar added to the denominator to improve numerical stability.

## Properties

### `axis`

The feature dimension.

``` swift
let axis: Int
```

### `momentum`

The momentum for the running mean and running variance.

``` swift
let momentum: Scalar
```

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

### `epsilon`

The variance epsilon value.

``` swift
let epsilon: Scalar
```

### `runningMean`

The running mean.

``` swift
var runningMean: Parameter<Scalar>
```

### `runningVariance`

The running variance.

``` swift
var runningVariance: Parameter<Scalar>
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

### `doTraining(_:offset:scale:axis:)`

``` swift
private func doTraining(_ input: Tensor<Scalar>, offset: Tensor<Scalar>, scale: Tensor<Scalar>, axis: Int) -> Tensor<Scalar>
```

### `doInference(_:offset:scale:)`

``` swift
private func doInference(_ input: Tensor<Scalar>, offset: Tensor<Scalar>, scale: Tensor<Scalar>) -> Tensor<Scalar>
```
