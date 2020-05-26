# AdaMax

AdaMax optimizer.

``` swift
public class AdaMax<Model: Differentiable & KeyPathIterable>: Optimizer where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions
    & KeyPathIterable, Model.TangentVector.VectorSpaceScalar == Float
```

A variant of Adam based on the infinity-norm.

Reference: Section 7 of ["Adam - A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980v8)

## Inheritance

[`Optimizer`](/Optimizer)

## Nested Type Aliases

### `Model`

``` swift
public typealias Model = Model
```

## Initializers

### `init(for:learningRate:beta1:beta2:epsilon:decay:)`

Note: The default parameters follow those provided in the paper.

``` swift
public init(for model: __shared Model, learningRate: Float = 0.002, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8, decay: Float = 0)
```

### `init(copying:to:)`

``` swift
public required init(copying other: AdaMax, to device: Device)
```

## Properties

### `learningRate`

The learning rate.

``` swift
var learningRate: Float
```

### `beta1`

Decay rate used to estimate the first moment (mean) of gradients.

``` swift
var beta1: Float
```

### `beta2`

Decay rate used to estimate the exponentially weighted infinity norm.

``` swift
var beta2: Float
```

### `epsilon`

A small scalar added to the denominator to improve numerical stability.

``` swift
var epsilon: Float
```

### `decay`

The learning rate decay.

``` swift
var decay: Float
```

### `step`

The step count.

``` swift
var step: Int
```

### `firstMoments`

The first moments of the weights.

``` swift
var firstMoments: Model.TangentVector
```

### `infinityNorm`

The exponentially weighted infinity norm of the weights.

``` swift
var infinityNorm: Model.TangentVector
```

## Methods

### `update(_:along:)`

``` swift
public func update(_ model: inout Model, along direction: Model.TangentVector)
```
