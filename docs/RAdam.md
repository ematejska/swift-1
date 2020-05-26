# RAdam

RAdam optimizer.

``` swift
public class RAdam<Model: Differentiable>: Optimizer where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions
    & KeyPathIterable, Model.TangentVector.VectorSpaceScalar == Float
```

Rectified Adam, a variant of Adam that introduces a term to rectify the adaptive learning rate
variance.

Reference: \["On the Variance of the Adaptive Learning Rate and Beyond"\]
https://arxiv.org/pdf/1908.03265.pdf

## Inheritance

[`Optimizer`](/Optimizer)

## Nested Type Aliases

### `Model`

``` swift
public typealias Model = Model
```

## Initializers

### `init(for:learningRate:beta1:beta2:epsilon:decay:)`

``` swift
public init(for model: __shared Model, learningRate: Float = 1e-3, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8, decay: Float = 0)
```

### `init(copying:to:)`

``` swift
public required init(copying other: RAdam, to device: Device)
```

## Properties

### `learningRate`

The learning rate.

``` swift
var learningRate: Float
```

### `beta1`

A coefficient used to calculate the first and second moments of the gradients.

``` swift
var beta1: Float
```

### `beta2`

A coefficient used to calculate the first and second moments of the gradients.

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

The current step.

``` swift
var step: Int
```

### `firstMoments`

The first moments of the weights.

``` swift
var firstMoments: Model.TangentVector
```

### `secondMoments`

The second moments of the weights.

``` swift
var secondMoments: Model.TangentVector
```

## Methods

### `update(_:along:)`

``` swift
public func update(_ model: inout Model, along direction: Model.TangentVector)
```
