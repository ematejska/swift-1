# AMSGrad

AMSGrad optimizer.

``` swift
public class AMSGrad<Model: Differentiable & KeyPathIterable>: Optimizer where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions
    & KeyPathIterable, Model.TangentVector.VectorSpaceScalar == Float
```

This algorithm is a modification of Adam with better convergence properties when close to local
optima.

Reference: ["On the Convergence of Adam and Beyond"](https://openreview.net/pdf?id=ryQu7f-RZ)

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
public required init(copying other: AMSGrad, to device: Device)
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

### `secondMomentsMax`

The maximum of the second moments of the weights.

``` swift
var secondMomentsMax: Model.TangentVector
```

## Methods

### `update(_:along:)`

``` swift
public func update(_ model: inout Model, along direction: Model.TangentVector)
```
