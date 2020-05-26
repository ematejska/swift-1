# AdaDelta

ADADELTA optimizer.

``` swift
public class AdaDelta<Model: Differentiable>: Optimizer where Model.TangentVector: VectorProtocol & PointwiseMultiplicative
    & ElementaryFunctions & KeyPathIterable, Model.TangentVector.VectorSpaceScalar == Float
```

ADADELTA is a more robust extension of AdaGrad. ADADELTA adapts learning rates based on a moving
window of gradient updates rather than by accumulating all past gradient norms. It can thus
adapt faster to changing dynamics of the optimization problem space.

Reference: ["ADADELTA: An Adaptive Learning Rate Method"](https://arxiv.org/abs/1212.5701)

## Inheritance

[`Optimizer`](/Optimizer)

## Nested Type Aliases

### `Model`

``` swift
public typealias Model = Model
```

## Initializers

### `init(for:learningRate:rho:epsilon:decay:)`

``` swift
public init(for model: __shared Model, learningRate: Float = 1, rho: Float = 0.95, epsilon: Float = 1e-6, decay: Float = 0)
```

### `init(copying:to:)`

``` swift
public required init(copying other: AdaDelta, to device: Device)
```

## Properties

### `learningRate`

The learning rate.

``` swift
var learningRate: Float
```

### `rho`

The decay factor, corresponding to fraction of gradient to keep at each time step.

``` swift
var rho: Float
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

### `averageSquared`

The accumulated, exponentially decaying average of squared gradients.

``` swift
var averageSquared: Model.TangentVector
```

### `accumulatedDelta`

The accumulated parameter updates.

``` swift
var accumulatedDelta: Model.TangentVector
```

## Methods

### `update(_:along:)`

``` swift
public func update(_ model: inout Model, along direction: Model.TangentVector)
```
