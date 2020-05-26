# AdaGrad

AdaGrad optimizer.

``` swift
public class AdaGrad<Model: Differentiable>: Optimizer where Model.TangentVector: VectorProtocol & PointwiseMultiplicative
    & ElementaryFunctions & KeyPathIterable, Model.TangentVector.VectorSpaceScalar == Float
```

Individually adapts the learning rates of all model parameters by scaling them inversely
proportional to the square root of the sum of all the historical squared values of the gradient.

Reference: ["Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

## Inheritance

[`Optimizer`](/Optimizer)

## Nested Type Aliases

### `Model`

``` swift
public typealias Model = Model
```

## Initializers

### `init(for:learningRate:rho:epsilon:)`

``` swift
public init(for model: __shared Model, learningRate: Float = 0.001, rho: Float = 0.9, epsilon: Float = 1e-8)
```

### `init(copying:to:)`

``` swift
public required init(copying other: AdaGrad, to device: Device)
```

## Properties

### `learningRate`

The learning rate.

``` swift
var learningRate: Float
```

### `rho`

The smoothing factor (ρ). Typical values are `0.5`, `0.9`, and `0.99`, for smoothing over 2,
10, and 100 examples, respectively.

``` swift
var rho: Float
```

### `epsilon`

A small scalar added to the denominator to improve numerical stability.

``` swift
var epsilon: Float
```

### `alpha`

The alpha values for all model differentiable variables.

``` swift
var alpha: Model.TangentVector
```

## Methods

### `update(_:along:)`

``` swift
public func update(_ model: inout Model, along direction: Model.TangentVector)
```
