# RMSProp

RMSProp optimizer.

``` swift
public class RMSProp<Model: Differentiable>: Optimizer where Model.TangentVector: VectorProtocol & PointwiseMultiplicative
    & ElementaryFunctions & KeyPathIterable, Model.TangentVector.VectorSpaceScalar == Float
```

It is recommended to leave the parameters of this optimizer at their default values (except for
the learning rate, which can be freely tuned). This optimizer is usually a good choice for
recurrent neural networks.

Reference: ["rmsprop: Divide the gradient by a running average of its recent magnitude"](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

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
public init(for model: __shared Model, learningRate: Float = 0.001, rho: Float = 0.9, epsilon: Float = 1e-8, decay: Float = 0)
```

### `init(copying:to:)`

``` swift
public required init(copying other: RMSProp, to device: Device)
```

## Properties

### `learningRate`

The learning rate.

``` swift
var learningRate: Float
```

### `rho`

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

The step count.

``` swift
var step: Float
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
