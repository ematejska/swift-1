# SGD

Stochastic gradient descent (SGD) optimizer.

``` swift
public class SGD<Model: Differentiable>: Optimizer where Model.TangentVector: VectorProtocol & ElementaryFunctions & KeyPathIterable, Model.TangentVector.VectorSpaceScalar == Float
```

An optimizer that implements stochastic gradient descent, with support for momentum, learning
rate decay, and Nesterov momentum.

## Inheritance

[`Optimizer`](/Optimizer)

## Nested Type Aliases

### `Model`

``` swift
public typealias Model = Model
```

## Initializers

### `init(for:learningRate:momentum:decay:nesterov:)`

``` swift
public init(for model: __shared Model, learningRate: Float = 0.01, momentum: Float = 0, decay: Float = 0, nesterov: Bool = false)
```

### `init(copying:to:)`

``` swift
public required init(copying other: SGD, to device: Device)
```

## Properties

### `learningRate`

The learning rate.

``` swift
var learningRate: Float
```

### `momentum`

The momentum factor. It accelerates stochastic gradient descent in the relevant direction
and dampens oscillations.

``` swift
var momentum: Float
```

### `decay`

The learning rate decay.

``` swift
var decay: Float
```

### `nesterov`

Use Nesterov momentum if true.

``` swift
var nesterov: Bool
```

### `velocity`

The velocity state of the model.

``` swift
var velocity: Model.TangentVector
```

### `step`

The set of steps taken.

``` swift
var step: Int
```

## Methods

### `update(_:along:)`

``` swift
public func update(_ model: inout Model, along direction: Model.TangentVector)
```
