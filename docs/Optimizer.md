# Optimizer

A numerical optimizer.

``` swift
public protocol Optimizer: CopyableToDevice
```

Optimizers apply an optimization algorithm to update a differentiable model.

## Inheritance

[`CopyableToDevice`](/CopyableToDevice)

## Requirements

## Model

The type of the model to optimize.

``` swift
associatedtype Model
```

## Scalar

The scalar parameter type.

``` swift
associatedtype Scalar
```

## learningRate

The learning rate.

``` swift
var learningRate: Scalar
```

## update(\_:along:)

Updates the given model along the given direction.

``` swift
mutating func update(_ model: inout Model, along direction: Model.TangentVector)
```
