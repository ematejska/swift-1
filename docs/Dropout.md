# Dropout

A dropout layer.

``` swift
@frozen public struct Dropout<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

Dropout consists in randomly setting a fraction of input units to `0` at each update during
training time, which helps prevent overfitting.

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init(probability:)`

Creates a dropout layer.

``` swift
public init(probability: Double)
```

> Precondition: probability must be a value between 0 and 1 (inclusive).

#### Parameters

  - probability: - probability: The probability of a node dropping out.

## Properties

### `probability`

``` swift
let probability: Double
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
