# ParameterlessLayer

A parameterless neural network layer.

``` swift
public protocol ParameterlessLayer: Layer
```

The `TangentVector` of parameterless layers is always `EmptyTangentVector`.

## Inheritance

[`Layer`](/Layer)

## Requirements

## callAsFunction(\_:)

``` swift
@differentiable func callAsFunction(_ input: Input) -> Output
```
