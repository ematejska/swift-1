# Layer

A neural network layer.

``` swift
public protocol Layer: Module
```

Types that conform to `Layer` represent functions that map inputs to outputs. They may have an
internal state represented by parameters, such as weight tensors.

`Layer` instances define a differentiable `callAsFunction(_:)` method for mapping inputs to
outputs.

## Inheritance

[`Module`](/Module)

## Requirements

## callAsFunction(\_:)

Returns the output obtained from applying the layer to the given input.

``` swift
@differentiable func callAsFunction(_ input: Input) -> Output
```

### Parameters

  - input: - input: The input to the layer.

### Returns

The output.
