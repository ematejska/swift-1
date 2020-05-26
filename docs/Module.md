# Module

``` swift
public protocol Module: EuclideanDifferentiable, KeyPathIterable
```

## Inheritance

[`EuclideanDifferentiable`](/EuclideanDifferentiable), `KeyPathIterable`

## Requirements

## Input

The input type of the layer.

``` swift
associatedtype Input
```

## Output

The output type of the layer.

``` swift
associatedtype Output
```

## callAsFunction(\_:)

Returns the output obtained from applying the layer to the given input.

``` swift
@differentiable(wrt: self) func callAsFunction(_ input: Input) -> Output
```

### Parameters

  - input: - input: The input to the layer.

### Returns

The output.
