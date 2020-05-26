# ZeroPadding1D

A layer for adding zero-padding in the temporal dimension.

``` swift
public struct ZeroPadding1D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init(padding:)`

Creates a zero-padding 1D Layer.

``` swift
public init(padding: (Int, Int))
```

#### Parameters

  - padding: - padding: A tuple of two integers describing how many zeros to be padded at the beginning and end of the padding dimension.

### `init(padding:)`

Creates a zero-padding 1D Layer.

``` swift
public init(padding: Int)
```

#### Parameters

  - padding: - padding: An integer which describes how many zeros to be padded at the beginning and end of the padding dimension.

## Properties

### `padding`

The padding values along the temporal dimension.

``` swift
let padding: (Int, Int)
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
