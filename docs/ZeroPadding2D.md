# ZeroPadding2D

A layer for adding zero-padding in the spatial dimensions.

``` swift
public struct ZeroPadding2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init(padding:)`

Creates a zero-padding 2D Layer.

``` swift
public init(padding: ((Int, Int), (Int, Int)))
```

#### Parameters

  - padding: - padding: A tuple of 2 tuples of two integers describing how many zeros to be padded at the beginning and end of each padding dimensions.

### `init(padding:)`

Creates a zero-padding 2D Layer.

``` swift
public init(padding: (Int, Int))
```

#### Parameters

  - padding: - padding: Tuple of 2 integers that describes how many zeros to be padded at the beginning and end of each padding dimensions.

## Properties

### `padding`

The padding values along the spatial dimensions.

``` swift
let padding: ((Int, Int), (Int, Int))
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
