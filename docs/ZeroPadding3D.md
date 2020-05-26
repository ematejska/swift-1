# ZeroPadding3D

A layer for adding zero-padding in the spatial/spatio-temporal dimensions.

``` swift
public struct ZeroPadding3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init(padding:)`

Creates a zero-padding 3D Layer.

``` swift
public init(padding: ((Int, Int), (Int, Int), (Int, Int)))
```

#### Parameters

  - padding: - padding: A tuple of 3 tuples of two integers describing how many zeros to be padded at the beginning and end of each padding dimensions.

### `init(padding:)`

Creates a zero-padding 3D Layer.

``` swift
public init(padding: (Int, Int, Int))
```

#### Parameters

  - padding: - padding: Tuple of 3 integers that describes how many zeros to be padded at the beginning and end of each padding dimensions.

## Properties

### `padding`

The padding values along the spatial/spatio-temporal dimensions.

``` swift
let padding: ((Int, Int), (Int, Int), (Int, Int))
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
