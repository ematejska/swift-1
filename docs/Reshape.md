# Reshape

A reshape layer.

``` swift
@frozen public struct Reshape<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init(shape:)`

Creates a reshape layer.

``` swift
public init(shape: Tensor<Int32>)
```

#### Parameters

  - shape: - shape: The target shape, represented by a tensor.

### `init(_:)`

Creates a reshape layer.

``` swift
public init(_ shape: TensorShape)
```

#### Parameters

  - shape: - shape: The target shape.

## Properties

### `shape`

The target shape.

``` swift
let shape: Tensor<Int32>
```

### `_nontrivial`

``` swift
var _nontrivial
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
