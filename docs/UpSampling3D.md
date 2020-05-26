# UpSampling3D

An upsampling layer for 3-D inputs.

``` swift
@frozen public struct UpSampling3D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Initializers

### `init(size:)`

Creates an upsampling layer.

``` swift
public init(size: Int)
```

#### Parameters

  - size: - size: The upsampling factor for rows and columns.

## Properties

### `size`

``` swift
let size: Int
```

## Methods

### `repeatingElements(_:alongAxis:count:)`

Repeats the elements of a tensor along an axis, like `np.repeat`.
Function adapted from `def repeat_elements`:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/backend.py

``` swift
@differentiable private func repeatingElements(_ input: Tensor<Scalar>, alongAxis axis: Int, count: Int) -> Tensor<Scalar>
```

### `_vjpRepeatingElements(_:alongAxis:count:)`

``` swift
private func _vjpRepeatingElements(_ input: Tensor<Scalar>, alongAxis axis: Int, count: Int) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> (TangentVector, Tensor<Scalar>))
```

### `callAsFunction(_:)`

Returns the output obtained from applying the layer to the given input.

``` swift
@differentiable public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar>
```

#### Parameters

  - input: - input: The input to the layer.

#### Returns

The output.
