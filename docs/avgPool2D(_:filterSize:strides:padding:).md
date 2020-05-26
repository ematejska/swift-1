# avgPool2D(\_:filterSize:strides:padding:)

Returns a 2-D average pooling, with the specified filter sizes, strides,
and padding.

``` swift
@differentiable(wrt: input) public func avgPool2D<Scalar: TensorFlowFloatingPoint>(_ input: Tensor<Scalar>, filterSize: (Int, Int, Int, Int), strides: (Int, Int, Int, Int), padding: Padding) -> Tensor<Scalar>
```

## Parameters

  - input: - input: The input.
  - filterSize: - filterSize: The dimensions of the pooling kernel.
  - strides: - strides: The strides of the sliding filter for each dimension of the input.
  - padding: - padding: The padding for the operation.
