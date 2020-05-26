# depthwiseConv2D(\_:filter:strides:padding:)

Returns a 2-D depthwise convolution with the specified input, filter, strides, and padding.

``` swift
@differentiable(wrt: (input, filter)) public func depthwiseConv2D<Scalar: TensorFlowFloatingPoint>(_ input: Tensor<Scalar>, filter: Tensor<Scalar>, strides: (Int, Int, Int, Int), padding: Padding) -> Tensor<Scalar>
```

> Precondition: \`input\` must have rank 4.

> Precondition: \`filter\` must have rank 4.

## Parameters

  - input: - input: The input.
  - filter: - filter: The depthwise convolution filter.
  - strides: - strides: The strides of the sliding filter for each dimension of the input.
  - padding: - padding: The padding for the operation.
