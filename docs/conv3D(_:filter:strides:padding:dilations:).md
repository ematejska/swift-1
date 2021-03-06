# conv3D(\_:filter:strides:padding:dilations:)

Returns a 3-D convolution with the specified input, filter, strides, padding and dilations.

``` swift
@differentiable(wrt: (input, filter)) public func conv3D<Scalar: TensorFlowFloatingPoint>(_ input: Tensor<Scalar>, filter: Tensor<Scalar>, strides: (Int, Int, Int, Int, Int) = (1, 1, 1, 1, 1), padding: Padding = .valid, dilations: (Int, Int, Int, Int, Int) = (1, 1, 1, 1, 1)) -> Tensor<Scalar>
```

> Precondition: \`input\` must have rank \`5\`.

> Precondition: \`filter\` must have rank 5.

## Parameters

  - input: - input: The input.
  - filter: - filter: The convolution filter.
  - strides: - strides: The strides of the sliding filter for each dimension of the input.
  - padding: - padding: The padding for the operation.
  - dilations: - dilations: The dilation factor for each dimension of the input.
