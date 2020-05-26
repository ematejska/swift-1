# conv1D(\_:filter:stride:padding:dilation:)

Returns a 1-D convolution with the specified input, filter, stride, and padding.

``` swift
@differentiable(wrt: (input, filter)) public func conv1D<Scalar: TensorFlowFloatingPoint>(_ input: Tensor<Scalar>, filter: Tensor<Scalar>, stride: Int = 1, padding: Padding = .valid, dilation: Int = 1) -> Tensor<Scalar>
```

> Precondition: \`input\` must have rank \`3\`.

> Precondition: \`filter\` must have rank 3.

## Parameters

  - input: - input: The input.
  - filter: - filter: The convolution filter.
  - stride: - stride: The stride of the sliding filter.
  - padding: - padding: The padding for the operation.
  - dilation: - dilation: The dilation factor.
