# hingeLoss(predicted:expected:reduction:)

Computes the hinge loss between `predicted` and `expected`.
`loss = reduction(max(0, 1 - predicted * expected))`
`expected` values are expected to be -1 or 1.

``` swift
@differentiable(wrt: predicted) public func hingeLoss<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>, reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _mean) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
  - reduction: - reduction: Reduction to apply on the computed element-wise loss values.
