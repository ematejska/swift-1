# l1Loss(predicted:expected:reduction:)

Computes the L1 loss between `expected` and `predicted`.
`loss = reduction(abs(expected - predicted))`

``` swift
@differentiable(wrt: predicted) public func l1Loss<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>, reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _sum) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
  - reduction: - reduction: Reduction to apply on the computed element-wise loss values.
