# categoricalHingeLoss(predicted:expected:reduction:)

Computes the categorical hinge loss between `predicted` and `expected`.
`loss = maximum(negative - positive + 1, 0)`
where `negative = max((1 - expected) * predicted)` and
`positive = sum(predicted * expected)`

``` swift
@differentiable(wrt: predicted) public func categoricalHingeLoss<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>, reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _mean) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
  - reduction: - reduction: Reduction to apply on the computed element-wise loss values.
