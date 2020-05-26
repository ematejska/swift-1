# huberLoss(predicted:expected:delta:reduction:)

Computes the Huber loss between `predicted` and `expected`.

``` swift
@differentiable(wrt: predicted) public func huberLoss<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>, delta: Scalar, reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _sum) -> Tensor<Scalar>
```

For each value `x` in `error = expected - predicted`:

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
  - delta: - delta: A floating point scalar representing the point where the Huber loss function changes from quadratic to linear.
  - reduction: - reduction: Reduction to apply on the computed element-wise loss values.
