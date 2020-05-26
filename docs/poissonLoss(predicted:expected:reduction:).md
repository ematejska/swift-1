# poissonLoss(predicted:expected:reduction:)

Computes the Poisson loss between predicted and expected
The Poisson loss is the mean of the elements of the `Tensor`
`predicted - expected * log(predicted)`.

``` swift
@differentiable(wrt: predicted) public func poissonLoss<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>, reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _mean) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
  - reduction: - reduction: Reduction to apply on the computed element-wise loss values.
