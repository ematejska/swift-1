# meanAbsolutePercentageError(predicted:expected:)

Computes the mean absolute percentage error between `predicted` and `expected`.
`loss = 100 * mean(abs((expected - predicted) / abs(expected)))`

``` swift
@differentiable(wrt: predicted) public func meanAbsolutePercentageError<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
