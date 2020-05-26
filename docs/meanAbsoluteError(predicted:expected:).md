# meanAbsoluteError(predicted:expected:)

Computes the mean of absolute difference between labels and predictions.
`loss = mean(abs(expected - predicted))`

``` swift
@differentiable(wrt: predicted) public func meanAbsoluteError<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
