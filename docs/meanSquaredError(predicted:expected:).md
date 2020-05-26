# meanSquaredError(predicted:expected:)

Computes the mean of squares of errors between labels and predictions.
`loss = mean(square(expected - predicted))`

``` swift
@differentiable(wrt: predicted) public func meanSquaredError<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
