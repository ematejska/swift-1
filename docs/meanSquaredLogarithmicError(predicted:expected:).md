# meanSquaredLogarithmicError(predicted:expected:)

Computes the mean squared logarithmic error between `predicted` and `expected`
`loss = square(log(expected) - log(predicted))`

``` swift
@differentiable(wrt: predicted) public func meanSquaredLogarithmicError<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>) -> Tensor<Scalar>
```

> Note: Negative tensor entries will be clamped at \`0\` to avoid undefined logarithmic behavior, as \`log(\_:)\` is undefined for negative reals.

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
