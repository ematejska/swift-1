# logCoshLoss(predicted:expected:reduction:)

Computes the logarithm of the hyperbolic cosine of the prediction error.
`logcosh = log((exp(x) + exp(-x))/2)`,
where x is the error `predicted - expected`

``` swift
@differentiable(wrt: predicted) public func logCoshLoss<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>, reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _mean) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
  - reduction: - reduction: Reduction to apply on the computed element-wise loss values.
