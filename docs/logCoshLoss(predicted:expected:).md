# logCoshLoss(predicted:expected:)

Returns the logarithm of the hyperbolic cosine of the error between predictions and
expectations.

``` swift
@differentiable(wrt: predicted) public func logCoshLoss<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
