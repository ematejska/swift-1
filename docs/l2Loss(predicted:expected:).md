# l2Loss(predicted:expected:)

Returns the L2 loss between predictions and expectations.

``` swift
@differentiable(wrt: predicted) public func l2Loss<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
