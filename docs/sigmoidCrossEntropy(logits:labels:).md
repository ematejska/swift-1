# sigmoidCrossEntropy(logits:labels:)

Returns the sigmoid cross entropy (binary cross entropy) between logits and labels.

``` swift
@differentiable(wrt: logits) public func sigmoidCrossEntropy<Scalar: TensorFlowFloatingPoint>(logits: Tensor<Scalar>, labels: Tensor<Scalar>) -> Tensor<Scalar>
```

## Parameters

  - logits: - logits: The unscaled output of a neural network.
  - labels: - labels: Integer values that correspond to the correct output.
