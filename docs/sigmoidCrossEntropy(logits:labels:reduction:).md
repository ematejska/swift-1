# sigmoidCrossEntropy(logits:labels:reduction:)

Computes the sigmoid cross entropy (binary cross entropy) between logits and labels.
Use this cross-entropy loss when there are only two label classes (assumed to
be 0 and 1). For each example, there should be a single floating-point value
per prediction.

``` swift
@differentiable(wrt: logits) public func sigmoidCrossEntropy<Scalar: TensorFlowFloatingPoint>(logits: Tensor<Scalar>, labels: Tensor<Scalar>, reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _mean) -> Tensor<Scalar>
```

## Parameters

  - logits: - logits: The unscaled output of a neural network.
  - labels: - labels: Integer values that correspond to the correct output.
  - reduction: - reduction: Reduction to apply on the computed element-wise loss values.
