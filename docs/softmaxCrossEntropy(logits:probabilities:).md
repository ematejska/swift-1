# softmaxCrossEntropy(logits:probabilities:)

Returns the softmax cross entropy (categorical cross entropy) between logits and labels.

``` swift
@differentiable(wrt: logits) public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(logits: Tensor<Scalar>, probabilities: Tensor<Scalar>) -> Tensor<Scalar>
```

## Parameters

  - logits: - logits: One-hot encoded outputs from a neural network.
  - labels: - labels: Indices (zero-indexed) of the correct outputs.
