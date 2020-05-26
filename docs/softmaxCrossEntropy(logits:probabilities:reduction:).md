# softmaxCrossEntropy(logits:probabilities:reduction:)

Computes the sparse softmax cross entropy (categorical cross entropy) between logits and labels.
Use this crossentropy loss function when there are two or more label classes.
We expect labels to be provided provided in a `one_hot` representation.
There should be `# classes` floating point values per feature.

``` swift
@differentiable(wrt: logits) public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(logits: Tensor<Scalar>, probabilities: Tensor<Scalar>, reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _mean) -> Tensor<Scalar>
```

## Parameters

  - logits: - logits: Unscaled log probabilities from a neural network.
  - probabilities: - probabilities: Probability values that correspond to the correct output. Each row must be a valid probability distribution.
  - reduction: - reduction: Reduction to apply on the computed element-wise loss values.
