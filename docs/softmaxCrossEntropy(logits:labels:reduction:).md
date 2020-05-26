# softmaxCrossEntropy(logits:labels:reduction:)

Computes the sparse softmax cross entropy (categorical cross entropy) between logits and labels.
Use this crossentropy loss function when there are two or more label classes.
We expect labels to be provided as integers. There should be `# classes`
floating point values per feature for `logits` and a single floating point value per feature for `expected`.

``` swift
@differentiable(wrt: logits) public func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(logits: Tensor<Scalar>, labels: Tensor<Int32>, reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _mean) -> Tensor<Scalar>
```

## Parameters

  - logits: - logits: One-hot encoded outputs from a neural network.
  - labels: - labels: Indices (zero-indexed) of the correct outputs.
  - reduction: - reduction: Reduction to apply on the computed element-wise loss values.
