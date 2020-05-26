# kullbackLeiblerDivergence(predicted:expected:)

Returns the Kullback-Leibler divergence (KL divergence) between between expectations and
predictions. Given two distributions `p` and `q`, KL divergence computes `p * log(p / q)`.

``` swift
@differentiable(wrt: predicted) public func kullbackLeiblerDivergence<Scalar: TensorFlowFloatingPoint>(predicted: Tensor<Scalar>, expected: Tensor<Scalar>) -> Tensor<Scalar>
```

## Parameters

  - predicted: - predicted: Predicted outputs from a neural network.
  - expected: - expected: Expected values, i.e. targets, that correspond to the correct output.
