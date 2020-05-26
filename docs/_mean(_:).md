# \_mean(\_:)

Workaround for TF-1030 so that we can use mean as a default argument for reductions.
`Tensor<Scalar>.mean()` is the preferred way to do this.

``` swift
@differentiable public func _mean<Scalar: TensorFlowFloatingPoint>(_ value: Tensor<Scalar>) -> Tensor<Scalar>
```
