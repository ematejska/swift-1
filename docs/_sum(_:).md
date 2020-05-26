# \_sum(\_:)

Workaround for TF-1030 so that we can use sum as a default argument for reductions.
`Tensor<Scalar>.sum()` is the preferred way to do this.

``` swift
@differentiable public func _sum<Scalar: TensorFlowFloatingPoint>(_ value: Tensor<Scalar>) -> Tensor<Scalar>
```
