# softplus(\_:)

Returns the softplus of the specified tensor element-wise.
Specifically, computes `log(exp(features) + 1)`.

``` swift
@inlinable @differentiable public func softplus<T: TensorFlowFloatingPoint>(_ features: Tensor<T>) -> Tensor<T>
```
