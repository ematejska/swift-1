# sigmoid(\_:)

Returns the sigmoid of the specified tensor element-wise.
Specifically, computes `1 / (1 + exp(-x))`.

``` swift
@inlinable @differentiable public func sigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```
