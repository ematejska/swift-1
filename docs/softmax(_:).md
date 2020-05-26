# softmax(\_:)

Returns the softmax of the specified tensor along the last axis.
Specifically, computes `exp(x) / exp(x).sum(alongAxes: -1)`.

``` swift
@inlinable @differentiable public func softmax<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```
