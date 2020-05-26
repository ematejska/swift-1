# softmax(\_:alongAxis:)

Returns the softmax of the specified tensor along the specified axis.
Specifically, computes `exp(x) / exp(x).sum(alongAxes: axis)`.

``` swift
@inlinable @differentiable public func softmax<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, alongAxis axis: Int) -> Tensor<T>
```
