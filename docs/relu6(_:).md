# relu6(\_:)

Returns a tensor by applying the ReLU6 activation function, namely `min(max(0, x), 6)`.

``` swift
@inlinable @differentiable public func relu6<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```
