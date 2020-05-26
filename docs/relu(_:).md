# relu(\_:)

Returns a tensor by applying the ReLU activation function to the specified tensor element-wise.
Specifically, computes `max(0, x)`.

``` swift
@inlinable @differentiable public func relu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```
