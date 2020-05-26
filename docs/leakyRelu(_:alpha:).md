# leakyRelu(\_:alpha:)

Returns a tensor by applying the leaky ReLU activation function
to the specified tensor element-wise.
Specifically, computes `max(x, x * alpha)`.

``` swift
@inlinable @differentiable(wrt: x) public func leakyRelu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, alpha: Double = 0.2) -> Tensor<T>
```
