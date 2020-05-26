# selu(\_:)

Returns a tensor by applying the SeLU activation function, namely
`scale * alpha * (exp(x) - 1)` if `x < 0`, and `scale * x` otherwise.

``` swift
@inlinable @differentiable public func selu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```

> Note: This is designed to be used together with the variance scaling layer initializers. Please refer to \[Self-Normalizing Neural Networks\](https://arxiv.org/abs/1706.02515) for more information.
