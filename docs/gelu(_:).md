# gelu(\_:)

Returns the Gaussian Error Linear Unit (GELU) activations of the specified tensor element-wise.

``` swift
@inlinable @differentiable public func gelu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```

Specifically, `gelu` approximates `xP(X <= x)`, where `P(X <= x)` is the Standard Gaussian
cumulative distribution, by computing: x \* \[0.5 \* (1 + tanh\[√(2/π) \* (x + 0.044715 \* x^3)\])\].

See [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415).
