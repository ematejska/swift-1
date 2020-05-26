# swish(\_:)

Returns a tensor by applying the swish activation function, namely
`x * sigmoid(x)`.

``` swift
@inlinable @differentiable public func swish<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```

Source: "Searching for Activation Functions" (Ramachandran et al. 2017)
https://arxiv.org/abs/1710.05941
