# logSigmoid(\_:)

Returns the log-sigmoid of the specified tensor element-wise. Specifically,
`log(1 / (1 + exp(-x)))`. For numerical stability, we use `-softplus(-x)`.

``` swift
@inlinable @differentiable public func logSigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```
