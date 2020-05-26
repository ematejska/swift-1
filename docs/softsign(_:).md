# softsign(\_:)

Returns the softsign of the specified tensor element-wise.
Specifically, computes `features/ (abs(features) + 1)`.

``` swift
@inlinable @differentiable public func softsign<T: TensorFlowFloatingPoint>(_ features: Tensor<T>) -> Tensor<T>
```
