# sign(\_:)

Returns an indication of the sign of the specified tensor element-wise.
Specifically, computes `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

``` swift
@inlinable @differentiable(where T: TensorFlowFloatingPoint) public func sign<T: Numeric>(_ x: Tensor<T>) -> Tensor<T>
```
