# elu(\_:)

Returns a tensor by applying an exponential linear unit.
Specifically, computes `exp(x) - 1` if \< 0, `x` otherwise.
See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)

``` swift
@inlinable @differentiable public func elu<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```
