# squaredDifference(\_:\_:)

Returns the squared difference between `x` and `y`.

``` swift
@inlinable @differentiable(where T: TensorFlowFloatingPoint) public func squaredDifference<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

## Returns

`(x - y) ^ 2`.
