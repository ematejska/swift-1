# min(\_:\_:)

Returns the element-wise minimum of two tensors.

``` swift
@inlinable @differentiable(where T: TensorFlowFloatingPoint) public func min<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable
```

> Note: \`min\` supports broadcasting.

# min(\_:\_:)

Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.

``` swift
@inlinable @differentiable(wrt: rhs where T: TensorFlowFloatingPoint) public func min<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable
```

# min(\_:\_:)

Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.

``` swift
@inlinable @differentiable(wrt: lhs where T: TensorFlowFloatingPoint) public func min<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable
```
