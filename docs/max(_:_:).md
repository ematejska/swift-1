# max(\_:\_:)

Returns the element-wise maximum of two tensors.

``` swift
@inlinable @differentiable(where T: TensorFlowFloatingPoint) public func max<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable
```

> Note: \`max\` supports broadcasting.

# max(\_:\_:)

Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.

``` swift
@inlinable @differentiable(wrt: rhs where T: TensorFlowFloatingPoint) public func max<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable
```

# max(\_:\_:)

Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.

``` swift
@inlinable @differentiable(wrt: lhs where T: TensorFlowFloatingPoint) public func max<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable
```
