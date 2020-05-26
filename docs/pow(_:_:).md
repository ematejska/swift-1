# pow(\_:\_:)

Returns the power of the first tensor to the second tensor.

``` swift
@inlinable @differentiable public func pow<T: TensorFlowFloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T>
```

# pow(\_:\_:)

Returns the power of the scalar to the tensor, broadcasting the scalar.

``` swift
@inlinable @differentiable(wrt: rhs where T: TensorFlowFloatingPoint) public func pow<T: TensorFlowFloatingPoint>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T>
```

# pow(\_:\_:)

Returns the power of the tensor to the scalar, broadcasting the scalar.

``` swift
@inlinable @differentiable(wrt: lhs where T: TensorFlowFloatingPoint) public func pow<T: TensorFlowFloatingPoint>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T>
```

# pow(\_:\_:)

Returns the power of the tensor to the scalar, broadcasting the scalar.

``` swift
@inlinable @differentiable public func pow<T: TensorFlowFloatingPoint>(_ x: Tensor<T>, _ n: Int) -> Tensor<T>
```
