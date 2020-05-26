# gradient(of:)

``` swift
@inlinable public func gradient<T, R>(of f: @escaping @differentiable (T) -> R) -> (T) -> T.TangentVector where R: FloatingPoint, R.TangentVector == R
```

# gradient(of:)

``` swift
@inlinable public func gradient<T, U, R>(of f: @escaping @differentiable (T, U) -> R) -> (T, U) -> (T.TangentVector, U.TangentVector) where R: FloatingPoint, R.TangentVector == R
```

# gradient(of:)

``` swift
@inlinable public func gradient<T, U, V, R>(of f: @escaping @differentiable (T, U, V) -> R) -> (T, U, V) -> (T.TangentVector, U.TangentVector, V.TangentVector) where R: FloatingPoint, R.TangentVector == R
```

# gradient(of:)

``` swift
@inlinable public func gradient<T, R>(of f: @escaping @differentiable (T) -> Tensor<R>) -> (T) -> T.TangentVector where T: Differentiable, R: TensorFlowFloatingPoint
```

# gradient(of:)

``` swift
@inlinable public func gradient<T, U, R>(of f: @escaping @differentiable (T, U) -> Tensor<R>) -> (T, U) -> (T.TangentVector, U.TangentVector) where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint
```

# gradient(of:)

``` swift
@inlinable public func gradient<T, U, V, R>(of f: @escaping @differentiable (T, U, V) -> Tensor<R>) -> (T, U, V) -> (T.TangentVector, U.TangentVector, V.TangentVector) where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint
```
