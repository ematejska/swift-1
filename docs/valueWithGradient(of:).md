# valueWithGradient(of:)

``` swift
@inlinable public func valueWithGradient<T, R>(of f: @escaping @differentiable (T) -> R) -> (T) -> (value: R, gradient: T.TangentVector) where R: FloatingPoint, R.TangentVector == R
```

# valueWithGradient(of:)

``` swift
@inlinable public func valueWithGradient<T, U, R>(of f: @escaping @differentiable (T, U) -> R) -> (T, U) -> (value: R, gradient: (T.TangentVector, U.TangentVector)) where R: FloatingPoint, R.TangentVector == R
```

# valueWithGradient(of:)

``` swift
@inlinable public func valueWithGradient<T, U, V, R>(of f: @escaping @differentiable (T, U, V) -> R) -> (T, U, V)
  -> (value: R,
      gradient: (T.TangentVector, U.TangentVector, V.TangentVector)) where R: FloatingPoint, R.TangentVector == R
```

# valueWithGradient(of:)

``` swift
@inlinable public func valueWithGradient<T, R>(of f: @escaping @differentiable (T) -> Tensor<R>) -> (T) -> (value: Tensor<R>, gradient: T.TangentVector) where T: Differentiable, R: TensorFlowFloatingPoint
```

# valueWithGradient(of:)

``` swift
@inlinable public func valueWithGradient<T, U, R>(of f: @escaping @differentiable (T, U) -> Tensor<R>) -> (T, U) -> (value: Tensor<R>, gradient: (T.TangentVector, U.TangentVector)) where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint
```

# valueWithGradient(of:)

``` swift
@inlinable public func valueWithGradient<T, U, V, R>(of f: @escaping @differentiable (T, U, V) -> Tensor<R>) -> (T, U, V) -> (
  value: Tensor<R>,
  gradient: (T.TangentVector, U.TangentVector, V.TangentVector)
) where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint
```
