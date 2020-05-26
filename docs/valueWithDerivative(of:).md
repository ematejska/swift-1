# valueWithDerivative(of:)

``` swift
@inlinable public func valueWithDerivative<T: FloatingPoint, R>(of f: @escaping @differentiable (T) -> R) -> (T) -> (value: R, derivative: R.TangentVector) where T.TangentVector == T
```

# valueWithDerivative(of:)

``` swift
@inlinable public func valueWithDerivative<T: FloatingPoint, U: FloatingPoint, R>(of f: @escaping @differentiable (T, U) -> R) -> (T, U) -> (value: R, derivative: R.TangentVector) where T.TangentVector == T, U.TangentVector == U
```

# valueWithDerivative(of:)

``` swift
@inlinable public func valueWithDerivative<T: FloatingPoint, U: FloatingPoint, V: FloatingPoint, R>(of f: @escaping @differentiable (T, U, V) -> R) -> (T, U, V) -> (value: R, derivative: R.TangentVector) where T.TangentVector == T, U.TangentVector == U, V.TangentVector == V
```
