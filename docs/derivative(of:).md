# derivative(of:)

``` swift
@inlinable public func derivative<T: FloatingPoint, R>(of f: @escaping @differentiable (T) -> R) -> (T) -> R.TangentVector where T.TangentVector == T
```

# derivative(of:)

``` swift
@inlinable public func derivative<T: FloatingPoint, U: FloatingPoint, R>(of f: @escaping @differentiable (T, U) -> R) -> (T, U) -> R.TangentVector where T.TangentVector == T, U.TangentVector == U
```

# derivative(of:)

``` swift
@inlinable public func derivative<T: FloatingPoint, U: FloatingPoint, V: FloatingPoint, R>(of f: @escaping @differentiable (T, U, V) -> R) -> (T, U, V) -> R.TangentVector where T.TangentVector == T, U.TangentVector == U, V.TangentVector == V
```
