# valueWithDerivative(at:\_:in:)

``` swift
@inlinable public func valueWithDerivative<T: FloatingPoint, U: FloatingPoint, R>(at x: T, _ y: U, in f: @escaping @differentiable (T, U) -> R) -> (value: R, derivative: R.TangentVector) where T.TangentVector == T, U.TangentVector == U
```
