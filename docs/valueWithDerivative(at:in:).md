# valueWithDerivative(at:in:)

``` swift
@inlinable public func valueWithDerivative<T: FloatingPoint, R>(at x: T, in f: @escaping @differentiable (T) -> R) -> (value: R, derivative: R.TangentVector) where T.TangentVector == T
```
