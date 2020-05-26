# derivative(at:\_:in:)

``` swift
@inlinable public func derivative<T: FloatingPoint, U: FloatingPoint, R>(at x: T, _ y: U, in f: @differentiable (T, U) -> R) -> R.TangentVector where T.TangentVector == T, U.TangentVector == U
```
