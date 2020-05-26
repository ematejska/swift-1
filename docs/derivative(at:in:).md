# derivative(at:in:)

``` swift
@inlinable public func derivative<T: FloatingPoint, R>(at x: T, in f: @differentiable (T) -> R) -> R.TangentVector where T.TangentVector == T
```
