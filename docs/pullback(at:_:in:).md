# pullback(at:\_:in:)

``` swift
@inlinable public func pullback<T, U, R>(at x: T, _ y: U, in f: @differentiable (T, U) -> R) -> (R.TangentVector) -> (T.TangentVector, U.TangentVector)
```
