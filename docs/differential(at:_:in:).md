# differential(at:\_:in:)

``` swift
@inlinable public func differential<T, U, R>(at x: T, _ y: U, in f: @differentiable (T, U) -> R) -> (T.TangentVector, U.TangentVector) -> R.TangentVector
```
