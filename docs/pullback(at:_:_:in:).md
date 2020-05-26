# pullback(at:\_:\_:in:)

``` swift
@inlinable public func pullback<T, U, V, R>(at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R) -> (R.TangentVector)
    -> (T.TangentVector, U.TangentVector, V.TangentVector)
```
