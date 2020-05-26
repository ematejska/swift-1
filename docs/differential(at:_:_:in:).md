# differential(at:\_:\_:in:)

``` swift
@inlinable public func differential<T, U, V, R>(at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R) -> (T.TangentVector, U.TangentVector, V.TangentVector) -> (R.TangentVector)
```
