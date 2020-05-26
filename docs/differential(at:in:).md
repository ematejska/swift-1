# differential(at:in:)

``` swift
@inlinable public func differential<T, R>(at x: T, in f: @differentiable (T) -> R) -> (T.TangentVector) -> R.TangentVector
```
