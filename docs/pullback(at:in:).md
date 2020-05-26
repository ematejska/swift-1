# pullback(at:in:)

``` swift
@inlinable public func pullback<T, R>(at x: T, in f: @differentiable (T) -> R) -> (R.TangentVector) -> T.TangentVector
```
