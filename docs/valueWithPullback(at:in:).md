# valueWithPullback(at:in:)

``` swift
@inlinable public func valueWithPullback<T, R>(at x: T, in f: @differentiable (T) -> R) -> (value: R, pullback: (R.TangentVector) -> T.TangentVector)
```
