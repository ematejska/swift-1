# valueWithPullback(at:\_:in:)

``` swift
@inlinable public func valueWithPullback<T, U, R>(at x: T, _ y: U, in f: @differentiable (T, U) -> R) -> (value: R,
      pullback: (R.TangentVector) -> (T.TangentVector, U.TangentVector))
```
