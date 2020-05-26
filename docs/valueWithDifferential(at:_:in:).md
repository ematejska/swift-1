# valueWithDifferential(at:\_:in:)

``` swift
@inlinable public func valueWithDifferential<T, U, R>(at x: T, _ y: U, in f: @differentiable (T, U) -> R) -> (value: R,
      differential: (T.TangentVector, U.TangentVector) -> R.TangentVector)
```
