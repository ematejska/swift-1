# valueWithDifferential(at:in:)

``` swift
@inlinable public func valueWithDifferential<T, R>(at x: T, in f: @differentiable (T) -> R) -> (value: R, differential: (T.TangentVector) -> R.TangentVector)
```
