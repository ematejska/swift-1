# withoutDerivative(at:)

Returns `x` like an identity function. When used in a context where `x` is
being differentiated with respect to, this function will not produce any
derivative at `x`.

``` swift
@inlinable @inline(__always) @_semantics("autodiff.nonvarying") public func withoutDerivative<T>(at x: T) -> T
```
