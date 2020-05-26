# withoutDerivative(at:in:)

Applies the given closure `body` to `x`. When used in a context where `x` is
being differentiated with respect to, this function will not produce any
derivative at `x`.

``` swift
@inlinable @inline(__always) @_semantics("autodiff.nonvarying") public func withoutDerivative<T, R>(at x: T, in body: (T) -> R) -> R
```
