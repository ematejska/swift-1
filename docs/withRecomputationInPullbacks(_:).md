# withRecomputationInPullbacks(\_:)

Make a function be recomputed in its pullback, known as "checkpointing" in
traditional automatic differentiation.

``` swift
@inlinable public func withRecomputationInPullbacks<T, U>(_ body: @escaping @differentiable (T) -> U) -> @differentiable (T) -> U where T: Differentiable, U: Differentiable
```
