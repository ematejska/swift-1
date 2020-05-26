# differentiableFunction(from:)

Create a differentiable function from a vector-Jacobian products function.

``` swift
@inlinable public func differentiableFunction<T: Differentiable, R: Differentiable>(from vjp: @escaping (T)
           -> (value: R, pullback: (R.TangentVector) -> T.TangentVector)) -> @differentiable (T) -> R
```

# differentiableFunction(from:)

Create a differentiable function from a vector-Jacobian products function.

``` swift
@inlinable public func differentiableFunction<T, U, R>(from vjp: @escaping (T, U)
           -> (value: R, pullback: (R.TangentVector)
             -> (T.TangentVector, U.TangentVector))) -> @differentiable (T, U) -> R
```
