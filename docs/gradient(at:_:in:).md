# gradient(at:\_:in:)

``` swift
@inlinable public func gradient<T, U, R>(at x: T, _ y: U, in f: @differentiable (T, U) -> R) -> (T.TangentVector, U.TangentVector) where R: FloatingPoint, R.TangentVector == R
```

# gradient(at:\_:in:)

``` swift
@inlinable public func gradient<T, U, R>(at x: T, _ y: U, in f: @differentiable (T, U) -> Tensor<R>) -> (T.TangentVector, U.TangentVector) where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint
```
