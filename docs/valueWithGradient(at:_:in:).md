# valueWithGradient(at:\_:in:)

``` swift
@inlinable public func valueWithGradient<T, U, R>(at x: T, _ y: U, in f: @differentiable (T, U) -> R) -> (value: R, gradient: (T.TangentVector, U.TangentVector)) where R: FloatingPoint, R.TangentVector == R
```

# valueWithGradient(at:\_:in:)

``` swift
@inlinable public func valueWithGradient<T, U, R>(at x: T, _ y: U, in f: @differentiable (T, U) -> Tensor<R>) -> (value: Tensor<R>, gradient: (T.TangentVector, U.TangentVector)) where T: Differentiable, U: Differentiable, R: TensorFlowFloatingPoint
```
