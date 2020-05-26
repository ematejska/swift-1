# valueWithGradient(at:\_:\_:in:)

``` swift
@inlinable public func valueWithGradient<T, U, V, R>(at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R) -> (value: R,
      gradient: (T.TangentVector, U.TangentVector, V.TangentVector)) where R: FloatingPoint, R.TangentVector == R
```

# valueWithGradient(at:\_:\_:in:)

``` swift
@inlinable public func valueWithGradient<T, U, V, R>(at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> Tensor<R>) -> (value: Tensor<R>, gradient: (T.TangentVector, U.TangentVector, V.TangentVector)) where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint
```
