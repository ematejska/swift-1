# valueWithGradient(at:in:)

``` swift
@inlinable public func valueWithGradient<T, R>(at x: T, in f: @differentiable (T) -> R) -> (value: R, gradient: T.TangentVector) where R: FloatingPoint, R.TangentVector == R
```

# valueWithGradient(at:in:)

``` swift
@inlinable public func valueWithGradient<T, R>(at x: T, in f: @differentiable (T) -> Tensor<R>) -> (value: Tensor<R>, gradient: T.TangentVector) where T: Differentiable, R: TensorFlowFloatingPoint
```
