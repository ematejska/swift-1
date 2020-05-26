# gradient(at:\_:\_:in:)

``` swift
@inlinable public func gradient<T, U, V, R>(at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R) -> (T.TangentVector, U.TangentVector, V.TangentVector) where R: FloatingPoint, R.TangentVector == R
```

# gradient(at:\_:\_:in:)

``` swift
@inlinable public func gradient<T, U, V, R>(at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> Tensor<R>) -> (T.TangentVector, U.TangentVector, V.TangentVector) where T: Differentiable, U: Differentiable, V: Differentiable, R: TensorFlowFloatingPoint
```
