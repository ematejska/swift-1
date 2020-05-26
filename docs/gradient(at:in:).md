# gradient(at:in:)

``` swift
@inlinable public func gradient<T, R>(at x: T, in f: @differentiable (T) -> R) -> T.TangentVector where R: FloatingPoint, R.TangentVector == R
```

# gradient(at:in:)

``` swift
@inlinable public func gradient<T, R>(at x: T, in f: @differentiable (T) -> Tensor<R>) -> T.TangentVector where T: Differentiable, R: TensorFlowFloatingPoint
```
