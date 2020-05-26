# valueWithDerivative(at:\_:\_:in:)

``` swift
@inlinable public func valueWithDerivative<T: FloatingPoint, U: FloatingPoint, V: FloatingPoint, R>(at x: T, _ y: U, _ z: V, in f: @escaping @differentiable (T, U, V) -> R) -> (value: R, derivative: R.TangentVector) where T.TangentVector == T, U.TangentVector == U, V.TangentVector == V
```
