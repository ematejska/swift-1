# derivative(at:\_:\_:in:)

``` swift
@inlinable public func derivative<T: FloatingPoint, U: FloatingPoint, V: FloatingPoint, R>(at x: T, _ y: U, _ z: V, in f: @differentiable (T, U, V) -> R) -> R.TangentVector where T.TangentVector == T, U.TangentVector == U, V.TangentVector == V
```
