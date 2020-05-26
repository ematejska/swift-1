# transpose(of:)

``` swift
@inlinable public func transpose<T, R>(of body: @escaping @differentiable(linear) (T) -> R) -> @differentiable(linear) (R) -> T
```
