# Differentiable

A type that mathematically represents a differentiable manifold whose
tangent spaces are finite-dimensional.

``` swift
public protocol Differentiable
```

## Requirements

## TangentVector

A type representing a differentiable value's derivatives.

``` swift
associatedtype TangentVector
```

Mathematically, this is equivalent to the tangent bundle of the
differentiable manifold represented by the differentiable type.

## move(along:)

Moves `self` along the given direction. In Riemannian geometry, this is
equivalent to exponential map, which moves `self` on the geodesic surface
along the given tangent vector.

``` swift
mutating func move(along direction: TangentVector)
```

## zeroTangentVectorInitializer

A closure that produces a zero tangent vector, capturing minimal
necessary information from `self`.

``` swift
var zeroTangentVectorInitializer: () -> TangentVector
```

`move(along: zeroTangentVectorInitializer())` should not modify
`self`.

In some cases, the zero tangent vector of `self` is equal to
`TangentVector.zero`. In other cases, the zero tangent vector depends on
information in `self`, such as shape for an n-dimensional array type.
For differentiable programming, it is more memory-efficient to define a
custom `zeroTangentVectorInitializer` property which returns a closure
that captures and uses only the necessary information to create a zero
tangent vector. For example:

``` 
struct Vector {
    var scalars: [Float]
    var count: Int { scalars.count }
    init(scalars: [Float]) { ... }
    init(repeating repeatedElement: Float, count: Int) { ... }
}

extension Vector: AdditiveArithmetic { ... }

extension Vector: Differentiable {
    typealias TangentVector = Vector

    @noDerivative
    var zeroTangentVectorInitializer: () -> TangentVector {
        let count = self.count
        return { TangentVector(repeating: 0, count: count) }
    }
}
```
