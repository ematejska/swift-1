# EuclideanDifferentiable

A type that is differentiable in the Euclidean space.
The type may represent a vector space, or consist of a vector space and some
other non-differentiable component.

``` swift
public protocol EuclideanDifferentiable: Differentiable
```

Mathematically, this represents a product manifold that consists of
a differentiable vector space and some arbitrary manifold, where the tangent
bundle of the entire product manifold is equal to the vector space
component.

This abstraction is useful for representing common differentiable data
structures that contain both differentiable vector properties and other
stored properties that do not have a derivative, e.g.

``` swift
struct Perceptron: @memberwise EuclideanDifferentiable {
    var weight: SIMD16<Float>
    var bias: Float
    @noDerivative var useBias: Bool
}
```

> Note: Conform a type to \`EuclideanDifferentiable\` if it is differentiable only with respect to its vector space component and when its \`TangentVector\` is equal to its vector space component.

## Inheritance

[`Differentiable`](/Differentiable)

## Requirements

## differentiableVectorView

The differentiable vector component of `self`.

``` swift
var differentiableVectorView: TangentVector
```
