# cosineDistance(\_:\_:)

Returns the cosine distance between `x` and `y`. Cosine distance is defined as
`1 - cosineSimilarity(x, y)`.

``` swift
@differentiable public func cosineDistance<Scalar: TensorFlowFloatingPoint>(_ x: Tensor<Scalar>, _ y: Tensor<Scalar>) -> Tensor<Scalar>
```
