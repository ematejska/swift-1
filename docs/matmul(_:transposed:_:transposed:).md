# matmul(\_:transposed:\_:transposed:)

Performs matrix multiplication with another tensor and produces the result.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public func matmul<Scalar: Numeric>(_ lhs: Tensor<Scalar>, transposed transposeLhs: Bool = false, _ rhs: Tensor<Scalar>, transposed transposeRhs: Bool = false) -> Tensor<Scalar>
```
