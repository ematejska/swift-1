# cholesky(\_:)

Returns the Cholesky decomposition of one or more square matrices.

``` swift
@inlinable @differentiable public func cholesky<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices.

The input has to be symmetric and positive definite. Only the lower-triangular
part of the input will be used for this operation. The upper-triangular part
will not be read.

The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

## Parameters

  - input: - input: A tensor of shape `[..., M, M]`.
