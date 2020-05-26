# triangularSolve(matrix:rhs:lower:adjoint:)

Returns the solution `x` to the system of linear equations represented by `Ax = b`.

``` swift
@inlinable @differentiable public func triangularSolve<T: TensorFlowFloatingPoint>(matrix: Tensor<T>, rhs: Tensor<T>, lower: Bool = true, adjoint: Bool = false) -> Tensor<T>
```

> Precondition: \`matrix\` must be a tensor with shape \`\[..., M, M\]\`.

> Precondition: \`rhs\` must be a tensor with shape \`\[..., M, K\]\`.

## Parameters

  - matrix: - matrix: The input triangular coefficient matrix, representing `A` in `Ax = b`.
  - rhs: - rhs: Right-hand side values, representing `b` in `Ax = b`.
  - lower: - lower: Whether `matrix` is lower triangular (`true`) or upper triangular (`false`). The default value is `true`.
  - adjoint: - adjoint: If `true`, solve with the adjoint of `matrix` instead of `matrix`. The default value is `false`.

## Returns

The solution `x` to the system of linear equations represented by `Ax = b`. `x` has the same shape as `b`.
