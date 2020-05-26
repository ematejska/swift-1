# trace(\_:)

Computes the trace of an optionally batched matrix.
The trace is the the sum along the main diagonal of each inner-most matrix.

``` swift
@inlinable @differentiable(wrt: matrix where T: TensorFlowFloatingPoint) public func trace<T: TensorFlowNumeric>(_ matrix: Tensor<T>) -> Tensor<T>
```

The input is a tensor with shape `[..., M, N]`.
The output is a tensor with shape `[...]`.

> Precondition: \`matrix\` must be a tensor with shape \`\[..., M, N\]\`.

## Parameters

  - matrix: - matrix: A tensor of shape `[..., M, N]`.
