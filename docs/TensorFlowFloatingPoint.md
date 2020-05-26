# TensorFlowFloatingPoint

A floating-point data type that conforms to `Differentiable` and is compatible with TensorFlow.

``` swift
public protocol TensorFlowFloatingPoint: TensorFlowScalar & BinaryFloatingPoint & Differentiable & ElementaryFunctions
```

> Note: \`Tensor\` conditionally conforms to \`Differentiable\` when the \`Scalar\` associated type conforms \`TensorFlowFloatingPoint\`.

## Inheritance

`BinaryFloatingPoint`, [`Differentiable`](/Differentiable), `ElementaryFunctions`, [`TensorFlowScalar`](/TensorFlowScalar)
