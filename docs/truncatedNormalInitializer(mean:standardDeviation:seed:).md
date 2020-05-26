# truncatedNormalInitializer(mean:standardDeviation:seed:)

Returns a function that creates a tensor by initializing all its values randomly from a
truncated Normal distribution. The generated values follow a Normal distribution with mean
`mean` and standard deviation `standardDeviation`, except that values whose magnitude is more
than two standard deviations from the mean are dropped and resampled.

``` swift
public func truncatedNormalInitializer<Scalar: TensorFlowFloatingPoint>(mean: Tensor<Scalar> = Tensor<Scalar>(0), standardDeviation: Tensor<Scalar> = Tensor<Scalar>(1), seed: TensorFlowSeed = Context.local.randomSeed) -> ParameterInitializer<Scalar>
```

## Parameters

  - mean: - mean: Mean of the Normal distribution.
  - standardDeviation: - standardDeviation: Standard deviation of the Normal distribution.

## Returns

A truncated normal parameter initializer function.
