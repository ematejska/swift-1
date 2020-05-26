# heNormal(seed:)

Returns a function that creates a tensor by performing He (Kaiming) normal initialization for the
specified shape, randomly sampling scalar values from a truncated normal distribution centered
on `0` with standard deviation `sqrt(2 / fanIn)`, where `fanIn` represents the number of input
features multiplied by the receptive field size, if present.

``` swift
public func heNormal<Scalar: TensorFlowFloatingPoint>(seed: TensorFlowSeed = Context.local.randomSeed) -> ParameterInitializer<Scalar>
```
