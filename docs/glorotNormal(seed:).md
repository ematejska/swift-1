# glorotNormal(seed:)

Returns a function that creates a tensor by performing Glorot (Xavier) normal initialization for
the specified shape, randomly sampling scalar values from a truncated normal distribution centered
on `0` with standard deviation `sqrt(2 / (fanIn + fanOut))`, where `fanIn`/`fanOut` represent
the number of input and output features multiplied by the receptive field size, if present.

``` swift
public func glorotNormal<Scalar: TensorFlowFloatingPoint>(seed: TensorFlowSeed = Context.local.randomSeed) -> ParameterInitializer<Scalar>
```
