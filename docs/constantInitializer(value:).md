# constantInitializer(value:)

Returns a function that creates a tensor by initializing all its values to the provided value.

``` swift
public func constantInitializer<Scalar: TensorFlowFloatingPoint>(value: Scalar) -> ParameterInitializer<Scalar>
```

# constantInitializer(value:)

Returns a function that creates a tensor by initializing it to the provided value. Note that
broadcasting of the provided value is *not* supported.

``` swift
public func constantInitializer<Scalar: TensorFlowFloatingPoint>(value: Tensor<Scalar>) -> ParameterInitializer<Scalar>
```
