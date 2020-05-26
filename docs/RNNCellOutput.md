# RNNCellOutput

An output to a recurrent neural network.

``` swift
public struct RNNCellOutput<Output: Differentiable, State: Differentiable>: Differentiable
```

## Inheritance

[`Differentiable`](/Differentiable)

## Initializers

### `init(output:state:)`

``` swift
@differentiable public init(output: Output, state: State)
```

## Properties

### `output`

The output at the current time step.

``` swift
var output: Output
```

### `state`

The current state.

``` swift
var state: State
```
