# RNNCellInput

An input to a recurrent neural network.

``` swift
public struct RNNCellInput<Input: Differentiable, State: Differentiable>: Differentiable
```

## Inheritance

[`Differentiable`](/Differentiable)

## Initializers

### `init(input:state:)`

``` swift
@differentiable public init(input: Input, state: State)
```

## Properties

### `input`

The input at the current time step.

``` swift
var input: Input
```

### `state`

The previous state.

``` swift
var state: State
```
