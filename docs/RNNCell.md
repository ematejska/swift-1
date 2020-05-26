# RNNCell

A recurrent neural network cell.

``` swift
public protocol RNNCell: Layer
```

## Inheritance

[`Layer`](/Layer)

## Requirements

## TimeStepInput

The input at a time step.

``` swift
associatedtype TimeStepInput
```

## TimeStepOutput

The output at a time step.

``` swift
associatedtype TimeStepOutput
```

## State

The state that may be preserved across time steps.

``` swift
associatedtype State
```

## zeroState(for:)

Returns a zero-valued state with shape compatible with the provided input.

``` swift
func zeroState(for input: TimeStepInput) -> State
```
