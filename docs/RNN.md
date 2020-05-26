# RNN

``` swift
public struct RNN<Cell: RNNCell>: Layer
```

## Inheritance

[`Layer`](/Layer)

## Nested Type Aliases

### `Input`

``` swift
public typealias Input = [Cell.TimeStepInput]
```

### `Output`

``` swift
public typealias Output = [Cell.TimeStepOutput]
```

## Initializers

### `init(_:)`

``` swift
public init(_ cell: @autoclosure () -> Cell)
```

## Properties

### `cell`

``` swift
var cell: Cell
```

## Methods

### `callAsFunction(_:initialState:)`

``` swift
@differentiable(wrt: (self, inputs, initialState)) public func callAsFunction(_ inputs: [Cell.TimeStepInput], initialState: Cell.State) -> [Cell.TimeStepOutput]
```

### `call(_:initialState:)`

``` swift
@differentiable(wrt: (self, inputs, initialState)) public func call(_ inputs: [Cell.TimeStepInput], initialState: Cell.State) -> [Cell.TimeStepOutput]
```

### `_vjpCallAsFunction(_:initialState:)`

``` swift
@usableFromInline internal func _vjpCallAsFunction(_ inputs: [Cell.TimeStepInput], initialState: Cell.State) -> (
    value: [Cell.TimeStepOutput],
    pullback: (Array<Cell.TimeStepOutput>.TangentVector)
      -> (TangentVector, Array<Cell.TimeStepInput>.TangentVector, Cell.State.TangentVector)
  )
```

### `callAsFunction(_:)`

``` swift
@differentiable public func callAsFunction(_ inputs: [Cell.TimeStepInput]) -> [Cell.TimeStepOutput]
```

### `lastOutput(from:initialState:)`

``` swift
@differentiable(wrt: (self, inputs, initialState)) public func lastOutput(from inputs: [Cell.TimeStepInput], initialState: Cell.State) -> Cell.TimeStepOutput
```

### `lastOutput(from:)`

``` swift
@differentiable(wrt: (self, inputs)) public func lastOutput(from inputs: [Cell.TimeStepInput]) -> Cell.TimeStepOutput
```
