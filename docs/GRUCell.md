# GRUCell

An GRU cell.

``` swift
public struct GRUCell<Scalar: TensorFlowFloatingPoint>: RNNCell
```

## Inheritance

[`RNNCell`](/RNNCell)

## Nested Type Aliases

### `TimeStepInput`

``` swift
public typealias TimeStepInput = Tensor<Scalar>
```

### `TimeStepOutput`

``` swift
public typealias TimeStepOutput = State
```

### `Input`

``` swift
public typealias Input = RNNCellInput<TimeStepInput, State>
```

### `Output`

``` swift
public typealias Output = RNNCellOutput<TimeStepOutput, State>
```

## Initializers

### `init(inputSize:hiddenSize:weightInitializer:biasInitializer:)`

Creates a `GRUCell` with the specified input size and hidden state size.

``` swift
public init(inputSize: Int, hiddenSize: Int, weightInitializer: ParameterInitializer<Scalar> = glorotUniform(), biasInitializer: ParameterInitializer<Scalar> = zeros())
```

#### Parameters

  - inputSize: - inputSize: The number of features in 2-D input tensors.
  - hiddenSize: - hiddenSize: The number of features in 2-D hidden states.

## Properties

### `updateWeight2`

``` swift
var updateWeight2: Tensor<Scalar>
```

### `updateWeight1`

``` swift
var updateWeight1
```

### `resetWeight2`

``` swift
var resetWeight2: Tensor<Scalar>
```

### `resetWeight1`

``` swift
var resetWeight1
```

### `outputWeight2`

``` swift
var outputWeight2: Tensor<Scalar>
```

### `outputWeight1`

``` swift
var outputWeight1
```

### `resetBias`

``` swift
var resetBias: Tensor<Scalar>
```

### `updateBias`

``` swift
var updateBias
```

### `outputBias`

``` swift
var outputBias
```

### `stateShape`

``` swift
var stateShape: TensorShape
```

## Methods

### `zeroState(for:)`

``` swift
public func zeroState(for input: Tensor<Scalar>) -> State
```

### `callAsFunction(_:)`

Returns the output obtained from applying the layer to the given input.

``` swift
@differentiable public func callAsFunction(_ input: Input) -> Output
```

#### Parameters

  - input: - input: The input to the layer.

#### Returns

The hidden state.
