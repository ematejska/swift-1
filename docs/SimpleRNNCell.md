# SimpleRNNCell

A simple RNN cell.

``` swift
public struct SimpleRNNCell<Scalar: TensorFlowFloatingPoint>: RNNCell
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

### `init(inputSize:hiddenSize:seed:)`

Creates a `SimpleRNNCell` with the specified input size and hidden state size.

``` swift
public init(inputSize: Int, hiddenSize: Int, seed: TensorFlowSeed = Context.local.randomSeed)
```

#### Parameters

  - inputSize: - inputSize: The number of features in 2-D input tensors.
  - hiddenSize: - hiddenSize: The number of features in 2-D hidden states.
  - seed: - seed: The random seed for initialization. The default value is random.

## Properties

### `weight`

``` swift
var weight: Tensor<Scalar>
```

### `bias`

``` swift
var bias: Tensor<Scalar>
```

## Methods

### `zeroState(for:)`

Returns a zero-valued state with shape compatible with the provided input.

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
