# LSTMCell

An LSTM cell.

``` swift
public struct LSTMCell<Scalar: TensorFlowFloatingPoint>: RNNCell
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

### `init(inputSize:hiddenSize:)`

Creates a `LSTMCell` with the specified input size and hidden state size.

``` swift
public init(inputSize: Int, hiddenSize: Int)
```

#### Parameters

  - inputSize: - inputSize: The number of features in 2-D input tensors.
  - hiddenSize: - hiddenSize: The number of features in 2-D hidden states.

## Properties

### `fusedWeight`

``` swift
var fusedWeight: Tensor<Scalar>
```

### `fusedBias`

``` swift
var fusedBias: Tensor<Scalar>
```

### `inputWeight`

``` swift
var inputWeight: Tensor<Scalar>
```

### `updateWeight`

``` swift
var updateWeight: Tensor<Scalar>
```

### `forgetWeight`

``` swift
var forgetWeight: Tensor<Scalar>
```

### `outputWeight`

``` swift
var outputWeight: Tensor<Scalar>
```

### `inputBias`

``` swift
var inputBias: Tensor<Scalar>
```

### `updateBias`

``` swift
var updateBias: Tensor<Scalar>
```

### `forgetBias`

``` swift
var forgetBias: Tensor<Scalar>
```

### `outputBias`

``` swift
var outputBias: Tensor<Scalar>
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
