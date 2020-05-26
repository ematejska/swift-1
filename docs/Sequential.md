# Sequential

A layer that sequentially composes two or more other layers.

``` swift
public struct Sequential<Layer1: Module, Layer2: Layer>: Module where Layer1.Output == Layer2.Input, Layer1.TangentVector.VectorSpaceScalar == Layer2.TangentVector.VectorSpaceScalar
```

### Examples:

``` 
let inputSize = 28 * 28
let hiddenSize = 300
var classifier = Sequential {
     Dense<Float>(inputSize: inputSize, outputSize: hiddenSize, activation: relu)
     Dense<Float>(inputSize: hiddenSize, outputSize: 3, activation: identity)
 }
```

``` 
var autoencoder = Sequential {
    // The encoder.
    Dense<Float>(inputSize: 28 * 28, outputSize: 128, activation: relu)
    Dense<Float>(inputSize: 128, outputSize: 64, activation: relu)
    Dense<Float>(inputSize: 64, outputSize: 12, activation: relu)
    Dense<Float>(inputSize: 12, outputSize: 3, activation: relu)
    // The decoder.
    Dense<Float>(inputSize: 3, outputSize: 12, activation: relu)
    Dense<Float>(inputSize: 12, outputSize: 64, activation: relu)
    Dense<Float>(inputSize: 64, outputSize: 128, activation: relu)
    Dense<Float>(inputSize: 128, outputSize: imageHeight * imageWidth, activation: tanh)
}
```

## Inheritance

[`Module`](/Module), [`Layer`](/Layer)

## Initializers

### `init(_:_:)`

``` swift
public init(_ layer1: Layer1, _ layer2: Layer2)
```

### `init(layers:)`

``` swift
public init(layers: () -> Self)
```

## Properties

### `layer1`

``` swift
var layer1: Layer1
```

### `layer2`

``` swift
var layer2: Layer2
```

## Methods

### `callAsFunction(_:)`

``` swift
@differentiable(wrt: self) public func callAsFunction(_ input: Layer1.Input) -> Layer2.Output
```
