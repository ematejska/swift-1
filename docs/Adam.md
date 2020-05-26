# Adam

Adam optimizer.

``` swift
public class Adam<Model: Differentiable>: Optimizer where Model.TangentVector: VectorProtocol & PointwiseMultiplicative
    & ElementaryFunctions & KeyPathIterable, Model.TangentVector.VectorSpaceScalar == Float
```

Implements the Adam optimization algorithm. Adam is a stochastic gradient descent method that
computes individual adaptive learning rates for different parameters from estimates of first-
and second-order moments of the gradients.

Reference: ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980v8)
(Kingma and Ba, 2014).

### Examples:

``` 
...
// Instantiate an agent's policy - approximated by the neural network (`net`) after defining it 
in advance.
var net = Net(observationSize: Int(observationSize), hiddenSize: hiddenSize, actionCount: actionCount)
// Define the Adam optimizer for the network with a learning rate set to 0.01.
let optimizer = Adam(for: net, learningRate: 0.01)
...
// Begin training the agent (over a certain number of episodes).
while true {
...
    // Implementing the gradient descent with the Adam optimizer:
    // Define the gradients (use withLearningPhase to call a closure under a learning phase).
    let gradients = withLearningPhase(.training) {
        TensorFlow.gradient(at: net) { net -> Tensor<Float> in
            // Return a softmax (loss) function
            return loss = softmaxCrossEntropy(logits: net(input), probabilities: target)
        }
    }
    // Update the differentiable variables of the network (`net`) along the gradients with the Adam 
optimizer.
    optimizer.update(&net, along: gradients)
    ...
    }
}
```

``` 
...
// Instantiate the generator and the discriminator networks after defining them.
var generator = Generator()
var discriminator = Discriminator()
// Define the Adam optimizers for each network with a learning rate set to 2e-4 and beta1 - to 0.5.
let adamOptimizerG = Adam(for: generator, learningRate: 2e-4, beta1: 0.5)
let adamOptimizerD = Adam(for: discriminator, learningRate: 2e-4, beta1: 0.5)
...
Start the training loop over a certain number of epochs (`epochCount`).
for epoch in 1...epochCount {
    // Start the training phase.
    ...
    for batch in trainingShuffled.batched(batchSize) {
        // Implementing the gradient descent with the Adam optimizer:
        // 1) Update the generator.
        ...
        let 𝛁generator = TensorFlow.gradient(at: generator) { generator -> Tensor<Float> in
            ...
            return loss
            }
        // Update the differentiable variables of the generator along the gradients (`𝛁generator`) 
        // with the Adam optimizer.
        adamOptimizerG.update(&generator, along: 𝛁generator)

        // 2) Update the discriminator.
        ...
        let 𝛁discriminator = TensorFlow.gradient(at: discriminator) { discriminator -> Tensor<Float> in
            ...
            return loss
        }
        // Update the differentiable variables of the discriminator along the gradients (`𝛁discriminator`) 
        // with the Adam optimizer.
        adamOptimizerD.update(&discriminator, along: 𝛁discriminator)
        }
}       
```

## Parameters

  - learningRate: - learningRate: A Float. The learning rate (default value: 1e-3).
  - beta1: - beta1: A Float. The exponentian decay rate for the 1st moment estimates (default value: 0.9).
  - beta2: - beta2: A Float. The exponentian decay rate for the 2nd moment estimates (default value: 0.999).
  - epsilon: - epsilon: A Float. A small scalar added to the denominator to improve numerical stability (default value: 1e-8).
  - decay: - decay: A Float. The learning rate decay (default value: 0).

## Inheritance

[`Optimizer`](/Optimizer)

## Nested Type Aliases

### `Model`

``` swift
public typealias Model = Model
```

## Initializers

### `init(for:learningRate:beta1:beta2:epsilon:decay:)`

``` swift
public init(for model: __shared Model, learningRate: Float = 1e-3, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8, decay: Float = 0)
```

### `init(copying:to:)`

``` swift
public required init(copying other: Adam, to device: Device)
```

## Properties

### `learningRate`

The learning rate.

``` swift
var learningRate: Float
```

### `beta1`

A coefficient used to calculate the first moments of the gradients.

``` swift
var beta1: Float
```

### `beta2`

A coefficient used to calculate the second moments of the gradients.

``` swift
var beta2: Float
```

### `epsilon`

A small scalar added to the denominator to improve numerical stability.

``` swift
var epsilon: Float
```

### `decay`

The learning rate decay.

``` swift
var decay: Float
```

### `step`

The current step.

``` swift
var step: Int
```

### `firstMoments`

The first moments of the weights.

``` swift
var firstMoments: Model.TangentVector
```

### `secondMoments`

The second moments of the weights.

``` swift
var secondMoments: Model.TangentVector
```

## Methods

### `update(_:along:)`

``` swift
public func update(_ model: inout Model, along direction: Model.TangentVector)
```
