# Embedding

An embedding layer.

``` swift
public struct Embedding<Scalar: TensorFlowFloatingPoint>: Module
```

`Embedding` is effectively a lookup table that maps indices from a fixed vocabulary to fixed-size
(dense) vector representations, e.g. `[[0], [3]] -> [[0.25, 0.1], [0.6, -0.2]]`.

## Inheritance

[`Module`](/Module)

## Initializers

### `init(vocabularySize:embeddingSize:embeddingsInitializer:)`

Creates an `Embedding` layer with randomly initialized embeddings of shape
`(vocabularySize, embeddingSize)` so that each vocabulary index is given a vector
representation.

``` swift
public init(vocabularySize: Int, embeddingSize: Int, embeddingsInitializer: ParameterInitializer<Scalar> = { Tensor(randomUniform: $0) })
```

#### Parameters

  - vocabularySize: - vocabularySize: The number of distinct indices (words) in the vocabulary. This number should be the largest integer index plus one.
  - embeddingSize: - embeddingSize: The number of entries in a single embedding vector representation.
  - embeddingsInitializer: - embeddingsInitializer: Initializer to use for the embedding parameters.

### `init(embeddings:)`

Creates an `Embedding` layer from the provided embeddings. Useful for introducing
pretrained embeddings into a model.

``` swift
public init(embeddings: Tensor<Scalar>)
```

#### Parameters

  - embeddings: - embeddings: The pretrained embeddings table.

## Properties

### `embeddings`

A learnable lookup table that maps vocabulary indices to their dense vector representations.

``` swift
var embeddings: Tensor<Scalar>
```

## Methods

### `callAsFunction(_:)`

Returns an output by replacing each index in the input with corresponding dense vector representation.

``` swift
@differentiable(wrt: self) public func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Scalar>
```

#### Returns

The tensor created by replacing input indices with their vector representations.
