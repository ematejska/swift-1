# Dataset

Represents a potentially large set of elements.

``` swift
@frozen public struct Dataset<Element: TensorGroup>
```

A `Dataset` can be used to represent an input pipeline as a collection of element tensors.

## Inheritance

`Sequence`

## Nested Type Aliases

### `Iterator`

``` swift
public typealias Iterator = DatasetIterator<Element>
```

## Initializers

### `init(_handle:)`

``` swift
@inlinable public init(_handle: VariantHandle)
```

### `init(randomSeed:)`

``` swift
@inlinable public init(randomSeed: Int64)
```

### `init(elements:)`

Creates a dataset from a batch of elements as a tensor.

``` swift
@inlinable public init(elements: Element)
```

## Properties

### `_handle`

``` swift
let _handle: VariantHandle
```

## Methods

### `makeIterator()`

Returns an iterator over the elements of this dataset.

``` swift
@inlinable public func makeIterator() -> DatasetIterator<Element>
```

### `map(_:)`

``` swift
@inlinable public func map<ResultElement: TensorGroup>(_ transform: (Element) -> ResultElement) -> Dataset<ResultElement>
```

### `map(parallelCallCount:_:)`

``` swift
@inlinable public func map<ResultElement: TensorGroup>(parallelCallCount: Int, _ transform: (Element) -> ResultElement) -> Dataset<ResultElement>
```

### `filter(_:)`

``` swift
@inlinable public func filter(_ isIncluded: (Element) -> Tensor<Bool>) -> Dataset
```

### `prefetched(count:)`

``` swift
@inlinable public func prefetched(count: Int) -> Dataset
```

### `shuffled(sampleCount:randomSeed:reshuffleForEachIterator:)`

``` swift
@inlinable public func shuffled(sampleCount: Int, randomSeed: Int64, reshuffleForEachIterator: Bool = true) -> Dataset
```

### `batched(_:)`

``` swift
@inlinable public func batched(_ batchSize: Int) -> Dataset
```

### `repeated(count:)`

``` swift
@inlinable public func repeated(count: Int? = nil) -> Dataset
```
