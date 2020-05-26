# DatasetIterator

The type that allows iteration over a dataset's elements.

``` swift
@frozen public struct DatasetIterator<Element: TensorGroup>
```

## Inheritance

`IteratorProtocol`

## Initializers

### `init(_handle:)`

``` swift
@usableFromInline internal init(_handle: ResourceHandle)
```

## Properties

### `_handle`

``` swift
let _handle: ResourceHandle
```

## Methods

### `next()`

Advances to the next element and returns it, or `nil` if no next element exists.

``` swift
@inlinable public mutating func next() -> Element?
```
