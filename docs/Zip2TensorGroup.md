# Zip2TensorGroup

A 2-tuple-like struct that conforms to TensorGroup that represents a tuple of 2 types conforming
to `TensorGroup`.

``` swift
@frozen public struct Zip2TensorGroup<T: TensorGroup, U: TensorGroup>: TensorGroup
```

## Inheritance

[`TensorGroup`](/TensorGroup)

## Initializers

### `init(_:_:)`

``` swift
public init(_ first: T, _ second: U)
```

### `init(_handles:)`

``` swift
public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle
```

## Properties

### `first`

``` swift
var first: T
```

### `second`

``` swift
var second: U
```

### `_tensorHandles`

``` swift
var _tensorHandles: [_AnyTensorHandle]
```
