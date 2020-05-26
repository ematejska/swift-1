# StringTensor

`StringTensor` is a multi-dimensional array whose elements are `String`s.

``` swift
@frozen public struct StringTensor
```

## Inheritance

[`TensorGroup`](/TensorGroup), [`_LazyTensorCompatible`](/_LazyTensorCompatible)

## Initializers

### `init(handle:)`

``` swift
@inlinable public init(handle: TensorHandle<String>)
```

### `init(shape:scalars:)`

``` swift
@inlinable public init(shape: TensorShape, scalars: [String])
```

### `init(_:)`

Creates a 0-D `StringTensor` from a scalar value.

``` swift
@inlinable public init(_ value: String)
```

### `init(_:)`

Creates a 1-D `StringTensor` in from contiguous scalars.

``` swift
@inlinable public init(_ scalars: [String])
```

### `init(_owning:)`

``` swift
public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?)
```

### `init(_handles:)`

``` swift
public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle
```

## Properties

### `_concreteInputLazyTensor`

Returns `Self` that wraps `_concreteInputLazyTensor` of the underlying
`_AnyTensorHandle`

``` swift
var _concreteInputLazyTensor: StringTensor
```

### `handle`

The underlying `TensorHandle`.

``` swift
let handle: TensorHandle<String>
```

> Note: \`handle\` is public to allow user defined ops, but should not normally be used otherwise.

### `array`

``` swift
var array: ShapedArray<String>
```

### `scalars`

``` swift
var scalars: [String]
```

### `_unknownShapeList`

``` swift
var _unknownShapeList: [TensorShape?]
```

### `_typeList`

``` swift
var _typeList: [TensorDataType]
```

### `_tensorHandles`

``` swift
var _tensorHandles: [_AnyTensorHandle]
```

### `_lazyTensor`

``` swift
var _lazyTensor: LazyTensorHandle?
```

### `_concreteLazyTensor`

``` swift
var _concreteLazyTensor: StringTensor
```

## Methods

### `elementsEqual(_:)`

Computes `self == other` element-wise.

``` swift
@inlinable public func elementsEqual(_ other: StringTensor) -> Tensor<Bool>
```

> Note: \`elementsEqual\` supports broadcasting.

### `_unpackTensorHandles(into:)`

``` swift
public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?)
```
