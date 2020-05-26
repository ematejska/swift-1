# VariantHandle

``` swift
public struct VariantHandle
```

## Inheritance

[`TensorGroup`](/TensorGroup), [`_LazyTensorCompatible`](/_LazyTensorCompatible)

## Initializers

### `init(_owning:)`

``` swift
public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?)
```

### `init(_handles:)`

``` swift
public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle
```

### `init(owning:)`

``` swift
@usableFromInline init(owning cTensorHandle: CTensorHandle)
```

### `init(handle:)`

``` swift
@usableFromInline init(handle: _AnyTensorHandle)
```

## Properties

### `_concreteInputLazyTensor`

Returns `Self` that wraps `_concreteInputLazyTensor` of the underlying
`_AnyTensorHandle`

``` swift
var _concreteInputLazyTensor: VariantHandle
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

### `handle`

``` swift
let handle: _AnyTensorHandle
```

### `_cTensorHandle`

``` swift
var _cTensorHandle: CTensorHandle
```

### `_lazyTensor`

``` swift
var _lazyTensor: LazyTensorHandle?
```

### `_concreteLazyTensor`

``` swift
var _concreteLazyTensor: VariantHandle
```

## Methods

### `_unpackTensorHandles(into:)`

``` swift
public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?)
```
