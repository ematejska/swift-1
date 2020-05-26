# TensorHandle

`TensorHandle` is the type used by ops. It includes a `Scalar` type, which
compiler internals can use to determine the datatypes of parameters when
they are extracted into a tensor program.

``` swift
public struct TensorHandle<Scalar> where Scalar: _TensorFlowDataTypeCompatible
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

### `init(_owning:)`

``` swift
public init(_owning cTensorHandle: CTensorHandle)
```

### `init(handle:)`

``` swift
public init(handle: _AnyTensorHandle)
```

### `init(copyingFromCTensor:)`

``` swift
@usableFromInline init(copyingFromCTensor cTensor: CTensor)
```

### `init(shape:byteCount:bufferInitializer:)`

Create a `TensorHandle` with a closure that initializes the underlying buffer.

``` swift
@inlinable init(shape: [Int], byteCount: Int, bufferInitializer: (UnsafeMutableRawPointer) -> Void)
```

Users initializing `TensorHandle`s with non-`String` scalars should use the
`init(shape:scalarsInitializer:)` initializer instead of this one. It enforces additional
constraints on the buffer that hold for all non-`String` scalars.

`bufferInitializer` receives a buffer with exactly `byteCount` bytes of capacity.
`bufferInitializer` must initialize the entire buffer.

## Properties

### `_concreteInputLazyTensor`

Returns `Self` that wraps `_concreteInputLazyTensor` of the underlying
`_AnyTensorHandle`

``` swift
var _concreteInputLazyTensor: TensorHandle
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

### `rank`

The number of dimensions of the `Tensor`.

``` swift
var rank: Int
```

### `shape`

The shape of the `Tensor`.

``` swift
var shape: TensorShape
```

### `_lazyTensor`

``` swift
var _lazyTensor: LazyTensorHandle?
```

### `_concreteLazyTensor`

``` swift
var _concreteLazyTensor: TensorHandle
```

## Methods

### `_unpackTensorHandles(into:)`

``` swift
public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?)
```

### `makeHostCopy()`

Create a `ShapedArray` with contents of the underlying `TensorHandle`. If the `TensorHandle`
is on the accelerator, it will be copied to the host.

``` swift
@usableFromInline @inline(never) func makeHostCopy() -> ShapedArray<Scalar>
```

#### Returns

A `ShapedArray`.

### `makeCopy()`

``` swift
func makeCopy() -> TFETensorHandle
```
