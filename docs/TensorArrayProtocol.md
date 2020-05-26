# TensorArrayProtocol

A protocol representing types that can be mapped to `Array<CTensorHandle>`.

``` swift
public protocol TensorArrayProtocol
```

This protocol is defined separately from `TensorGroup` in order for the number of tensors to be
determined at runtime. For example, `[Tensor<Float>]` may have an unknown number of elements at
compile time.

This protocol can be derived automatically for structs whose stored properties all conform to
the `TensorGroup` protocol. It cannot be derived automatically for structs whose properties all
conform to `TensorArrayProtocol` due to the constructor requirement (i.e., in such cases it
would be impossible to know how to break down `count` among the stored properties).

## Requirements

## \_unpackTensorHandles(into:)

Writes the tensor handles to `address`, which must be allocated with enough capacity to hold
`_tensorHandleCount` handles. The tensor handles written to `address` are borrowed: this
container still owns them.

``` swift
func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?)
```

## \_tensorHandleCount

``` swift
var _tensorHandleCount: Int32
```

## \_typeList

``` swift
var _typeList: [TensorDataType]
```

## \_tensorHandles

``` swift
var _tensorHandles: [_AnyTensorHandle]
```

## init(\_owning:count:)

``` swift
init(_owning tensorHandles: UnsafePointer<CTensorHandle>?, count: Int)
```

## init(\_handles:)

``` swift
init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle
```
