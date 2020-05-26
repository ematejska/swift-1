# TensorGroup

A protocol representing types that can be mapped to and from `Array<CTensorHandle>`.

``` swift
public protocol TensorGroup: TensorArrayProtocol
```

When a `TensorGroup` is used as an argument to a tensor operation, it is passed as an argument
list whose elements are the tensor fields of the type.

When a `TensorGroup` is returned as a result of a tensor operation, it is initialized with its
tensor fields set to the tensor operation's tensor results.

## Inheritance

[`TensorArrayProtocol`](/TensorArrayProtocol)

## Requirements

## \_typeList

The types of the tensor stored properties in this type.

``` swift
var _typeList: [TensorDataType]
```

## init(\_owning:)

Initializes a value of this type, taking ownership of the `_tensorHandleCount` tensors
starting at address `tensorHandles`.

``` swift
init(_owning tensorHandles: UnsafePointer<CTensorHandle>?)
```
