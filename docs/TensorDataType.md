# TensorDataType

A TensorFlow dynamic type value that can be created from types that conform to
`TensorFlowScalar`.

``` swift
public struct TensorDataType: Equatable
```

## Inheritance

`Equatable`

## Initializers

### `init(_:)`

``` swift
@usableFromInline internal init(_ cDataType: TF_DataType)
```

## Properties

### `_cDataType`

``` swift
var _cDataType: TF_DataType
```
