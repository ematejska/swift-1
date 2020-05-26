# TFETensorHandle

Class wrapping a C pointer to a TensorHandle.  This class owns the
TensorHandle and is responsible for destroying it.

``` swift
public class TFETensorHandle: _AnyTensorHandle
```

## Inheritance

[`_AnyTensorHandle`](/_AnyTensorHandle)

## Initializers

### `init(_owning:)`

``` swift
public init(_owning base: CTensorHandle)
```

## Properties

### `valueDescription`

``` swift
var valueDescription: String
```

### `_cTensorHandle`

``` swift
let _cTensorHandle: CTensorHandle
```

### `_tfeTensorHandle`

``` swift
var _tfeTensorHandle: TFETensorHandle
```

### `rank`

The number of dimensions of the underlying `Tensor`.

``` swift
var rank: Int
```

### `shape`

The shape of the underlying `Tensor`.

``` swift
var shape: TensorShape
```

## Methods

### `tfDataTypeAsString(_:)`

``` swift
static func tfDataTypeAsString(_ cDataType: TF_DataType) -> String
```

### `elementsEqual(_:)`

Returns true if the underlying tensors are equal.

``` swift
func elementsEqual(_ other: TFETensorHandle) -> Bool
```
