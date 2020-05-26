# NumpyScalarCompatible

A type that is bitwise compatible with one or more NumPy scalar types.

``` swift
public protocol NumpyScalarCompatible
```

## Requirements

## numpyScalarTypes

The NumPy scalar types that this type is bitwise compatible with. Must
be nonempty.

``` swift
var numpyScalarTypes: [PythonObject]
```

## ctype

The Python `ctypes` scalar type corresponding to this type.

``` swift
var ctype: PythonObject
```
