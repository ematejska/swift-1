# PythonError

An error produced by a failable Python operation.

``` swift
public enum PythonError
```

## Inheritance

`CustomStringConvertible`, `Equatable`, `Error`

## Enumeration Cases

### `exception`

A Python runtime exception, produced by calling a Python function.

``` swift
case exception(: PythonObject, traceback: PythonObject?)
```

### `invalidCall`

A failed call on a `PythonObject`.
Reasons for failure include:

``` swift
case invalidCall(: PythonObject)
```

### `invalidModule`

A module import error.

``` swift
case invalidModule(: String)
```

## Properties

### `description`

``` swift
var description: String
```
