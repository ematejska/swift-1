# PythonInterface

An interface for Python.

``` swift
@dynamicMemberLookup public struct PythonInterface
```

`PythonInterface` allows interaction with Python. It can be used to import
modules and dynamically access Python builtin types and functions.

> Note: It is not intended for \`PythonInterface\` to be initialized directly. Instead, please use the global instance of \`PythonInterface\` called \`Python\`.

## Initializers

### `init()`

``` swift
init()
```

## Properties

### `builtins`

A dictionary of the Python builtins.

``` swift
let builtins: PythonObject
```

### `version`

``` swift
var version: PythonObject
```

### `versionInfo`

``` swift
var versionInfo: PythonObject
```

## Methods

### `attemptImport(_:)`

``` swift
public func attemptImport(_ name: String) throws -> PythonObject
```

### `` `import`(_:) ``

``` swift
public func `import`(_ name: String) -> PythonObject
```
