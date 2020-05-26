# ConvertibleFromPython

A type that can be initialized from a `PythonObject`.

``` swift
public protocol ConvertibleFromPython
```

## Requirements

## init?(\_:)

Creates a new instance from the given `PythonObject`, if possible.

``` swift
init?(_ object: PythonObject)
```

> Note: Conversion may fail if the given \`PythonObject\` instance is incompatible (e.g. a Python \`string\` object cannot be converted into an \`Int\`).
