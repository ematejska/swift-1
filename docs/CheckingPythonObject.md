# CheckingPythonObject

A `PythonObject` wrapper that enables member accesses.
Member access operations return an `Optional` result. When member access
fails, `nil` is returned.

``` swift
@dynamicMemberLookup public struct CheckingPythonObject
```

## Initializers

### `init(_:)`

``` swift
fileprivate init(_ base: PythonObject)
```

## Properties

### `base`

The underlying `PythonObject`.

``` swift
var base: PythonObject
```

### `tuple2`

Converts to a 2-tuple, if possible.

``` swift
var tuple2: (PythonObject, PythonObject)?
```

### `tuple3`

Converts to a 3-tuple, if possible.

``` swift
var tuple3: (PythonObject, PythonObject, PythonObject)?
```

### `tuple4`

Converts to a 4-tuple, if possible.

``` swift
var tuple4: (PythonObject, PythonObject, PythonObject, PythonObject)?
```
