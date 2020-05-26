# ThrowingPythonObject

A `PythonObject` wrapper that enables throwing method calls.
Exceptions produced by Python functions are reflected as Swift errors and
thrown.

``` swift
public struct ThrowingPythonObject
```

> Note: It is intentional that \`ThrowingPythonObject\` does not have the \`@dynamicCallable\` attribute because the call syntax is unintuitive: \`x.throwing(arg1, arg2, ...)\`. The methods will still be named \`dynamicallyCall\` until further discussion/design.

## Initializers

### `init(_:)`

``` swift
fileprivate init(_ base: PythonObject)
```

## Properties

### `base`

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

## Methods

### `dynamicallyCall(withArguments:)`

Call `self` with the specified positional arguments.
If the call fails for some reason, `PythonError.invalidCall` is thrown.

``` swift
@discardableResult public func dynamicallyCall(withArguments args: PythonConvertible) throws -> PythonObject
```

> Precondition: \`self\` must be a Python callable.

#### Parameters

  - args: - args: Positional arguments for the Python callable.

### `dynamicallyCall(withArguments:)`

Call `self` with the specified positional arguments.
If the call fails for some reason, `PythonError.invalidCall` is thrown.

``` swift
@discardableResult public func dynamicallyCall(withArguments args: [PythonConvertible] = []) throws -> PythonObject
```

> Precondition: \`self\` must be a Python callable.

#### Parameters

  - args: - args: Positional arguments for the Python callable.

### `dynamicallyCall(withKeywordArguments:)`

Call `self` with the specified arguments.
If the call fails for some reason, `PythonError.invalidCall` is thrown.

``` swift
@discardableResult public func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, PythonConvertible> = [:]) throws -> PythonObject
```

> Precondition: \`self\` must be a Python callable.

#### Parameters

  - args: - args: Positional or keyword arguments for the Python callable.
