# \_ExecutionContext

The host of any tensor computation.

``` swift
public final class _ExecutionContext
```

## Initializers

### `init()`

Initializes a new execution context by initializing available devices.

``` swift
@usableFromInline init()
```

## Properties

### `global`

Global context storing all available devices, loaded functions, etc.

``` swift
let global: _ExecutionContext
```

### `deviceNames`

List of devices available to this execution context.
Devices are represented by their names in TensorFlow notation.
See documentation for `withDevice(named:perform:)` to learn about device names.

``` swift
var deviceNames: [String]
```

### `tensorFlowConfig`

The buffer storing a serialized TensorFlow config proto.

``` swift
let tensorFlowConfig: UnsafeMutablePointer<TF_Buffer>
```

### `eagerContext`

The TFE\_Context object.

``` swift
let eagerContext: CTFEContext
```

### `status`

The status for checking TensorFlow errors.

``` swift
let status: CTFStatus
```

### `mutex`

The mutex for preventing potential concurrent access.

``` swift
var mutex: Mutex
```

### `currentDeviceName`

Returns a valid TensorFlow device name, which corresponds to the closest enclosing call to
one of the overloads of withDevice. A return value of `nil` indicates the absence of a
withDevice call on the call stack or the presence of an immediately enclosing
`withDefaultDevice(perform)` call.

``` swift
var currentDeviceName: String?
```

## Methods

### `makeOp(_:_:)`

``` swift
@usableFromInline static func makeOp(_ name: String, _ outputCount: Int) -> TFTensorOperation
```

### `withDevice(_:_:perform:)`

See documentation for the top-level `withDevice(_:_:perform)`.

``` swift
func withDevice<R>(_ kind: DeviceKind, _ index: UInt = 0, perform body: () throws -> R) rethrows -> R
```

### `withDevice(named:perform:)`

See documentation for the top-level `withDevice(named:perform)`.

``` swift
func withDevice<R>(named name: String, perform body: () throws -> R) rethrows -> R
```

### `withDefaultDevice(perform:)`

See documentation for the top-level `withDefaultDevice(perform)`.

``` swift
func withDefaultDevice<R>(perform body: () throws -> R) rethrows -> R
```

### `sync(execute:)`

Synchronously execute the body, preventing asynchronous computation from corrupting the
context data.

``` swift
private func sync<Result>(execute body: () throws -> Result) rethrows -> Result
```
