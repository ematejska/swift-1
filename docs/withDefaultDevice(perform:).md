# withDefaultDevice(perform:)

Executes a closure, allowing TensorFlow to place TensorFlow operations on any device. This
should restore the default placement behavior.

``` swift
public func withDefaultDevice<R>(perform body: () throws -> R) rethrows -> R
```

## Parameters

  - body: - body: A closure whose TensorFlow operations are to be executed on the specified kind of device.
