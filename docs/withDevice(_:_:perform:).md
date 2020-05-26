# withDevice(\_:\_:perform:)

Executes a closure, making TensorFlow operations run on a specific kind of device.

``` swift
public func withDevice<R>(_ kind: DeviceKind, _ index: UInt = 0, perform body: () throws -> R) rethrows -> R
```

## Parameters

  - kind: - kind: A kind of device to run TensorFlow operations on.
  - index: - index: The device to run the ops on.
  - body: - body: A closure whose TensorFlow operations are to be executed on the specified kind of device.
