# withDevice(named:perform:)

Executes a closure, making TensorFlow operations run on a device with
a specific name.

``` swift
public func withDevice<R>(named name: String, perform body: () throws -> R) rethrows -> R
```

Some examples of device names:

## Parameters

  - name: - name: Device name.
  - body: - body: A closure whose TensorFlow operations are to be executed on the specified kind of device.
