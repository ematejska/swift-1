# CopyableToDevice

A type whose nested properties and elements can be copied to a `Device`.

``` swift
public protocol CopyableToDevice: _CopyableToDevice
```

## Inheritance

[`_CopyableToDevice`](/_CopyableToDevice)

## Requirements

## init(copying:to:)

Creates a copy of `other` on the given `Device`.

``` swift
init(copying other: Self, to device: Device)
```

All cross-device references are moved to the given `Device`.
