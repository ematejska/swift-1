# \_CopyableToDevice

A type whose nested properties and elements can be copied to `Device`s.

``` swift
public protocol _CopyableToDevice
```

> Note: Do not ever use this API directly. This is an implementation detail to support \`KeyPathIterable.move(to:)\` and \`KeyPathIterable.init(copying:to:)\`.

> Note: this workaround is necessary because \`CopyableToDevice\` is a protocol with \`Self\` requirements, so \`x as? CopyableToDevice\` does not work.

## Requirements

## \_move(\_:\_:to:)

``` swift
static func _move<Root>(_ root: inout Root, _ rootKeyPath: PartialKeyPath<Root>, to: Device)
```
