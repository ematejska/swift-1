# \_TensorElementLiteral

Represents a literal element for conversion to a `Tensor`.

``` swift
@frozen public struct _TensorElementLiteral<Scalar> where Scalar: TensorFlowScalar
```

> Note: Do not ever use this API directly. This is implicitly created during the conversion from an array literal to a \`Tensor\`, and is purely for implementation purposes.

## Inheritance

`ExpressibleByArrayLiteral`, `ExpressibleByBooleanLiteral`, `ExpressibleByFloatLiteral`, `ExpressibleByIntegerLiteral`, [`TensorGroup`](/TensorGroup)

## Nested Type Aliases

### `ArrayLiteralElement`

``` swift
public typealias ArrayLiteralElement = _TensorElementLiteral<Scalar>
```

## Initializers

### `init(arrayLiteral:)`

``` swift
@inlinable public init(arrayLiteral elements: _TensorElementLiteral<Scalar>)
```

### `init(_owning:)`

``` swift
public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?)
```

### `init(_handles:)`

``` swift
public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle
```

## Properties

### `tensor`

``` swift
let tensor: Tensor<Scalar>
```

### `_unknownShapeList`

``` swift
var _unknownShapeList: [TensorShape?]
```

### `_typeList`

``` swift
var _typeList: [TensorDataType]
```

### `_tensorHandles`

``` swift
var _tensorHandles: [_AnyTensorHandle]
```

## Methods

### `_unpackTensorHandles(into:)`

``` swift
public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?)
```
