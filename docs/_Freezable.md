# \_Freezable

A wrapper around a differentiable value with "freezable" derivatives.

``` swift
@propertyWrapper public struct _Freezable<Value: Differentiable>
```

When `isFrozen` is true, accesses to `wrappedValue` have a derivative of zero.

## Inheritance

[`Differentiable`](/Differentiable), [`EuclideanDifferentiable`](/EuclideanDifferentiable)

## Nested Type Aliases

### `TangentVector`

``` swift
public typealias TangentVector = Value.TangentVector
```

## Initializers

### `init(wrappedValue:)`

``` swift
public init(wrappedValue: Value)
```

## Properties

### `isFrozen`

``` swift
var isFrozen: Bool
```

### `_value`

``` swift
var _value: Value
```

### `projectedValue`

``` swift
var projectedValue: Self
```

### `wrappedValue`

The wrapped differentiable value.

``` swift
var wrappedValue: Value
```

## Methods

### `_vjpValue()`

``` swift
@usableFromInline func _vjpValue() -> (value: Value, pullback: (Value.TangentVector) -> TangentVector)
```

### `freeze()`

Freeze derivatives for `wrappedValue`. Accesses to `wrappedValue` will always have a
derivative of zero.

``` swift
public mutating func freeze()
```

### `unfreeze()`

Unfreeze derivatives for `wrappedValue`.

``` swift
public mutating func unfreeze()
```

### `move(along:)`

``` swift
public mutating func move(along direction: TangentVector)
```
