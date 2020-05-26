# PythonObject

`PythonObject` represents an object in Python and supports dynamic member
lookup. Any member access like `object.foo` will dynamically request the
Python runtime for a member with the specified name in this object.

``` swift
@dynamicCallable @dynamicMemberLookup public struct PythonObject
```

`PythonObject` is passed to and returned from all Python function calls and
member references. It supports standard Python arithmetic and comparison
operators.

Internally, `PythonObject` is implemented as a reference-counted pointer to
a Python C API `PyObject`.

## Inheritance

`Comparable`, [`ConvertibleFromPython`](/ConvertibleFromPython), `CustomPlaygroundDisplayConvertible`, `CustomReflectable`, `CustomStringConvertible`, `Equatable`, `ExpressibleByArrayLiteral`, `ExpressibleByBooleanLiteral`, `ExpressibleByDictionaryLiteral`, `ExpressibleByFloatLiteral`, `ExpressibleByIntegerLiteral`, `ExpressibleByStringLiteral`, `Hashable`, `MutableCollection`, [`PythonConvertible`](/PythonConvertible), `Sequence`, `SignedNumeric`, `Strideable`

## Nested Type Aliases

### `Magnitude`

``` swift
public typealias Magnitude = PythonObject
```

### `Stride`

``` swift
public typealias Stride = PythonObject
```

### `Index`

``` swift
public typealias Index = PythonObject
```

### `Element`

``` swift
public typealias Element = PythonObject
```

### `Key`

``` swift
public typealias Key = PythonObject
```

### `Value`

``` swift
public typealias Value = PythonObject
```

## Initializers

### `init(_:)`

``` swift
@usableFromInline init(_ pointer: PyReference)
```

### `init(_:)`

Creates a new instance and a new reference.

``` swift
init(_ pointer: OwnedPyObjectPointer)
```

### `init(consuming:)`

Creates a new instance consuming the specified `PyObject` pointer.

``` swift
init(consuming pointer: PyObjectPointer)
```

### `init(_:)`

Creates a new instance from a `PythonConvertible` value.

``` swift
init<T: PythonConvertible>(_ object: T)
```

### `init(_:)`

``` swift
public init(_ object: PythonObject)
```

### `init(tupleOf:)`

``` swift
init(tupleOf elements: PythonConvertible)
```

### `init(tupleContentsOf:)`

``` swift
init<T: Collection>(tupleContentsOf elements: T) where T.Element == PythonConvertible
```

### `init(tupleContentsOf:)`

``` swift
init<T: Collection>(tupleContentsOf elements: T) where T.Element: PythonConvertible
```

### `init(exactly:)`

``` swift
public init<T: BinaryInteger>(exactly value: T)
```

### `init(booleanLiteral:)`

``` swift
public init(booleanLiteral value: Bool)
```

### `init(integerLiteral:)`

``` swift
public init(integerLiteral value: Int)
```

### `init(floatLiteral:)`

``` swift
public init(floatLiteral value: Double)
```

### `init(stringLiteral:)`

``` swift
public init(stringLiteral value: String)
```

### `init(arrayLiteral:)`

``` swift
public init(arrayLiteral elements: PythonObject)
```

### `init(dictionaryLiteral:)`

``` swift
public init(dictionaryLiteral elements: (PythonObject, PythonObject))
```

## Properties

### `reference`

The underlying `PyReference`.

``` swift
var reference: PyReference
```

### `borrowedPyObject`

``` swift
var borrowedPyObject: PyObjectPointer
```

### `ownedPyObject`

``` swift
var ownedPyObject: OwnedPyObjectPointer
```

### `description`

A textual description of this `PythonObject`, produced by `Python.str`.

``` swift
var description: String
```

### `playgroundDescription`

``` swift
var playgroundDescription: Any
```

### `customMirror`

``` swift
var customMirror: Mirror
```

### `pythonObject`

``` swift
var pythonObject: PythonObject
```

### `throwing`

Returns a callable version of this `PythonObject`. When called, the result
throws a Swift error if the underlying Python function throws a Python
exception.

``` swift
var throwing: ThrowingPythonObject
```

### `checking`

Returns a `PythonObject` wrapper capable of member accesses.

``` swift
var checking: CheckingPythonObject
```

### `tuple2`

Converts to a 2-tuple.

``` swift
var tuple2: (PythonObject, PythonObject)
```

### `tuple3`

Converts to a 3-tuple.

``` swift
var tuple3: (PythonObject, PythonObject, PythonObject)
```

### `tuple4`

Converts to a 4-tuple.

``` swift
var tuple4: (PythonObject, PythonObject, PythonObject, PythonObject)
```

### `magnitude`

``` swift
var magnitude: PythonObject
```

### `startIndex`

``` swift
var startIndex: Index
```

### `endIndex`

``` swift
var endIndex: Index
```

## Methods

### `dynamicallyCall(withArguments:)`

Call `self` with the specified positional arguments.

``` swift
@discardableResult func dynamicallyCall(withArguments args: [PythonConvertible] = []) -> PythonObject
```

> Precondition: \`self\` must be a Python callable.

#### Parameters

  - args: - args: Positional arguments for the Python callable.

### `dynamicallyCall(withKeywordArguments:)`

Call `self` with the specified arguments.

``` swift
@discardableResult func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, PythonConvertible> = [:]) -> PythonObject
```

> Precondition: \`self\` must be a Python callable.

#### Parameters

  - args: - args: Positional or keyword arguments for the Python callable.

### `converted(withError:by:)`

``` swift
func converted<T: Equatable>(withError errorValue: T, by converter: (OwnedPyObjectPointer) -> T) -> T?
```

### `+(lhs:rhs:)`

``` swift
static func +(lhs: PythonObject, rhs: PythonObject) -> PythonObject
```

### `-(lhs:rhs:)`

``` swift
static func -(lhs: PythonObject, rhs: PythonObject) -> PythonObject
```

### `*(lhs:rhs:)`

``` swift
static func *(lhs: PythonObject, rhs: PythonObject) -> PythonObject
```

### `/(lhs:rhs:)`

``` swift
static func /(lhs: PythonObject, rhs: PythonObject) -> PythonObject
```

### `+=(lhs:rhs:)`

``` swift
static func +=(lhs: inout PythonObject, rhs: PythonObject)
```

### `-=(lhs:rhs:)`

``` swift
static func -=(lhs: inout PythonObject, rhs: PythonObject)
```

### `*=(lhs:rhs:)`

``` swift
static func *=(lhs: inout PythonObject, rhs: PythonObject)
```

### `/=(lhs:rhs:)`

``` swift
static func /=(lhs: inout PythonObject, rhs: PythonObject)
```

### `distance(to:)`

``` swift
public func distance(to other: PythonObject) -> Stride
```

### `advanced(by:)`

``` swift
public func advanced(by stride: Stride) -> PythonObject
```

### `compared(to:byOp:)`

``` swift
private func compared(to other: PythonObject, byOp: Int32) -> Bool
```

### `==(lhs:rhs:)`

``` swift
public static func ==(lhs: PythonObject, rhs: PythonObject) -> Bool
```

### `!=(lhs:rhs:)`

``` swift
public static func !=(lhs: PythonObject, rhs: PythonObject) -> Bool
```

### `<(lhs:rhs:)`

``` swift
public static func <(lhs: PythonObject, rhs: PythonObject) -> Bool
```

### `<=(lhs:rhs:)`

``` swift
public static func <=(lhs: PythonObject, rhs: PythonObject) -> Bool
```

### `>(lhs:rhs:)`

``` swift
public static func >(lhs: PythonObject, rhs: PythonObject) -> Bool
```

### `>=(lhs:rhs:)`

``` swift
public static func >=(lhs: PythonObject, rhs: PythonObject) -> Bool
```

### `hash(into:)`

``` swift
public func hash(into hasher: inout Hasher)
```

### `index(after:)`

``` swift
public func index(after i: Index) -> Index
```

### `makeIterator()`

``` swift
public func makeIterator() -> Iterator
```
