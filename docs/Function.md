# Function

A layer that encloses a custom differentiable function.

``` swift
public struct Function<Input: Differentiable, Output: Differentiable>: ParameterlessLayer
```

## Inheritance

[`ParameterlessLayer`](/ParameterlessLayer)

## Nested Type Aliases

### `Body`

``` swift
public typealias Body = @differentiable (Input) -> Output
```

## Initializers

### `init(_:)`

``` swift
public init(_ body: @escaping Body)
```

## Properties

### `body`

``` swift
let body: Body
```

## Methods

### `callAsFunction(_:)`

``` swift
@differentiable public func callAsFunction(_ input: Input) -> Output
```
