# Context

A context that stores thread-local contextual information used by deep learning APIs such as
layers.

``` swift
public struct Context
```

Use `Context.local` to retrieve the current thread-local context.

Examples:

## Initializers

### `init()`

Creates a context with default properties.

``` swift
public init()
```

## Properties

### `learningPhase`

The learning phase.

``` swift
var learningPhase: LearningPhase
```

### `randomSeed`

The random seed.

``` swift
var randomSeed: TensorFlowSeed
```

> Note: Whenever obtained, the random seed is also updated so that future stateless random TensorFlow op executions will result in non-deterministic results.

### `_randomSeed`

``` swift
var _randomSeed: TensorFlowSeed
```

### `randomNumberGenerator`

The random number generator.

``` swift
var randomNumberGenerator: AnyRandomNumberGenerator
```

### `globalTensorCount`

``` swift
var globalTensorCount: Int
```

### `local`

The current thread-local context.

``` swift
var local: Context
```

> Note: Accessing this property is thread-safe.
