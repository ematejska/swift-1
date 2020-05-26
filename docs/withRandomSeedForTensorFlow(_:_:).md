# withRandomSeedForTensorFlow(\_:\_:)

Calls the given closure within a context that has everything identical to the current context
except for the given random seed.

``` swift
public func withRandomSeedForTensorFlow<R>(_ randomSeed: TensorFlowSeed, _ body: () throws -> R) rethrows -> R
```

## Parameters

  - randomSeed: - randomSeed: A random seed that will be set before the closure gets called and restored after the closure returns.
  - body: - body: A nullary closure. If the closure has a return value, that value is also used as the return value of the `withRandomSeedForTensorFlow(_:_:)` function.

## Returns

The return value, if any, of the `body` closure.
