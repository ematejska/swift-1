# withRandomNumberGeneratorForTensorFlow(\_:\_:)

Calls the given closure within a context that has everything identical to the current context
except for the given random number generator.

``` swift
public func withRandomNumberGeneratorForTensorFlow<G: RandomNumberGenerator, R>(_ randomNumberGenerator: inout G, _ body: () throws -> R) rethrows -> R
```

## Parameters

  - randomNumberGenerator: - randomNumberGenerator: A random number generator that will be set before the closure gets called and restored after the closure returns.
  - body: - body: A nullary closure. If the closure has a return value, that value is also used as the return value of the `withRandomNumberGeneratorForTensorFlow(_:_:)` function.

## Returns

The return value, if any, of the `body` closure.
