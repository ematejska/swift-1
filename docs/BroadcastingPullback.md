# BroadcastingPullback

A pullback function that performs the transpose of broadcasting two `Tensors`.

``` swift
public struct BroadcastingPullback
```

## Initializers

### `init(_:_:)`

Constructs the pullback from broadcasting `lhs` and `rhs`.

``` swift
public init<T: TensorFlowFloatingPoint, U: TensorFlowFloatingPoint>(_ lhs: Tensor<T>, _ rhs: Tensor<U>)
```

## Properties

### `lhsShape`

``` swift
let lhsShape: [Int64]
```

### `rhsShape`

``` swift
let rhsShape: [Int64]
```

## Methods

### `callAsFunction(_:_:)`

``` swift
public func callAsFunction<T: TensorFlowFloatingPoint, U: TensorFlowFloatingPoint>(_ lhsGrad: Tensor<T>, _ rhsGrad: Tensor<U>) -> (Tensor<T>, Tensor<U>)
```

### `computeReductionAxes(_:_:)`

Compute the axis needed to sum along in order to map back from the
broadcasted shape to the individual argument shapes.

``` swift
@usableFromInline static func computeReductionAxes(_ lhsShape: [Int64], _ rhsShape: [Int64]) -> (lhsAxes: [Int64], rhsAxes: [Int64])
```
