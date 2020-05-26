# LayerBuilder

``` swift
@_functionBuilder public struct LayerBuilder
```

## Methods

### `buildBlock(_:_:)`

``` swift
public static func buildBlock<L1: Module, L2: Layer>(_ l1: L1, _ l2: L2) -> Sequential<L1, L2> where L1.Output == L2.Input
```

### `buildBlock(_:_:_:)`

``` swift
public static func buildBlock<L1: Module, L2: Layer, L3: Layer>(_ l1: L1, _ l2: L2, _ l3: L3) -> Sequential<L1, Sequential<L2, L3>> where L1.Output == L2.Input, L2.Output == L3.Input, L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar, L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar
```

### `buildBlock(_:_:_:_:)`

``` swift
public static func buildBlock<L1: Module, L2: Layer, L3: Layer, L4: Layer>(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4) -> Sequential<L1, Sequential<L2, Sequential<L3, L4>>> where L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar, L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar, L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar
```

### `buildBlock(_:_:_:_:_:)`

``` swift
public static func buildBlock<L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer>(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5) -> Sequential<L1, Sequential<L2, Sequential<L3, Sequential<L4, L5>>>> where L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input, L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar, L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar, L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar, L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar
```

### `buildBlock(_:_:_:_:_:_:)`

``` swift
public static func buildBlock<L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer>(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6) -> Sequential<L1, Sequential<L2, Sequential<L3, Sequential<L4, Sequential<L5, L6>>>>> where L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input, L5.Output == L6.Input, L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar, L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar, L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar, L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar, L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar
```

### `buildBlock(_:_:_:_:_:_:_:)`

``` swift
public static func buildBlock<L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer, L7: Layer>(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6, _ l7: L7) -> Sequential<
      L1, Sequential<L2, Sequential<L3, Sequential<L4, Sequential<L5, Sequential<L6, L7>>>>>
    > where L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input, L5.Output == L6.Input, L6.Output == L7.Input, L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar, L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar, L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar, L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar, L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar, L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar
```

### `buildBlock(_:_:_:_:_:_:_:_:)`

``` swift
public static func buildBlock<L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer, L7: Layer, L8: Layer>(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6, _ l7: L7, _ l8: L8) -> Sequential<
      L1,
      Sequential<
        L2,
        Sequential<L3, Sequential<L4, Sequential<L5, Sequential<L6, Sequential<L7, L8>>>>>
      >
    > where L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input, L5.Output == L6.Input, L6.Output == L7.Input, L7.Output == L8.Input, L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar, L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar, L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar, L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar, L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar, L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar, L7.TangentVector.VectorSpaceScalar == L8.TangentVector.VectorSpaceScalar
```

### `buildBlock(_:_:_:_:_:_:_:_:_:)`

``` swift
public static func buildBlock<L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer, L7: Layer, L8: Layer, L9: Layer>(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6, _ l7: L7, _ l8: L8, _ l9: L9) -> Sequential<
      L1,
      Sequential<
        L2,
        Sequential<
          L3,
          Sequential<
            L4, Sequential<L5, Sequential<L6, Sequential<L7, Sequential<L8, L9>>>>
          >
        >
      >
    > where L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input, L5.Output == L6.Input, L6.Output == L7.Input, L7.Output == L8.Input, L8.Output == L9.Input, L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar, L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar, L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar, L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar, L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar, L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar, L7.TangentVector.VectorSpaceScalar == L8.TangentVector.VectorSpaceScalar, L8.TangentVector.VectorSpaceScalar == L9.TangentVector.VectorSpaceScalar
```

### `buildBlock(_:_:_:_:_:_:_:_:_:_:)`

``` swift
public static func buildBlock<L1: Module, L2: Layer, L3: Layer, L4: Layer, L5: Layer, L6: Layer, L7: Layer, L8: Layer, L9: Layer, L10: Layer>(_ l1: L1, _ l2: L2, _ l3: L3, _ l4: L4, _ l5: L5, _ l6: L6, _ l7: L7, _ l8: L8, _ l9: L9, _ l10: L10) -> Sequential<
      L1,
      Sequential<
        L2,
        Sequential<
          L3,
          Sequential<
            L4,
            Sequential<
              L5, Sequential<L6, Sequential<L7, Sequential<L8, Sequential<L9, L10>>>>
            >
          >
        >
      >
    > where L1.Output == L2.Input, L2.Output == L3.Input, L3.Output == L4.Input, L4.Output == L5.Input, L5.Output == L6.Input, L6.Output == L7.Input, L7.Output == L8.Input, L8.Output == L9.Input, L9.Output == L10.Input, L1.TangentVector.VectorSpaceScalar == L2.TangentVector.VectorSpaceScalar, L2.TangentVector.VectorSpaceScalar == L3.TangentVector.VectorSpaceScalar, L3.TangentVector.VectorSpaceScalar == L4.TangentVector.VectorSpaceScalar, L4.TangentVector.VectorSpaceScalar == L5.TangentVector.VectorSpaceScalar, L5.TangentVector.VectorSpaceScalar == L6.TangentVector.VectorSpaceScalar, L6.TangentVector.VectorSpaceScalar == L7.TangentVector.VectorSpaceScalar, L7.TangentVector.VectorSpaceScalar == L8.TangentVector.VectorSpaceScalar, L8.TangentVector.VectorSpaceScalar == L9.TangentVector.VectorSpaceScalar, L9.TangentVector.VectorSpaceScalar == L10.TangentVector.VectorSpaceScalar
```
