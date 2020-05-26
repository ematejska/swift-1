# TFTensorOperation

``` swift
public protocol TFTensorOperation: TensorOperation
```

## Inheritance

[`TensorOperation`](/TensorOperation)

## Requirements

## addInput(\_:)

``` swift
func addInput<Scalar: TensorFlowScalar>(_ input: Tensor<Scalar>)
```

## addInput(\_:)

``` swift
func addInput(_ input: StringTensor)
```

## addInput(\_:)

``` swift
func addInput(_ input: VariantHandle)
```

## addInput(\_:)

``` swift
func addInput(_ input: ResourceHandle)
```

## addInputList(\_:)

``` swift
func addInputList<T: TensorArrayProtocol>(_ input: T)
```

## updateAttribute(\_:\_:)

``` swift
func updateAttribute(_ name: String, _ value: TensorDataType)
```

## updateAttribute(\_:\_:)

``` swift
func updateAttribute(_ name: String, _ value: TensorShape)
```

## updateAttribute(\_:\_:)

``` swift
func updateAttribute(_ name: String, _ value: TensorShape?)
```

## updateAttribute(\_:\_:)

``` swift
func updateAttribute(_ name: String, _ value: [TensorDataType])
```

## updateAttribute(\_:\_:)

``` swift
func updateAttribute(_ name: String, _ value: [TensorShape])
```

## updateAttribute(\_:\_:)

``` swift
func updateAttribute(_ name: String, _ value: [TensorShape?])
```

## updateAttribute(\_:\_:)

``` swift
func updateAttribute<In: TensorGroup, Out: TensorGroup>(_ name: String, _ value: (In) -> Out)
```

## updateAttribute(\_:\_:)

``` swift
func updateAttribute(_ name: String, _ value: _TensorFunctionPointer)
```

## execute()

``` swift
func execute()
```

## execute(\_:)

``` swift
func execute<T0: TensorArrayProtocol>(_ count0: Int) -> (T0)
```

## execute(\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol>(_ count0: Int, _ count1: Int) -> (T0, T1)
```

## execute(\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int) -> (T0, T1, T2)
```

## execute(\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int) -> (T0, T1, T2, T3)
```

## execute(\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int) -> (T0, T1, T2, T3, T4)
```

## execute(\_:\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int, _ count5: Int) -> (T0, T1, T2, T3, T4, T5)
```

## execute(\_:\_:\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int, _ count5: Int, _ count6: Int) -> (T0, T1, T2, T3, T4, T5, T6)
```

## execute(\_:\_:\_:\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int, _ count5: Int, _ count6: Int, _ count7: Int) -> (T0, T1, T2, T3, T4, T5, T6, T7)
```

## execute(\_:\_:\_:\_:\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int, _ count5: Int, _ count6: Int, _ count7: Int, _ count8: Int) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8)
```

## execute(\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol, T9: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int, _ count5: Int, _ count6: Int, _ count7: Int, _ count8: Int, _ count9: Int) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)
```

## execute(\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol, T9: TensorArrayProtocol, T10: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int, _ count5: Int, _ count6: Int, _ count7: Int, _ count8: Int, _ count9: Int, _ count10: Int) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)
```

## execute(\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol, T9: TensorArrayProtocol, T10: TensorArrayProtocol, T11: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int, _ count5: Int, _ count6: Int, _ count7: Int, _ count8: Int, _ count9: Int, _ count10: Int, _ count11: Int) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11)
```

## execute(\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol, T9: TensorArrayProtocol, T10: TensorArrayProtocol, T11: TensorArrayProtocol, T12: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int, _ count5: Int, _ count6: Int, _ count7: Int, _ count8: Int, _ count9: Int, _ count10: Int, _ count11: Int, _ count12: Int) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12)
```

## execute(\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol, T9: TensorArrayProtocol, T10: TensorArrayProtocol, T11: TensorArrayProtocol, T12: TensorArrayProtocol, T13: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int, _ count5: Int, _ count6: Int, _ count7: Int, _ count8: Int, _ count9: Int, _ count10: Int, _ count11: Int, _ count12: Int, _ count13: Int) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13)
```

## execute(\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:\_:)

``` swift
func execute<T0: TensorArrayProtocol, T1: TensorArrayProtocol, T2: TensorArrayProtocol, T3: TensorArrayProtocol, T4: TensorArrayProtocol, T5: TensorArrayProtocol, T6: TensorArrayProtocol, T7: TensorArrayProtocol, T8: TensorArrayProtocol, T9: TensorArrayProtocol, T10: TensorArrayProtocol, T11: TensorArrayProtocol, T12: TensorArrayProtocol, T13: TensorArrayProtocol, T14: TensorArrayProtocol>(_ count0: Int, _ count1: Int, _ count2: Int, _ count3: Int, _ count4: Int, _ count5: Int, _ count6: Int, _ count7: Int, _ count8: Int, _ count9: Int, _ count10: Int, _ count11: Int, _ count12: Int, _ count13: Int, _ count14: Int) -> (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14)
```
