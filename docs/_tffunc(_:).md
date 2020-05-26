# \_tffunc(\_:)

Trace the given function and return the name of the corresponding `TF_Function: In -> Out` that
was created.

``` swift
public func _tffunc<In: TensorGroup, Out: TensorGroup>(_ fn: (In) -> Out) -> String
```
