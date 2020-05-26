# \_AnyTensorHandle

This protocol abstracts the underlying representation of a tensor. Any type
that conforms to this protocol can be used as a `TensorHandle` in the
`TensorFlow` library, as it much provide a way to convert the underlying tensor
handle into a `ConcreteTensorHandle`, which wraps a `TFE_TensorHandle *`
TODO(https://bugs.swift.org/browse/TF-527): This is defined as a class-bound

``` swift
public protocol _AnyTensorHandle: class
```

## Inheritance

`class`

## Requirements

## \_tfeTensorHandle

``` swift
var _tfeTensorHandle: TFETensorHandle
```

## rank

``` swift
var rank: Int
```

## shape

``` swift
var shape: TensorShape
```
