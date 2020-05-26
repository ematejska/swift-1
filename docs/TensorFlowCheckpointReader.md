# TensorFlowCheckpointReader

A TensorFlow checkpoint file reader.

``` swift
public class TensorFlowCheckpointReader
```

## Initializers

### `init(checkpointPath:)`

Creates a new TensorFlow checkpoint reader.

``` swift
public init(checkpointPath: String)
```

## Properties

### `status`

``` swift
let status: OpaquePointer
```

### `handle`

``` swift
let handle: OpaquePointer
```

### `checkpointPath`

The path to the checkpoint file.

``` swift
let checkpointPath: String
```

### `tensorCount`

The number of tensors stored in the checkpoint.

``` swift
var tensorCount: Int
```

### `tensorNames`

The names of the tensors stored in the checkpoint.

``` swift
var tensorNames: [String]
```

## Methods

### `containsTensor(named:)`

Returns `true` if the checkpoint contains a tensor with the provided name.

``` swift
public func containsTensor(named name: String) -> Bool
```

### `shapeOfTensor(named:)`

Returns the shape of the tensor with the provided name stored in the checkpoint.

``` swift
public func shapeOfTensor(named name: String) -> TensorShape
```

### `scalarTypeOfTensor(named:)`

Returns the scalar type of the tensor with the provided name stored in the checkpoint.

``` swift
public func scalarTypeOfTensor(named name: String) -> Any.Type
```

### `loadTensor(named:)`

Loads and returns the value of the tensor with the provided name stored in the checkpoint.

``` swift
public func loadTensor<Scalar: _TensorFlowDataTypeCompatible>(named name: String) -> ShapedArray<Scalar>
```
