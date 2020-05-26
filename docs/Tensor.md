# Tensor

A multidimensional array of elements that is a generalization of vectors and matrices to
potentially higher dimensions.

``` swift
@frozen public struct Tensor<Scalar: TensorFlowScalar>
```

The generic parameter `Scalar` describes the type of scalars in the tensor (such as `Int32`,
`Float`, etc).

## Inheritance

`AdditiveArithmetic`, `CustomReflectable`, [`Differentiable`](/Differentiable), [`EuclideanDifferentiable`](/EuclideanDifferentiable), [`CopyableToDevice`](/CopyableToDevice), [`AnyTensor`](/AnyTensor), `Codable`, [`ConvertibleFromNumpyArray`](/ConvertibleFromNumpyArray), `CustomPlaygroundDisplayConvertible`, `CustomStringConvertible`, `ElementaryFunctions`, `Equatable`, `ExpressibleByArrayLiteral`, [`PointwiseMultiplicative`](/PointwiseMultiplicative), [`VectorProtocol`](/VectorProtocol), [`TensorGroup`](/TensorGroup), [`_LazyTensorCompatible`](/_LazyTensorCompatible)

## Nested Type Aliases

### `ArrayLiteralElement`

The type of the elements of an array literal.

``` swift
public typealias ArrayLiteralElement = _TensorElementLiteral<Scalar>
```

## Initializers

### `init(copying:to:)`

Creates a copy of `other` on the given `Device`.

``` swift
public init(copying other: Tensor, to device: Device)
```

### `init(_:deviceAndPrecisionLike:)`

Promotes a scalar to a tensor with the same device and precision as the given tensor.

``` swift
@differentiable(where Scalar: TensorFlowFloatingPoint) public init(_ value: Scalar, deviceAndPrecisionLike tensor: Tensor)
```

### `init(handle:)`

``` swift
@inlinable public init(handle: TensorHandle<Scalar>)
```

### `init(_:on:)`

Creates a 0-D tensor from a scalar value.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public init(_ value: Scalar, on device: Device = .default)
```

### `init(_:on:)`

Creates a 1D tensor from scalars.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public init(_ scalars: [Scalar], on device: Device = .default)
```

### `init(_:on:)`

Creates a 1D tensor from scalars.

``` swift
@inlinable public init<C: RandomAccessCollection>(_ vector: C, on device: Device = .default) where C.Element == Scalar
```

### `init(shape:scalars:on:)`

Creates a tensor with the specified shape and contiguous scalars in row-major order.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public init(shape: TensorShape, scalars: [Scalar], on device: Device = .default)
```

> Precondition: The product of the dimensions of the shape must equal the number of scalars.

#### Parameters

  - shape: - shape: The shape of the tensor.
  - scalars: - scalars: The scalar contents of the tensor.

### `init(shape:scalars:on:)`

Creates a tensor with the specified shape and contiguous scalars in row-major order.

``` swift
@inlinable public init(shape: TensorShape, scalars: UnsafeBufferPointer<Scalar>, on device: Device = .default)
```

> Precondition: The product of the dimensions of the shape must equal the number of scalars.

#### Parameters

  - shape: - shape: The shape of the tensor.
  - scalars: - scalars: The scalar contents of the tensor.

### `init(shape:scalars:on:)`

Creates a tensor with the specified shape and contiguous scalars in row-major order.

``` swift
@inlinable public init<C: RandomAccessCollection>(shape: TensorShape, scalars: C, on device: Device = .default) where C.Element == Scalar
```

> Precondition: The product of the dimensions of the shape must equal the number of scalars.

#### Parameters

  - shape: - shape: The shape of the tensor.
  - scalars: - scalars: The scalar contents of the tensor.

### `init(_tensorElementLiterals:)`

Creates a tensor initialized with the given elements.

``` swift
@inlinable internal init(_tensorElementLiterals elements: [_TensorElementLiteral<Scalar>])
```

> Note: This is for conversion from tensor element literals. This is a separate method because \`ShapedArray\` initializers need to call it.

### `init(arrayLiteral:)`

Creates a tensor initialized with the given elements.

``` swift
@inlinable public init(arrayLiteral elements: _TensorElementLiteral<Scalar>)
```

### `init(_owning:)`

``` swift
public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?)
```

### `init(_handles:)`

``` swift
public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle
```

### `init(_:)`

``` swift
public init(_ array: __owned ShapedArray<Scalar>)
```

### `init(shape:repeating:)`

Creates a tensor with the specified shape and a single, repeated scalar value.

``` swift
@inlinable @available(*, deprecated, renamed: "init(repeating:shape:)") public init(shape: TensorShape, repeating repeatedValue: Scalar)
```

#### Parameters

  - shape: - shape: The dimensions of the tensor.
  - repeatedValue: - repeatedValue: The scalar value to repeat.

### `init(repeating:shape:)`

Creates a tensor with the specified shape and a single, repeated scalar value.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public init(repeating repeatedValue: Scalar, shape: TensorShape)
```

#### Parameters

  - repeatedValue: - repeatedValue: The scalar value to repeat.
  - shape: - shape: The dimensions of the tensor.

### `init(broadcasting:rank:)`

Creates a tensor by broadcasting the given scalar to a given rank with
all dimensions being 1.

``` swift
@inlinable public init(broadcasting scalar: Scalar, rank: Int)
```

### `init(_:)`

Creates a tensor of shape `[4]` from a 4-tuple.

``` swift
@inlinable internal init(_ scalars: (Scalar, Scalar, Scalar, Scalar))
```

> Note: This is intended for internal use, for example, to initialize a tensor attribute from \`convolved2D\`'s \`strides\` argument.

### `init(_:)`

Creates a tensor from an array of tensors (which may themselves be scalars).

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public init(_ elements: [Tensor])
```

### `init(stacking:alongAxis:)`

Stacks `tensors`, along the `axis` dimension, into a new tensor with rank one higher than
the current tensor and each tensor in `tensors`.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public init(stacking tensors: [Tensor], alongAxis axis: Int = 0)
```

Given that `tensors` all have shape `[A, B, C]`, and `tensors.count = N`, then:

For example:

``` 
// 'x' is [1, 4]
// 'y' is [2, 5]
// 'z' is [3, 6]
Tensor(stacking: [x, y, z]) // is [[1, 4], [2, 5], [3, 6]]
Tensor(stacking: [x, y, z], alongAxis: 1) // is [[1, 2, 3], [4, 5, 6]]
```

This is the opposite of `Tensor.unstacked(alongAxis:)`.

> Precondition: All tensors must have the same shape.

> Precondition: \`axis\` must be in the range \`\[-rank, rank)\`, where \`rank\` is the rank of the provided tensors.

#### Parameters

  - tensors: - tensors: Tensors to stack.
  - axis: - axis: Dimension along which to stack. Negative values wrap around.

#### Returns

The stacked tensor.

### `init(concatenating:alongAxis:)`

Concatenates `tensors` along the `axis` dimension.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public init(concatenating tensors: [Tensor], alongAxis axis: Int = 0)
```

Given that `tensors[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, then the concatenated result
has shape `[D0, D1, ... Raxis, ...Dn]`, where `Raxis = sum(Daxis(i))`. That is, the data
from the input tensors is joined along the `axis` dimension.

For example:

``` 
// t1 is [[1, 2, 3], [4, 5, 6]]
// t2 is [[7, 8, 9], [10, 11, 12]]
Tensor(concatenating: [t1, t2]) // is [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
Tensor(concatenating: [t1, t2], alongAxis: 1) // is [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

// t3 has shape [2, 3]
// t4 has shape [2, 3]
Tensor(concatenating: [t3, t4]) // has shape [4, 3]
Tensor(concatenating: [t3, t4], alongAxis: 1) // has shape [2, 6]
```

> Note: If you are concatenating along a new axis consider using \`Tensor.init(stacking:alongAxis:)\`.

> Precondition: All tensors must have the same rank and all dimensions except \`axis\` must be equal.

> Precondition: \`axis\` must be in the range \`\[-rank, rank)\`, where \`rank\` is the rank of the provided tensors.

#### Parameters

  - tensors: - tensors: Tensors to concatenate.
  - axis: - axis: Dimension along which to concatenate. Negative values wrap around.

#### Returns

The concatenated tensor.

## Properties

### `device`

The device on which `self` is allocated.

``` swift
var device: Device
```

### `_concreteInputLazyTensor`

Returns `Self` that wraps `_concreteInputLazyTensor` of the underlying
`_AnyTensorHandle`

``` swift
var _concreteInputLazyTensor: Tensor
```

### `isReducedPrecision`

Returns true if the physical scalar type is reduced precision.

``` swift
var isReducedPrecision: Bool
```

Currently, reduced precision physical scalar types include only `BFloat16`.

### `toReducedPrecision`

Returns a copy of `self` converted to `BFloat16` physical scalar type.

``` swift
var toReducedPrecision: Self
```

### `toFullPrecision`

Returns a copy of `self` converted to `Scalar` physical scalar type.

``` swift
var toFullPrecision: Self
```

### `handle`

The underlying `TensorHandle`.

``` swift
let handle: TensorHandle<Scalar>
```

> Note: \`handle\` is public to allow user defined ops, but should not normally be used.

### `_rawTensorHandle`

``` swift
var _rawTensorHandle: CTensorHandle
```

### `_tensorFlowDataType`

``` swift
var _tensorFlowDataType: TensorDataType
```

### `rank`

The number of dimensions of the `Tensor`.

``` swift
var rank: Int
```

### `shape`

The shape of the `Tensor`.

``` swift
var shape: TensorShape
```

### `scalarCount`

The number of scalars in the `Tensor`.

``` swift
var scalarCount: Int
```

### `rankTensor`

The rank of the tensor, represented as a `Tensor<Int32>`.

``` swift
var rankTensor: Tensor<Int32>
```

### `shapeTensor`

The dimensions of the tensor, represented as a `Tensor<Int32>`.

``` swift
var shapeTensor: Tensor<Int32>
```

### `scalarCountTensor`

The number of scalars in the tensor, represented as a `Tensor<Int32>`.

``` swift
var scalarCountTensor: Tensor<Int32>
```

### `isScalar`

Returns `true` if `rank` is equal to 0 and `false` otherwise.

``` swift
var isScalar: Bool
```

### `scalar`

Returns the single scalar element if `rank` is equal to 0 and `nil`
otherwise.

``` swift
var scalar: Scalar?
```

### `array`

``` swift
var array: ShapedArray<Scalar>
```

### `scalars`

``` swift
var scalars: [Scalar]
```

### `description`

A textual representation of the tensor.

``` swift
var description: String
```

> Note: use \`fullDescription\` for a non-pretty-printed description showing all scalars.

### `fullDescription`

A full, non-pretty-printed textual representation of the tensor, showing
all scalars.

``` swift
var fullDescription: String
```

### `playgroundDescription`

``` swift
var playgroundDescription: Any
```

### `customMirror`

``` swift
var customMirror: Mirror
```

### `_unknownShapeList`

``` swift
var _unknownShapeList: [TensorShape?]
```

### `_typeList`

``` swift
var _typeList: [TensorDataType]
```

### `_tensorHandles`

``` swift
var _tensorHandles: [_AnyTensorHandle]
```

### `_lazyTensor`

``` swift
var _lazyTensor: LazyTensorHandle?
```

### `_concreteLazyTensor`

``` swift
var _concreteLazyTensor: Tensor
```

## Methods

### `scalarized()`

Reshape to scalar.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public func scalarized() -> Scalar
```

> Precondition: The tensor has exactly one scalar.

### `description(lineWidth:edgeElementCount:summarizing:)`

A textual representation of the tensor. Returns a summarized description if `summarize` is
true and the element count exceeds twice the `edgeElementCount`.

``` swift
public func description(lineWidth: Int = 80, edgeElementCount: Int = 3, summarizing: Bool = false) -> String
```

#### Parameters

  - lineWidth: - lineWidth: The max line width for printing. Used to determine number of scalars to print per line.
  - edgeElementCount: - edgeElementCount: The maximum number of elements to print before and after summarization via ellipses (`...`).
  - summarizing: - summarizing: If true, summarize description if element count exceeds twice `edgeElementCount`.

### `_unpackTensorHandles(into:)`

``` swift
public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?)
```

### `unstacked(alongAxis:)`

Unpacks the given dimension of a rank-`R` tensor into multiple rank-`(R-1)` tensors.
Unpacks `N` tensors from this tensor by chipping it along the `axis` dimension, where `N`
is inferred from this tensor's shape. For example, given a tensor with shape
`[A, B, C, D]`:

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public func unstacked(alongAxis axis: Int = 0) -> [Tensor]
```

This is the opposite of `Tensor.init(stacking:alongAxis:)`.

> Precondition: \`axis\` must be in the range \`\[-rank, rank)\`, where \`rank\` is the rank of the provided tensors.

#### Parameters

  - axis: - axis: Dimension along which to unstack. Negative values wrap around.

#### Returns

Array containing the unstacked tensors.

### `split(count:alongAxis:)`

Splits a tensor into multiple tensors. The tensor is split along dimension `axis` into
`count` smaller tensors. This requires that `count` evenly divides `shape[axis]`.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public func split(count: Int, alongAxis axis: Int = 0) -> [Tensor]
```

For example:

``` 
// 'value' is a tensor with shape [5, 30]
// Split 'value' into 3 tensors along dimension 1:
let parts = value.split(count: 3, alongAxis: 1)
parts[0] // has shape [5, 10]
parts[1] // has shape [5, 10]
parts[2] // has shape [5, 10]
```

> Precondition: \`count\` must divide the size of dimension \`axis\` evenly.

> Precondition: \`axis\` must be in the range \`\[-rank, rank)\`, where \`rank\` is the rank of the provided tensors.

#### Parameters

  - count: - count: Number of splits to create.
  - axis: - axis: The dimension along which to split this tensor. Negative values wrap around.

#### Returns

An array containing the tensors part.

### `split(sizes:alongAxis:)`

Splits a tensor into multiple tensors. The tensor is split  into `sizes.shape[0]` pieces.
The shape of the `i`-th piece has the same shape as this tensor except along dimension
`axis` where the size is `sizes[i]`.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func split(sizes: Tensor<Int32>, alongAxis axis: Int = 0) -> [Tensor]
```

For example:

``` 
// 'value' is a tensor with shape [5, 30]
// Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1:
let parts = value.split(sizes: Tensor<Int32>([4, 15, 11]), alongAxis: 1)
parts[0] // has shape [5, 4]
parts[1] // has shape [5, 15]
parts[2] // has shape [5, 11]
```

> Precondition: The values in \`sizes\` must add up to the size of dimension \`axis\`.

> Precondition: \`axis\` must be in the range \`\[-rank, rank)\`, where \`rank\` is the rank of the provided tensors.

#### Parameters

  - sizes: - sizes: 1-D tensor containing the size of each split.
  - axis: - axis: Dimension along which to split this tensor. Negative values wrap around.

#### Returns

Array containing the tensors parts.

### `tiled(multiples:)`

Returns a tiled tensor, constructed by tiling this tensor.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func tiled(multiples: [Int]) -> Tensor
```

This constructor creates a new tensor by replicating this tensor `multiples` times. The
constructed tensor's `i`'th dimension has `self.shape[i] * multiples[i]` elements, and the
values of this tensor are replicated `multiples[i]` times along the `i`'th dimension. For
example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.

> Precondition: The expected \`rank\` of multiples must be \`1\`.

> Precondition: The shape of \`multiples\` must be \`\[tensor.rank\]\`.

> Precondition: All scalars in \`multiples\` must be non-negative.

### `tiled(multiples:)`

Returns a tiled tensor, constructed by tiling this tensor.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func tiled(multiples: Tensor<Int32>) -> Tensor
```

This constructor creates a new tensor by replicating this tensor `multiples` times. The
constructed tensor's `i`'th dimension has `self.shape[i] * multiples[i]` elements, and the
values of this tensor are replicated `multiples[i]` times along the `i`'th dimension. For
example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.

> Precondition: The expected \`rank\` of multiples must be \`1\`.

> Precondition: The shape of \`multiples\` must be \`\[tensor.rank\]\`.

### `reshaped(like:)`

Reshape to the shape of the specified `Tensor`.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func reshaped<T>(like other: Tensor<T>) -> Tensor
```

> Precondition: The number of scalars matches the new shape.

### `reshaped(to:)`

Reshape to the specified shape.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func reshaped(to newShape: TensorShape) -> Tensor
```

> Precondition: The number of scalars matches the new shape.

### `reshaped(toShape:)`

Reshape to the specified `Tensor` representing a shape.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func reshaped(toShape newShape: Tensor<Int32>) -> Tensor
```

> Precondition: The number of scalars matches the new shape.

### `flattened()`

Return a copy of the tensor collapsed into a 1-D `Tensor`, in row-major order.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func flattened() -> Tensor
```

### `expandingShape(at:)`

Returns a shape-expanded `Tensor`, with a dimension of 1 inserted at the specified shape
indices.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func expandingShape(at axes: Int) -> Tensor
```

### `expandingShape(at:)`

Returns a shape-expanded `Tensor`, with a dimension of 1 inserted at the
specified shape indices.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func expandingShape(at axes: [Int]) -> Tensor
```

### `rankLifted()`

Returns a rank-lifted `Tensor` with a leading dimension of 1.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func rankLifted() -> Tensor
```

### `squeezingShape(at:)`

Removes the specified dimensions of size 1 from the shape of a tensor. If no dimensions are
specified, then all dimensions of size 1 will be removed.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func squeezingShape(at axes: Int) -> Tensor
```

### `squeezingShape(at:)`

Removes the specified dimensions of size 1 from the shape of a tensor. If no dimensions are
specified, then all dimensions of size 1 will be removed.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func squeezingShape(at axes: [Int]) -> Tensor
```

### `transposed(permutation:)`

Returns a transposed tensor, with dimensions permuted in the specified order.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func transposed(permutation: Tensor<Int32>) -> Tensor
```

### `transposed(withPermutations:)`

Returns a transposed tensor, with dimensions permuted in the specified order.

``` swift
@available(*, deprecated, renamed: "transposed(permutation:)") @inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func transposed(withPermutations permutations: Tensor<Int32>) -> Tensor
```

### `transposed(permutation:)`

Returns a transposed tensor, with dimensions permuted in the specified order.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func transposed(permutation: [Int]) -> Tensor
```

### `transposed(withPermutations:)`

Returns a transposed tensor, with dimensions permuted in the specified order.

``` swift
@available(*, deprecated, renamed: "transposed(permutation:)") @inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func transposed(withPermutations permutations: [Int]) -> Tensor
```

### `transposed(permutation:)`

Returns a transposed tensor, with dimensions permuted in the specified order.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func transposed(permutation: Int) -> Tensor
```

### `transposed(withPermutations:)`

Returns a transposed tensor, with dimensions permuted in the specified order.

``` swift
@available(*, deprecated, renamed: "transposed(permutation:)") @inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func transposed(withPermutations permutations: Int) -> Tensor
```

### `transposed()`

Returns a transposed tensor, with dimensions permuted in reverse order.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func transposed() -> Tensor
```

### `reversed(inAxes:)`

Returns a tensor with specified dimensions reversed.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) func reversed(inAxes axes: Tensor<Int32>) -> Tensor
```

> Precondition: Each value in \`axes\` must be in the range \`-rank..\<rank\`.

> Precondition: There must be no duplication in \`axes\`.

### `reversed(inAxes:)`

Returns a tensor with specified dimensions reversed.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) func reversed(inAxes axes: [Int]) -> Tensor
```

> Precondition: Each value in \`axes\` must be in the range \`-rank..\<rank\`.

> Precondition: There must be no duplication in \`axes\`.

### `reversed(inAxes:)`

Returns a tensor with specified dimensions reversed.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) func reversed(inAxes axes: Int) -> Tensor
```

> Precondition: Each value in \`axes\` must be in the range \`-rank..\<rank\`.

> Precondition: There must be no duplication in \`axes\`.

### `concatenated(with:alongAxis:)`

Returns a concatenated tensor along the specified axis.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public func concatenated(with other: Tensor, alongAxis axis: Int = 0) -> Tensor
```

> Precondition: The tensors must have the same dimensions, except for the specified axis.

> Precondition: The axis must be in the range \`-rank..\<rank\`.

### `++(lhs:rhs:)`

Concatenation operator.

``` swift
@inlinable @differentiable(where Scalar: TensorFlowFloatingPoint) public static func ++(lhs: Tensor, rhs: Tensor) -> Tensor
```

> Note: \`++\` is a custom operator that does not exist in Swift, but does in Haskell/Scala. Its addition is not an insignificant language change and may be controversial. The existence/naming of \`++\` will be discussed during a later API design phase.

### `gathering(atIndices:alongAxis:)`

Returns a tensor by gathering slices of the input at `indices` along the `axis` dimension

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func gathering<Index: TensorFlowIndex>(atIndices indices: Tensor<Index>, alongAxis axis: Int = 0) -> Tensor
```

For 0-D (scalar) `indices`:

``` 
result[p_0,          ..., p_{axis-1},
       p_{axis + 1}, ..., p_{N-1}] =
self[p_0,          ..., p_{axis-1},
     indices,
     p_{axis + 1}, ..., p_{N-1}]
```

For 1-D (vector) `indices`:

``` 
result[p_0,          ..., p_{axis-1},
       i,
       p_{axis + 1}, ..., p_{N-1}] =
self[p_0,          ..., p_{axis-1},
     indices[i],
     p_{axis + 1}, ..., p_{N-1}]
```

In the general case, produces a resulting tensor where:

``` 
result[p_0,             ..., p_{axis-1},
       i_{batch\_dims}, ..., i_{M-1},
       p_{axis + 1},    ..., p_{N-1}] =
self[p_0,             ..., p_{axis-1},
     indices[i_0,     ..., i_{M-1}],
     p_{axis + 1},    ..., p_{N-1}]
```

where `N = self.rank` and `M = indices.rank`.

The shape of the resulting tensor is:
`self.shape[..<axis] + indices.shape + self.shape[(axis + 1)...]`.

> Note: On CPU, if an out-of-range index is found, an error is thrown. On GPU, if an out-of-range index is found, a 0 is stored in the corresponding output values.

> Precondition: \`axis\` must be in the range \`\[-rank, rank)\`.

#### Parameters

  - indices: - indices: Contains the indices to gather at.
  - axis: - axis: Dimension along which to gather. Negative values wrap around.

#### Returns

The gathered tensor.

### `batchGathering(atIndices:alongAxis:batchDimensionCount:)`

Returns slices of this tensor at `indices` along the `axis` dimension, while ignoring the
first `batchDimensionCount` dimensions that correspond to batch dimensions. The gather is
performed along the first non-batch dimension.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func batchGathering<Index: TensorFlowIndex>(atIndices indices: Tensor<Index>, alongAxis axis: Int = 1, batchDimensionCount: Int = 1) -> Tensor
```

Performs similar functionality to `gathering`, except that the resulting tensor shape is
now `shape[..<axis] + indices.shape[batchDimensionCount...] + shape[(axis + 1)...]`.

> Precondition: \`axis\` must be in the range \`-rank..\<rank\`, while also being greater than or equal to \`batchDimensionCount\`.

> Precondition: \`batchDimensionCount\` must be less than \`indices.rank\`.

#### Parameters

  - indices: - indices: Contains the indices to gather.
  - axis: - axis: Dimension along which to gather. Negative values wrap around.
  - batchDimensionCount: - batchDimensionCount: Number of leading batch dimensions to ignore.

#### Returns

The gathered tensor.

### `gathering(where:alongAxis:)`

Returns a tensor by gathering the values after applying the provided boolean mask to the input.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func gathering(where mask: Tensor<Bool>, alongAxis axis: Int = 0) -> Tensor
```

For example:

``` 
// 1-D example
// tensor is [0, 1, 2, 3]
// mask is [true, false, true, false]
tensor.gathering(where: mask) // is [0, 2]

// 2-D example
// tensor is [[1, 2], [3, 4], [5, 6]]
// mask is [true, false, true]
tensor.gathering(where: mask) // is [[1, 2], [5, 6]]
```

In general, `0 < mask.rank = K <= tensor.rank`, and the `mask`'s shape must match the first
K dimensions of the `tensor`'s shape. We then have:
`tensor.gathering(where: mask)[i, j1, ..., jd] = tensor[i1, ..., iK, j1, ..., jd]`, where
`[i1, ..., iK]` is the `i`th `true` entry of `mask` (row-major order).

The `axis` could be used with `mask` to indicate the axis to mask from. In that case,
`axis + mask.rank <= tensor.rank` and the `mask``'s shape must match the first `axis + mask.rank`dimensions of the`tensor\`'s shape.

> Precondition: The \`mask\` cannot be a scalar: \`mask.rank \!= 0\`.

#### Parameters

  - mask: - mask: K-D boolean tensor, where `K <= self.rank`.
  - axis: - axis: 0-D integer tensor representing the axis in `self` to mask from, where `K + axis <= self.rank`.

#### Returns

`(self.rank - K + 1)`-dimensional tensor populated by entries in this tensor corresponding to `true` values in `mask`.

### `nonZeroIndices()`

Returns the locations of non-zero / true values in this tensor.

``` swift
@inlinable public func nonZeroIndices() -> Tensor<Int64>
```

The coordinates are returned in a 2-D tensor where the first dimension (rows) represents the
number of non-zero elements, and the second dimension (columns) represents the coordinates
of the non-zero elements. Keep in mind that the shape of the output tensor can vary
depending on how many true values there are in this tensor. Indices are output in row-major
order.

For example:

``` 
// 'input' is [[true, false], [true, false]]
// 'input' has 2 true values and so the output has 2 rows.
// 'input' has rank of 2, and so the second dimension of the output has size 2.
input.nonZeroIndices() // is [[0, 0], [1, 0]]

// 'input' is [[[ true, false], [ true, false]],
//             [[false,  true], [false,  true]],
//             [[false, false], [false,  true]]]
// 'input' has 5 true values and so the output has 5 rows.
// 'input' has rank 3, and so the second dimension of the output has size 3.
input.nonZeroIndices() // is [[0, 0, 0],
                       //     [0, 1, 0],
                       //     [1, 0, 1],
                       //     [1, 1, 1],
                       //     [2, 1, 1]]
```

#### Returns

A tensor with shape `(num_true, rank(condition))`.

### `broadcasted(toShape:)`

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func broadcasted(toShape shape: Tensor<Int32>) -> Tensor
```

### `broadcasted(to:)`

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func broadcasted(to shape: TensorShape) -> Tensor
```

### `broadcasted(like:)`

Broadcast to the same shape as the specified `Tensor`.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func broadcasted<OtherScalar>(like other: Tensor<OtherScalar>) -> Tensor
```

> Precondition: The specified shape must be compatible for broadcasting.

### `.=(lhs:rhs:)`

``` swift
@inlinable public static func .=(lhs: inout Tensor, rhs: Tensor)
```

### `slice(lowerBounds:upperBounds:)`

Extracts a slice from the tensor defined by lower and upper bounds for
each dimension.

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func slice(lowerBounds: [Int], upperBounds: [Int]) -> Tensor
```

#### Parameters

  - lowerBounds: - lowerBounds: The lower bounds at each dimension.
  - upperBounds: - upperBounds: The upper bounds at each dimension.

### `slice(lowerBounds:sizes:)`

``` swift
@inlinable @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint) public func slice(lowerBounds: Tensor<Int32>, sizes: Tensor<Int32>) -> Tensor
```

### `isAxisInRange(_:)`

Returns `true` if the given axis is in the range `[-rank, rank)`.

``` swift
@usableFromInline internal func isAxisInRange<T: BinaryInteger>(_ axis: T) -> Bool
```

### `isAxisInRange(_:)`

Returns `true` if the given scalar tensor is in the range `[-rank, rank)`.

``` swift
@usableFromInline internal func isAxisInRange(_ axis: Tensor<Int32>) -> Bool
```

### `areAxesInRange(_:)`

Returns `true` if all given axes are in the range `[-rank, rank)`.

``` swift
@usableFromInline internal func areAxesInRange<T: BinaryInteger>(_ axes: [T]) -> Bool
```

### `areAxesInRange(_:)`

Returns `true` if all scalars of the given 1-D tensor are in the range `[-rank, rank)`.

``` swift
@usableFromInline internal func areAxesInRange(_ axes: Tensor<Int32>) -> Bool
```

### `replacing(with:where:)`

Replaces elements of this tensor with `other` in the lanes where `mask` is
`true`.

``` swift
@inlinable @differentiable(wrt: (self, other) where Scalar: TensorFlowFloatingPoint) public func replacing(with other: Tensor, where mask: Tensor<Bool>) -> Tensor
```

> Precondition: \`self\` and \`other\` must have the same shape. If \`self\` and \`other\` are scalar, then \`mask\` must also be scalar. If \`self\` and \`other\` have rank greater than or equal to \`1\`, then \`mask\` must be either have the same shape as \`self\` or be a 1-D \`Tensor\` such that \`mask.scalarCount == self.shape\[0\]\`.
