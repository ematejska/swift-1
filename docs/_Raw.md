# \_Raw

``` swift
public enum _Raw
```

## Properties

### `generatedTensorFlowVersion`

``` swift
let generatedTensorFlowVersion
```

### `generatedTensorFlowGitVersion`

``` swift
let generatedTensorFlowGitVersion
```

## Methods

### `argMax(_:dimension:)`

``` swift
public static func argMax<T: TensorFlowNumeric, OutputType: TensorFlowIndex>(_ input: Tensor<T>, dimension: Int64) -> Tensor<OutputType>
```

### `mean(_:reductionIndices:keepDims:)`

``` swift
public static func mean<T: TensorFlowNumeric>(_ input: Tensor<T>, reductionIndices: [Int64], keepDims: Bool = false) -> Tensor<T>
```

### `reshape(_:shape:)`

``` swift
public static func reshape<T: TensorFlowScalar>(_ tensor: Tensor<T>, shape: [Int64]) -> Tensor<T>
```

### `sum(_:reductionIndices:keepDims:)`

``` swift
public static func sum<T: TensorFlowNumeric>(_ input: Tensor<T>, reductionIndices: [Int64], keepDims: Bool = false) -> Tensor<T>
```

### `broadcastTo(_:shape:)`

``` swift
public static func broadcastTo<T: TensorFlowScalar>(_ input: Tensor<T>, shape: [Int64]) -> Tensor<T>
```

### `a()`

``` swift
@inlinable @inline(__always) public static func a() -> Tensor<Float>
```

### `abort(errorMsg:exitWithoutError:)`

Raise a exception to abort the process when called.

``` swift
@inlinable @inline(__always) public static func abort(errorMsg: String, exitWithoutError: Bool = false)
```

If exit\_without\_error is true, the process will exit normally,
otherwise it will exit with a SIGABORT signal.

Returns nothing but an exception.

### `abs(_:)`

Computes the absolute value of a tensor.

``` swift
@inlinable @inline(__always) public static func abs<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).

### `accumulateNV2(inputs:shape:)`

Returns the element-wise sum of a list of tensors.

``` swift
@inlinable @inline(__always) public static func accumulateNV2<T: TensorFlowNumeric>(inputs: [Tensor<T>], shape: TensorShape?) -> Tensor<T>
```

`tf.accumulate_n_v2` performs the same operation as `tf.add_n`, but does not
wait for all of its inputs to be ready before beginning to sum. This can
save memory if inputs are ready at different times, since minimum temporary
storage is proportional to the output size rather than the inputs size.

Unlike the original `accumulate_n`, `accumulate_n_v2` is differentiable.

Returns a `Tensor` of same shape and type as the elements of `inputs`.

#### Parameters

  - inputs: - inputs: A list of `Tensor` objects, each with same shape and type.

### `acos(_:)`

Computes acos of x element-wise.

``` swift
@inlinable @inline(__always) public static func acos<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

### `acosh(_:)`

Computes inverse hyperbolic cosine of x element-wise.

``` swift
@inlinable @inline(__always) public static func acosh<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Given an input tensor, the function computes inverse hyperbolic cosine of every element.
Input range is `[1, inf]`. It returns `nan` if the input lies outside the range.

``` python
x = tf.constant([-2, -0.5, 1, 1.2, 200, 10000, float("inf")])
tf.math.acosh(x) ==> [nan nan 0. 0.62236255 5.9914584 9.903487 inf]
```

### `add(_:_:)`

Returns x + y element-wise.

``` swift
@inlinable @inline(__always) public static func add<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `add(_:_:)`

Returns x + y element-wise.

``` swift
@inlinable @inline(__always) public static func add(_ x: StringTensor, _ y: StringTensor) -> StringTensor
```

*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `addManySparseToTensorsMap(sparseIndices:sparseValues:sparseShape:container:sharedName:)`

Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.

``` swift
@inlinable @inline(__always) public static func addManySparseToTensorsMap<T: TensorFlowScalar>(sparseIndices: Tensor<Int64>, sparseValues: Tensor<T>, sparseShape: Tensor<Int64>, container: String, sharedName: String) -> Tensor<Int64>
```

A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
`sparse_values`, and `sparse_shape`, where

`sparse_indices.shape[1] == sparse_shape.shape[0] == R`

An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
having a first `sparse_indices` column taking values between `[0, N)`, where
the minibatch size `N == sparse_shape[0]`.

The input `SparseTensor` must have rank `R` greater than 1, and the first
dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The stored
`SparseTensor` objects pointed to by each row of the output `sparse_handles`
will have rank `R-1`.

The `SparseTensor` values can then be read out as part of a minibatch by passing
the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
the correct `SparseTensorsMap` is accessed, ensure that the same
`container` and `shared_name` are passed to that Op.  If no `shared_name`
is provided here, instead use the *name* of the Operation created by calling
`AddManySparseToTensorsMap` as the `shared_name` passed to
`TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

### `addN(inputs:)`

Add all input tensors element wise.

``` swift
@inlinable @inline(__always) public static func addN<T: TensorFlowNumeric>(inputs: [Tensor<T>]) -> Tensor<T>
```

Inputs must be of same size and shape.

``` python
x = [9, 7, 10]
tf.math.add_n(x) ==> 26
```

### `addSparseToTensorsMap(sparseIndices:sparseValues:sparseShape:container:sharedName:)`

Add a `SparseTensor` to a `SparseTensorsMap` return its handle.

``` swift
@inlinable @inline(__always) public static func addSparseToTensorsMap<T: TensorFlowScalar>(sparseIndices: Tensor<Int64>, sparseValues: Tensor<T>, sparseShape: Tensor<Int64>, container: String, sharedName: String) -> Tensor<Int64>
```

A `SparseTensor` is represented by three tensors: `sparse_indices`,
`sparse_values`, and `sparse_shape`.

This operator takes the given `SparseTensor` and adds it to a container
object (a `SparseTensorsMap`).  A unique key within this container is generated
in the form of an `int64`, and this is the value that is returned.

The `SparseTensor` can then be read out as part of a minibatch by passing
the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
the correct `SparseTensorsMap` is accessed, ensure that the same
`container` and `shared_name` are passed to that Op.  If no `shared_name`
is provided here, instead use the *name* of the Operation created by calling
`AddSparseToTensorsMap` as the `shared_name` passed to
`TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

### `addV2(_:_:)`

Returns x + y element-wise.

``` swift
@inlinable @inline(__always) public static func addV2<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `adjustContrast(images:contrastFactor:minValue:maxValue:)`

Deprecated. Disallowed in GraphDef version \>= 2.

``` swift
@inlinable @inline(__always) public static func adjustContrast<T: TensorFlowNumeric>(images: Tensor<T>, contrastFactor: Tensor<Float>, minValue: Tensor<Float>, maxValue: Tensor<Float>) -> Tensor<Float>
```

### `adjustContrastv2(images:contrastFactor:)`

Adjust the contrast of one or more images.

``` swift
@inlinable @inline(__always) public static func adjustContrastv2<T: FloatingPoint & TensorFlowScalar>(images: Tensor<T>, contrastFactor: Tensor<Float>) -> Tensor<T>
```

`images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
interpreted as `[height, width, channels]`.  The other dimensions only
represent a collection of images, such as `[batch, height, width, channels].`

Contrast is adjusted independently for each channel of each image.

For each channel, the Op first computes the mean of the image pixels in the
channel and then adjusts each component of each pixel to
`(x - mean) * contrast_factor + mean`.

#### Parameters

  - images: - images: Images to adjust.  At least 3-D.

### `adjustHue(images:delta:)`

Adjust the hue of one or more images.

``` swift
@inlinable @inline(__always) public static func adjustHue<T: FloatingPoint & TensorFlowScalar>(images: Tensor<T>, delta: Tensor<Float>) -> Tensor<T>
```

`images` is a tensor of at least 3 dimensions.  The last dimension is
interpretted as channels, and must be three.

The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A delta is then applied all the hue values,
and then remapped back to RGB colorspace.

#### Parameters

  - images: - images: Images to adjust.  At least 3-D.
  - delta: - delta: A float delta to add to the hue.

### `adjustSaturation(images:scale:)`

Adjust the saturation of one or more images.

``` swift
@inlinable @inline(__always) public static func adjustSaturation<T: FloatingPoint & TensorFlowScalar>(images: Tensor<T>, scale: Tensor<Float>) -> Tensor<T>
```

`images` is a tensor of at least 3 dimensions.  The last dimension is
interpretted as channels, and must be three.

The input image is considered in the RGB colorspace. Conceptually, the RGB
colors are first mapped into HSV. A scale is then applied all the saturation
values, and then remapped back to RGB colorspace.

#### Parameters

  - images: - images: Images to adjust.  At least 3-D.
  - scale: - scale: A float scale to add to the saturation.

### `all(_:reductionIndices:keepDims:)`

Computes the "logical and" of elements across dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func all<Tidx: TensorFlowIndex>(_ input: Tensor<Bool>, reductionIndices: Tensor<Tidx>, keepDims: Bool = false) -> Tensor<Bool>
```

Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

#### Parameters

  - input: - input: The tensor to reduce.

### `allCandidateSampler(trueClasses:numTrue:numSampled:unique:seed:seed2:)`

Generates labels for candidate sampling with a learned unigram distribution.

``` swift
@inlinable @inline(__always) public static func allCandidateSampler(trueClasses: Tensor<Int64>, numTrue: Int64, numSampled: Int64, unique: Bool, seed: Int64 = 0, seed2: Int64 = 0) -> (
    sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>,
    sampledExpectedCount: Tensor<Float>
  )
```

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

### `allToAll(_:groupAssignment:concatDimension:splitDimension:splitCount:)`

An Op to exchange data across TPU replicas.

``` swift
@inlinable @inline(__always) public static func allToAll<T: TensorFlowScalar>(_ input: Tensor<T>, groupAssignment: Tensor<Int32>, concatDimension: Int64, splitDimension: Int64, splitCount: Int64) -> Tensor<T>
```

On each replica, the input is split into `split_count` blocks along
`split_dimension` and send to the other replicas given group\_assignment. After
receiving `split_count` - 1 blocks from other replicas, we concatenate the
blocks along `concat_dimension` as the output.

For example, suppose there are 2 TPU replicas:
replica 0 receives input: `[[A, B]]`
replica 1 receives input: `[[C, D]]`

group\_assignment=`[[0, 1]]`
concat\_dimension=0
split\_dimension=1
split\_count=2

replica 0's output: `[[A], [C]]`
replica 1's output: `[[B], [D]]`

#### Parameters

  - input: - input: The local input to the sum.

### `angle(_:)`

Returns the argument of a complex number.

``` swift
@inlinable @inline(__always) public static func angle<T: TensorFlowScalar, Tout: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<Tout>
```

Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the argument of each element in `input`. All elements in
`input` must be complex numbers of the form \\(a + bj\\), where *a*
is the real part and *b* is the imaginary part.

The argument returned by this operation is of the form \\(atan2(b, a)\\).

For example:

``` 
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.angle(input) ==> [2.0132, 1.056]
```

@compatibility(numpy)
Equivalent to np.angle.
@end\_compatibility

### `anonymousIterator(outputTypes:outputShapes:)`

A container for an iterator resource.

``` swift
@inlinable @inline(__always) public static func anonymousIterator(outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> ResourceHandle
```

### `anonymousIteratorV2(outputTypes:outputShapes:)`

A container for an iterator resource.

``` swift
@inlinable @inline(__always) public static func anonymousIteratorV2(outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> (handle: ResourceHandle, deleter: VariantHandle)
```

### `anonymousMemoryCache()`

``` swift
@inlinable @inline(__always) public static func anonymousMemoryCache() -> (handle: ResourceHandle, deleter: VariantHandle)
```

### `anonymousMultiDeviceIterator(devices:outputTypes:outputShapes:)`

A container for a multi device iterator resource.

``` swift
@inlinable @inline(__always) public static func anonymousMultiDeviceIterator(devices: [String], outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> (handle: ResourceHandle, deleter: VariantHandle)
```

### `anonymousRandomSeedGenerator(seed:seed2:)`

``` swift
@inlinable @inline(__always) public static func anonymousRandomSeedGenerator(seed: Tensor<Int64>, seed2: Tensor<Int64>) -> (handle: ResourceHandle, deleter: VariantHandle)
```

### `any(_:reductionIndices:keepDims:)`

Computes the "logical or" of elements across dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func any<Tidx: TensorFlowIndex>(_ input: Tensor<Bool>, reductionIndices: Tensor<Tidx>, keepDims: Bool = false) -> Tensor<Bool>
```

Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

#### Parameters

  - input: - input: The tensor to reduce.

### `approximateEqual(_:_:tolerance:)`

Returns the truth value of abs(x-y) \< tolerance element-wise.

``` swift
@inlinable @inline(__always) public static func approximateEqual<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>, tolerance: Double = 1e-05) -> Tensor<Bool>
```

### `argMax(_:dimension:)`

Returns the index with the largest value across dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func argMax<T: TensorFlowNumeric, Tidx: TensorFlowIndex, OutputType: TensorFlowIndex>(_ input: Tensor<T>, dimension: Tensor<Tidx>) -> Tensor<OutputType>
```

Note that in case of ties the identity of the return value is not guaranteed.

Usage:

``` python
import tensorflow as tf
a = [1, 10, 26.9, 2.8, 166.32, 62.3]
b = tf.math.argmax(input = a)
c = tf.keras.backend.eval(b)
# c = 4
# here a[4] = 166.32 which is the largest element of a across axis 0
```

#### Parameters

  - dimension: - dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`. Describes which dimension of the input Tensor to reduce across. For vectors, use dimension = 0.

### `argMin(_:dimension:)`

Returns the index with the smallest value across dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func argMin<T: TensorFlowNumeric, Tidx: TensorFlowIndex, OutputType: TensorFlowIndex>(_ input: Tensor<T>, dimension: Tensor<Tidx>) -> Tensor<OutputType>
```

Note that in case of ties the identity of the return value is not guaranteed.

Usage:

``` python
import tensorflow as tf
a = [1, 10, 26.9, 2.8, 166.32, 62.3]
b = tf.math.argmin(input = a)
c = tf.keras.backend.eval(b)
# c = 0
# here a[0] = 1 which is the smallest element of a across axis 0
```

#### Parameters

  - dimension: - dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`. Describes which dimension of the input Tensor to reduce across. For vectors, use dimension = 0.

### `asString(_:precision:scientific:shortest:width:fill:)`

Converts each entry in the given tensor to strings.

``` swift
@inlinable @inline(__always) public static func asString<T: TensorFlowScalar>(_ input: Tensor<T>, precision: Int64 = -1, scientific: Bool = false, shortest: Bool = false, width: Int64 = -1, fill: String) -> StringTensor
```

Supports many numeric types and boolean.

For Unicode, see the
\[https://www.tensorflow.org/tutorials/representation/unicode\](Working with Unicode text)
tutorial.

### `asin(_:)`

Computes the trignometric inverse sine of x element-wise.

``` swift
@inlinable @inline(__always) public static func asin<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

The `tf.math.asin` operation returns the inverse of `tf.math.sin`, such that
if `y = tf.math.sin(x)` then, `x = tf.math.asin(y)`.

**Note**: The output of `tf.math.asin` will lie within the invertible range
of sine, i.e \[-pi/2, pi/2\].

For example:

``` python
# Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
x = tf.constant([1.047, 0.785])
y = tf.math.sin(x) # [0.8659266, 0.7068252]
 
tf.math.asin(y) # [1.047, 0.785] = x
```

### `asinh(_:)`

Computes inverse hyperbolic sine of x element-wise.

``` swift
@inlinable @inline(__always) public static func asinh<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Given an input tensor, this function computes inverse hyperbolic sine
for every element in the tensor. Both input and output has a range of
`[-inf, inf]`.

``` python
x = tf.constant([-float("inf"), -2, -0.5, 1, 1.2, 200, 10000, float("inf")])
tf.math.asinh(x) ==> [-inf -1.4436355 -0.4812118 0.8813736 1.0159732 5.991471 9.903487 inf]
```

### `assert(condition:data:summarize:)`

Asserts that the given condition is true.

``` swift
@inlinable @inline(__always) public static func assert<T: TensorArrayProtocol>(condition: Tensor<Bool>, data: T, summarize: Int64 = 3)
```

If `condition` evaluates to false, print the list of tensors in `data`.
`summarize` determines how many entries of the tensors to print.

#### Parameters

  - condition: - condition: The condition to evaluate.
  - data: - data: The tensors to print out when condition is false.

### `assertNextDataset(inputDataset:transformations:outputTypes:outputShapes:)`

A transformation that asserts which transformations happen next.

``` swift
@inlinable @inline(__always) public static func assertNextDataset(inputDataset: VariantHandle, transformations: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

This transformation checks whether the camel-case names (i.e. "FlatMap", not
"flat\_map") of the transformations following this transformation match the list
of names in the `transformations` argument. If there is a mismatch, the
transformation raises an exception.

The check occurs when iterating over the contents of the dataset, which
means that the check happens *after* any static optimizations are applied
to the dataset graph.

#### Parameters

  - transformations: - transformations: A `tf.string` vector `tf.Tensor` identifying the transformations that are expected to happen next.

### `assignAddVariableOp(resource:value:)`

Adds a value to the current value of a variable.

``` swift
@inlinable @inline(__always) public static func assignAddVariableOp<Dtype: TensorFlowScalar>(resource: ResourceHandle, value: Tensor<Dtype>)
```

Any ReadVariableOp with a control dependency on this op is guaranteed to
see the incremented value or a subsequent newer one.

#### Parameters

  - resource: - resource: handle to the resource in which to store the variable.
  - value: - value: the value by which the variable will be incremented.

### `assignSubVariableOp(resource:value:)`

Subtracts a value from the current value of a variable.

``` swift
@inlinable @inline(__always) public static func assignSubVariableOp<Dtype: TensorFlowScalar>(resource: ResourceHandle, value: Tensor<Dtype>)
```

Any ReadVariableOp with a control dependency on this op is guaranteed to
see the decremented value or a subsequent newer one.

#### Parameters

  - resource: - resource: handle to the resource in which to store the variable.
  - value: - value: the value by which the variable will be incremented.

### `assignVariableOp(resource:value:)`

Assigns a new value to a variable.

``` swift
@inlinable @inline(__always) public static func assignVariableOp<Dtype: TensorFlowScalar>(resource: ResourceHandle, value: Tensor<Dtype>)
```

Any ReadVariableOp with a control dependency on this op is guaranteed to return
this value or a subsequent newer value of the variable.

#### Parameters

  - resource: - resource: handle to the resource in which to store the variable.
  - value: - value: the value to set the new tensor to use.

### `atan(_:)`

Computes the trignometric inverse tangent of x element-wise.

``` swift
@inlinable @inline(__always) public static func atan<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

The `tf.math.atan` operation returns the inverse of `tf.math.tan`, such that
if `y = tf.math.tan(x)` then, `x = tf.math.atan(y)`.

**Note**: The output of `tf.math.atan` will lie within the invertible range
of tan, i.e (-pi/2, pi/2).

For example:

``` python
# Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
x = tf.constant([1.047, 0.785])
y = tf.math.tan(x) # [1.731261, 0.99920404]
 
tf.math.atan(y) # [1.047, 0.785] = x
```

### `atan2(_:_:)`

Computes arctangent of `y/x` element-wise, respecting signs of the arguments.

``` swift
@inlinable @inline(__always) public static func atan2<T: FloatingPoint & TensorFlowScalar>(_ y: Tensor<T>, _ x: Tensor<T>) -> Tensor<T>
```

This is the angle ( \\theta \\in \[-\\pi, \\pi\] ) such that
\[ x = r \\cos(\\theta) \]
and
\[ y = r \\sin(\\theta) \]
where (r = \\sqrt(x^2 + y^2) ).

### `atanh(_:)`

Computes inverse hyperbolic tangent of x element-wise.

``` swift
@inlinable @inline(__always) public static func atanh<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Given an input tensor, this function computes inverse hyperbolic tangent
for every element in the tensor. Input range is `[-1,1]` and output range is
`[-inf, inf]`. If input is `-1`, output will be `-inf` and if the
input is `1`, output will be `inf`. Values outside the range will have
`nan` as output.

``` python
x = tf.constant([-float("inf"), -1, -0.5, 1, 0, 0.5, 10, float("inf")])
tf.math.atanh(x) ==> [nan -inf -0.54930615 inf  0. 0.54930615 nan nan]
```

### `attr(_:)`

``` swift
@inlinable @inline(__always) public static func attr(_ a: Int64)
```

### `attrBool(_:)`

``` swift
@inlinable @inline(__always) public static func attrBool(_ a: Bool)
```

### `attrBoolList(_:)`

``` swift
@inlinable @inline(__always) public static func attrBoolList(_ a: [Bool])
```

### `attrDefault(_:)`

``` swift
@inlinable @inline(__always) public static func attrDefault(_ a: String = "banana")
```

### `attrEmptyListDefault(_:)`

``` swift
@inlinable @inline(__always) public static func attrEmptyListDefault(_ a: [Double])
```

### `attrEnum(_:)`

``` swift
@inlinable @inline(__always) public static func attrEnum(_ a: A)
```

### `attrEnumList(_:)`

``` swift
@inlinable @inline(__always) public static func attrEnumList(_ a: [String])
```

### `attrFloat(_:)`

``` swift
@inlinable @inline(__always) public static func attrFloat(_ a: Double)
```

### `attrListDefault(_:)`

``` swift
@inlinable @inline(__always) public static func attrListDefault(_ a: [Int32] = [5, 15])
```

### `attrListMin(_:)`

``` swift
@inlinable @inline(__always) public static func attrListMin(_ a: [Int32])
```

### `attrListTypeDefault(_:_:)`

``` swift
@inlinable @inline(__always) public static func attrListTypeDefault<T: TensorFlowScalar>(_ a: [Tensor<T>], _ b: [Tensor<T>])
```

### `attrMin(_:)`

``` swift
@inlinable @inline(__always) public static func attrMin(_ a: Int64)
```

### `attrPartialShape(_:)`

``` swift
@inlinable @inline(__always) public static func attrPartialShape(_ a: TensorShape?)
```

### `attrPartialShapeList(_:)`

``` swift
@inlinable @inline(__always) public static func attrPartialShapeList(_ a: [TensorShape?])
```

### `attrShape(_:)`

``` swift
@inlinable @inline(__always) public static func attrShape(_ a: TensorShape?)
```

### `attrShapeList(_:)`

``` swift
@inlinable @inline(__always) public static func attrShapeList(_ a: [TensorShape?])
```

### `attrTypeDefault(_:)`

``` swift
@inlinable @inline(__always) public static func attrTypeDefault<T: TensorFlowScalar>(_ a: Tensor<T>)
```

### `audioMicrofrontend(audio:sampleRate:windowSize:windowStep:numChannels:upperBandLimit:lowerBandLimit:smoothingBits:evenSmoothing:oddSmoothing:minSignalRemaining:enablePcan:pcanStrength:pcanOffset:gainBits:enableLog:scaleShift:leftContext:rightContext:frameStride:zeroPadding:outScale:)`

Audio Microfrontend Op.

``` swift
@inlinable @inline(__always) public static func audioMicrofrontend<OutType: TensorFlowNumeric>(audio: Tensor<Int16>, sampleRate: Int64 = 16000, windowSize: Int64 = 25, windowStep: Int64 = 10, numChannels: Int64 = 32, upperBandLimit: Double = 7500, lowerBandLimit: Double = 125, smoothingBits: Int64 = 10, evenSmoothing: Double = 0.025, oddSmoothing: Double = 0.06, minSignalRemaining: Double = 0.05, enablePcan: Bool = false, pcanStrength: Double = 0.95, pcanOffset: Double = 80, gainBits: Int64 = 21, enableLog: Bool = true, scaleShift: Int64 = 6, leftContext: Int64 = 0, rightContext: Int64 = 0, frameStride: Int64 = 1, zeroPadding: Bool = false, outScale: Int64 = 1) -> Tensor<OutType>
```

This Op converts a sequence of audio data into one or more
feature vectors containing filterbanks of the input. The
conversion process uses a lightweight library to perform:

1.  A slicing window function
2.  Short-time FFTs
3.  Filterbank calculations
4.  Noise reduction
5.  PCAN Auto Gain Control
6.  Logarithmic scaling

Arguments
audio: 1D Tensor, int16 audio data in temporal ordering.
sample\_rate: Integer, the sample rate of the audio in Hz.
window\_size: Integer, length of desired time frames in ms.
window\_step: Integer, length of step size for the next frame in ms.
num\_channels: Integer, the number of filterbank channels to use.
upper\_band\_limit: Float, the highest frequency included in the filterbanks.
lower\_band\_limit: Float, the lowest frequency included in the filterbanks.
smoothing\_bits: Int, scale up signal by 2^(smoothing\_bits) before reduction.
even\_smoothing: Float, smoothing coefficient for even-numbered channels.
odd\_smoothing: Float, smoothing coefficient for odd-numbered channels.
min\_signal\_remaining: Float, fraction of signal to preserve in smoothing.
enable\_pcan: Bool, enable PCAN auto gain control.
pcan\_strength: Float, gain normalization exponent.
pcan\_offset: Float, positive value added in the normalization denominator.
gain\_bits: Int, number of fractional bits in the gain.
enable\_log: Bool, enable logarithmic scaling of filterbanks.
scale\_shift: Integer, scale filterbanks by 2^(scale\_shift).
left\_context: Integer, number of preceding frames to attach to each frame.
right\_context: Integer, number of preceding frames to attach to each frame.
frame\_stride: Integer, M frames to skip over, where output\[n\] = frame\[n\*M\].
zero\_padding: Bool, if left/right context is out-of-bounds, attach frame of
zeroes. Otherwise, frame\[0\] or frame\[size-1\] will be copied.
out\_scale: Integer, divide all filterbanks by this number.
out\_type: DType, type of the output Tensor, defaults to UINT16.

Returns
filterbanks: 2D Tensor, each row is a time frame, each column is a channel.

### `audioSpectrogram(_:windowSize:stride:magnitudeSquared:)`

Produces a visualization of audio data over time.

``` swift
@inlinable @inline(__always) public static func audioSpectrogram(_ input: Tensor<Float>, windowSize: Int64, stride: Int64, magnitudeSquared: Bool = false) -> Tensor<Float>
```

Spectrograms are a standard way of representing audio information as a series of
slices of frequency information, one slice for each window of time. By joining
these together into a sequence, they form a distinctive fingerprint of the sound
over time.

This op expects to receive audio data as an input, stored as floats in the range
\-1 to 1, together with a window width in samples, and a stride specifying how
far to move the window between slices. From this it generates a three
dimensional output. The first dimension is for the channels in the input, so a
stereo audio input would have two here for example. The second dimension is time,
with successive frequency slices. The third dimension has an amplitude value for
each frequency during that time slice.

This means the layout when converted and saved as an image is rotated 90 degrees
clockwise from a typical spectrogram. Time is descending down the Y axis, and
the frequency decreases from left to right.

Each value in the result represents the square root of the sum of the real and
imaginary parts of an FFT on the current window of samples. In this way, the
lowest dimension represents the power of each frequency in the current window,
and adjacent windows are concatenated in the next dimension.

To get a more intuitive and visual look at what this operation does, you can run
tensorflow/examples/wav\_to\_spectrogram to read in an audio file and save out the
resulting spectrogram as a PNG image.

#### Parameters

  - input: - input: Float representation of audio data.

### `audioSummary(tag:_:sampleRate:maxOutputs:)`

Outputs a `Summary` protocol buffer with audio.

``` swift
@inlinable @inline(__always) public static func audioSummary(tag: StringTensor, _ tensor: Tensor<Float>, sampleRate: Double, maxOutputs: Int64 = 3) -> StringTensor
```

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size, frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

#### Parameters

  - tag: - tag: Scalar. Used to build the `tag` attribute of the summary values.
  - tensor: - tensor: 2-D of shape `[batch_size, frames]`.

### `audioSummaryV2(tag:_:sampleRate:maxOutputs:)`

Outputs a `Summary` protocol buffer with audio.

``` swift
@inlinable @inline(__always) public static func audioSummaryV2(tag: StringTensor, _ tensor: Tensor<Float>, sampleRate: Tensor<Float>, maxOutputs: Int64 = 3) -> StringTensor
```

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size, frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

#### Parameters

  - tag: - tag: Scalar. Used to build the `tag` attribute of the summary values.
  - tensor: - tensor: 2-D of shape `[batch_size, frames]`.

### `autoShardDataset(inputDataset:numWorkers:index:autoShardPolicy:outputTypes:outputShapes:)`

Creates a dataset that shards the input dataset.

``` swift
@inlinable @inline(__always) public static func autoShardDataset(inputDataset: VariantHandle, numWorkers: Tensor<Int64>, index: Tensor<Int64>, autoShardPolicy: Int64 = 0, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

Creates a dataset that shards the input dataset by num\_workers, returning a
sharded dataset for the index-th worker. This attempts to automatically shard
a dataset by examining the Dataset graph and inserting a shard op before the
inputs to a reader Dataset (e.g. CSVDataset, TFRecordDataset).

This dataset will throw a NotFound error if we cannot shard the dataset
automatically.

#### Parameters

  - index: - index: A scalar representing the index of the current worker out of num\_workers.

### `avgPool(value:ksize:strides:padding:dataFormat:)`

Performs average pooling on the input.

``` swift
@inlinable @inline(__always) public static func avgPool<T: FloatingPoint & TensorFlowScalar>(value: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, dataFormat: DataFormat = .nhwc) -> Tensor<T>
```

Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.

#### Parameters

  - value: - value: 4-D with shape `[batch, height, width, channels]`.

### `avgPool3D(_:ksize:strides:padding:dataFormat:)`

Performs 3D average pooling on the input.

``` swift
@inlinable @inline(__always) public static func avgPool3D<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, dataFormat: DataFormat1 = .ndhwc) -> Tensor<T>
```

#### Parameters

  - input: - input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.

### `avgPool3DGrad(origInputShape:grad:ksize:strides:padding:dataFormat:)`

Computes gradients of average pooling function.

``` swift
@inlinable @inline(__always) public static func avgPool3DGrad<T: FloatingPoint & TensorFlowScalar>(origInputShape: Tensor<Int32>, grad: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, dataFormat: DataFormat1 = .ndhwc) -> Tensor<T>
```

#### Parameters

  - grad: - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.

### `avgPoolGrad(origInputShape:grad:ksize:strides:padding:dataFormat:)`

Computes gradients of the average pooling function.

``` swift
@inlinable @inline(__always) public static func avgPoolGrad<T: FloatingPoint & TensorFlowScalar>(origInputShape: Tensor<Int32>, grad: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, dataFormat: DataFormat = .nhwc) -> Tensor<T>
```

#### Parameters

  - grad: - grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the output of `avg_pool`.

### `b()`

``` swift
@inlinable @inline(__always) public static func b() -> Tensor<Float>
```

### `batch(inTensors:numBatchThreads:maxBatchSize:maxEnqueuedBatches:batchTimeoutMicros:allowedBatchSizes:gradTimeoutMicros:container:sharedName:batchingQueue:)`

Batches all input tensors nondeterministically.

``` swift
@inlinable @inline(__always) public static func batch<T: TensorArrayProtocol>(inTensors: T, numBatchThreads: Int64, maxBatchSize: Int64, maxEnqueuedBatches: Int64 = 10, batchTimeoutMicros: Int64, allowedBatchSizes: [Int32], gradTimeoutMicros: Int64, container: String, sharedName: String, batchingQueue: String) -> (batchedTensors: T, batchIndex: Tensor<Int64>, id: Tensor<Int64>)
```

When many instances of this Op are being run concurrently with the same
container/shared\_name in the same device, some will output zero-shaped Tensors
and others will output Tensors of size up to max\_batch\_size.

All Tensors in in\_tensors are batched together (so, for example, labels and
features should be batched with a single instance of this operation.

Each invocation of batch emits an `id` scalar which will be used to identify
this particular invocation when doing unbatch or its gradient.

Each op which emits a non-empty batch will also emit a non-empty batch\_index
Tensor, which, is a \[K, 3\] matrix where each row contains the invocation's id,
start, and length of elements of each set of Tensors present in batched\_tensors.

Batched tensors are concatenated along the first dimension, and all tensors in
in\_tensors must have the first dimension of the same size.

in\_tensors: The tensors to be batched.
num\_batch\_threads: Number of scheduling threads for processing batches of work.
Determines the number of batches processed in parallel.
max\_batch\_size: Batch sizes will never be bigger than this.
batch\_timeout\_micros: Maximum number of microseconds to wait before outputting
an incomplete batch.
allowed\_batch\_sizes: Optional list of allowed batch sizes. If left empty, does
nothing. Otherwise, supplies a list of batch sizes, causing the op to pad
batches up to one of those sizes. The entries must increase monotonically, and
the final entry must equal max\_batch\_size.
grad\_timeout\_micros: The timeout to use for the gradient. See Unbatch.
batched\_tensors: Either empty tensors or a batch of concatenated Tensors.
batch\_index: If out\_tensors is non-empty, has information to invert it.
container: Controls the scope of sharing of this batch.
id: always contains a scalar with a unique ID for this invocation of Batch.
shared\_name: Concurrently running instances of batch in the same device with the
same container and shared\_name will batch their elements together. If left
empty, the op name will be used as the shared name.
T: the types of tensors to be batched.

### `batchCholesky(_:)`

``` swift
@inlinable @inline(__always) public static func batchCholesky<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

### `batchCholeskyGrad(l:grad:)`

``` swift
@inlinable @inline(__always) public static func batchCholeskyGrad<T: FloatingPoint & TensorFlowScalar>(l: Tensor<T>, grad: Tensor<T>) -> Tensor<T>
```

### `batchDataset(inputDataset:batchSize:outputTypes:outputShapes:)`

Creates a dataset that batches `batch_size` elements from `input_dataset`.

``` swift
@inlinable @inline(__always) public static func batchDataset(inputDataset: VariantHandle, batchSize: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `batchDatasetV2(inputDataset:batchSize:dropRemainder:parallelCopy:outputTypes:outputShapes:)`

Creates a dataset that batches `batch_size` elements from `input_dataset`.

``` swift
@inlinable @inline(__always) public static func batchDatasetV2(inputDataset: VariantHandle, batchSize: Tensor<Int64>, dropRemainder: Tensor<Bool>, parallelCopy: Bool = false, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `batchFunction(inTensors:capturedTensors:f:numBatchThreads:maxBatchSize:batchTimeoutMicros:maxEnqueuedBatches:allowedBatchSizes:container:sharedName:batchingQueue:)`

Batches all the inputs tensors to the computation done by the function.

``` swift
@inlinable @inline(__always) public static func batchFunction<FIn: TensorGroup, FOut: TensorGroup, Tin: TensorArrayProtocol, Tcaptured: TensorArrayProtocol, Tout: TensorGroup>(inTensors: Tin, capturedTensors: Tcaptured, f: (FIn) -> FOut, numBatchThreads: Int64, maxBatchSize: Int64, batchTimeoutMicros: Int64, maxEnqueuedBatches: Int64 = 10, allowedBatchSizes: [Int32], container: String, sharedName: String, batchingQueue: String) -> Tout
```

So, for example, in the following code

``` python

# This input will be captured.
y = tf.placeholder_with_default(1.0, shape=[])

@tf.Defun(tf.float32)
def computation(a):
  return tf.matmul(a, a) + y

b = gen_batch_ops.batch_function(
        f=computation
        in_tensors=[a],
        captured_tensors=computation.captured_inputs,
        Tout=[o.type for o in computation.definition.signature.output_arg],
        num_batch_threads=1,
        max_batch_size=10,
        batch_timeout_micros=100000,  # 100ms
        allowed_batch_sizes=[3, 10],
        batching_queue="")

If more than one session.run call is simultaneously trying to compute `b`
the values of `a` will be gathered, non-deterministically concatenated
along the first axis, and only one thread will run the computation.

Assumes that all arguments of the function are Tensors which will be batched
along their first dimension.

Arguments that are captured, are not batched. The session.run call which does
the concatenation, will use the values of the captured tensors available to it.
Therefore, typical uses of captured tensors should involve values which remain
unchanged across session.run calls. Inference is a good example of this.

SparseTensor is not supported. The return value of the decorated function
must be a Tensor or a list/tuple of Tensors.

- Parameters:
  - in_tensors: The tensors to be batched.
  - captured_tensors: The tensors which are captured in the function, and don't need
      to be batched.

- Attrs:
  - num_batch_threads: Number of scheduling threads for processing batches of work.
      Determines the number of batches processed in parallel.
  - max_batch_size: Batch sizes will never be bigger than this.
  - batch_timeout_micros: Maximum number of microseconds to wait before outputting
      an incomplete batch.
  - max_enqueued_batches: Maximum number of batches enqueued. Default: 10.
  - allowed_batch_sizes: Optional list of allowed batch sizes. If left empty, does
      nothing. Otherwise, supplies a list of batch sizes, causing the op to pad
      batches up to one of those sizes. The entries must increase monotonically, and
      the final entry must equal max_batch_size.
  - container: Controls the scope of sharing of this batch.
  - shared_name: Concurrently running instances of batch in the same device with the
      same container and shared_name will batch their elements together. If left
      empty, the op name will be used as the shared name.
  - Tin: the types of tensors to be batched.
  - Tcaptured: the types of the captured tensors.
  - Tout: the types of the output tensors.

- Output out_tensors: The output tensors.
```

### `batchMatMul(_:_:adjX:adjY:)`

Multiplies slices of two tensors in batches.

``` swift
@inlinable @inline(__always) public static func batchMatMul<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>, adjX: Bool = false, adjY: Bool = false) -> Tensor<T>
```

Multiplies all slices of `Tensor` `x` and `y` (each slice can be
viewed as an element of a batch), and arranges the individual results
in a single output tensor of the same batch size. Each of the
individual slices can optionally be adjointed (to adjoint a matrix
means to transpose and conjugate it) before multiplication by setting
the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
and `[..., r_y, c_y]`.

The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

``` 
r_o = c_x if adj_x else r_x
c_o = r_y if adj_y else c_y
```

It is computed as:

``` 
output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
```

#### Parameters

  - x: - x: 2-D or higher with shape `[..., r_x, c_x]`.
  - y: - y: 2-D or higher with shape `[..., r_y, c_y]`.

### `batchMatMulV2(_:_:adjX:adjY:)`

Multiplies slices of two tensors in batches.

``` swift
@inlinable @inline(__always) public static func batchMatMulV2<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>, adjX: Bool = false, adjY: Bool = false) -> Tensor<T>
```

Multiplies all slices of `Tensor` `x` and `y` (each slice can be
viewed as an element of a batch), and arranges the individual results
in a single output tensor of the same batch size. Each of the
individual slices can optionally be adjointed (to adjoint a matrix
means to transpose and conjugate it) before multiplication by setting
the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
and `[..., r_y, c_y]`.

The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

``` 
r_o = c_x if adj_x else r_x
c_o = r_y if adj_y else c_y
```

It is computed as:

``` 
output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
```

*NOTE*: `BatchMatMulV2` supports broadcasting in the batch dimensions. More
about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

#### Parameters

  - x: - x: 2-D or higher with shape `[..., r_x, c_x]`.
  - y: - y: 2-D or higher with shape `[..., r_y, c_y]`.

### `batchMatrixBandPart(_:numLower:numUpper:)`

``` swift
@inlinable @inline(__always) public static func batchMatrixBandPart<T: TensorFlowScalar>(_ input: Tensor<T>, numLower: Tensor<Int64>, numUpper: Tensor<Int64>) -> Tensor<T>
```

### `batchMatrixDeterminant(_:)`

``` swift
@inlinable @inline(__always) public static func batchMatrixDeterminant<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

### `batchMatrixDiag(diagonal:)`

``` swift
@inlinable @inline(__always) public static func batchMatrixDiag<T: TensorFlowScalar>(diagonal: Tensor<T>) -> Tensor<T>
```

### `batchMatrixDiagPart(_:)`

``` swift
@inlinable @inline(__always) public static func batchMatrixDiagPart<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

### `batchMatrixInverse(_:adjoint:)`

``` swift
@inlinable @inline(__always) public static func batchMatrixInverse<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, adjoint: Bool = false) -> Tensor<T>
```

### `batchMatrixSetDiag(_:diagonal:)`

``` swift
@inlinable @inline(__always) public static func batchMatrixSetDiag<T: TensorFlowScalar>(_ input: Tensor<T>, diagonal: Tensor<T>) -> Tensor<T>
```

### `batchMatrixSolve(matrix:rhs:adjoint:)`

``` swift
@inlinable @inline(__always) public static func batchMatrixSolve<T: FloatingPoint & TensorFlowScalar>(matrix: Tensor<T>, rhs: Tensor<T>, adjoint: Bool = false) -> Tensor<T>
```

### `batchMatrixSolveLs(matrix:rhs:l2Regularizer:fast:)`

``` swift
@inlinable @inline(__always) public static func batchMatrixSolveLs<T: FloatingPoint & TensorFlowScalar>(matrix: Tensor<T>, rhs: Tensor<T>, l2Regularizer: Tensor<Double>, fast: Bool = true) -> Tensor<T>
```

### `batchMatrixTriangularSolve(matrix:rhs:lower:adjoint:)`

``` swift
@inlinable @inline(__always) public static func batchMatrixTriangularSolve<T: FloatingPoint & TensorFlowScalar>(matrix: Tensor<T>, rhs: Tensor<T>, lower: Bool = true, adjoint: Bool = false) -> Tensor<T>
```

### `batchNormWithGlobalNormalization(t:m:v:beta:gamma:varianceEpsilon:scaleAfterNormalization:)`

Batch normalization.

``` swift
@inlinable @inline(__always) public static func batchNormWithGlobalNormalization<T: TensorFlowNumeric>(t: Tensor<T>, m: Tensor<T>, v: Tensor<T>, beta: Tensor<T>, gamma: Tensor<T>, varianceEpsilon: Double, scaleAfterNormalization: Bool) -> Tensor<T>
```

This op is deprecated. Prefer `tf.nn.batch_normalization`.

#### Parameters

  - t: - t: A 4D input Tensor.
  - m: - m: A 1D mean Tensor with size matching the last dimension of t. This is the first output from tf.nn.moments, or a saved moving average thereof.
  - v: - v: A 1D variance Tensor with size matching the last dimension of t. This is the second output from tf.nn.moments, or a saved moving average thereof.
  - beta: - beta: A 1D beta Tensor with size matching the last dimension of t. An offset to be added to the normalized tensor.
  - gamma: - gamma: A 1D gamma Tensor with size matching the last dimension of t. If "scale\_after\_normalization" is true, this tensor will be multiplied with the normalized tensor.

### `batchNormWithGlobalNormalizationGrad(t:m:v:gamma:backprop:varianceEpsilon:scaleAfterNormalization:)`

Gradients for batch normalization.

``` swift
@inlinable @inline(__always) public static func batchNormWithGlobalNormalizationGrad<T: TensorFlowNumeric>(t: Tensor<T>, m: Tensor<T>, v: Tensor<T>, gamma: Tensor<T>, backprop: Tensor<T>, varianceEpsilon: Double, scaleAfterNormalization: Bool) -> (dx: Tensor<T>, dm: Tensor<T>, dv: Tensor<T>, db: Tensor<T>, dg: Tensor<T>)
```

This op is deprecated. See `tf.nn.batch_normalization`.

#### Parameters

  - t: - t: A 4D input Tensor.
  - m: - m: A 1D mean Tensor with size matching the last dimension of t. This is the first output from tf.nn.moments, or a saved moving average thereof.
  - v: - v: A 1D variance Tensor with size matching the last dimension of t. This is the second output from tf.nn.moments, or a saved moving average thereof.
  - gamma: - gamma: A 1D gamma Tensor with size matching the last dimension of t. If "scale\_after\_normalization" is true, this Tensor will be multiplied with the normalized Tensor.
  - backprop: - backprop: 4D backprop Tensor.

### `batchSelfAdjointEig(_:)`

``` swift
@inlinable @inline(__always) public static func batchSelfAdjointEig<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

### `batchSelfAdjointEigV2(_:computeV:)`

``` swift
@inlinable @inline(__always) public static func batchSelfAdjointEigV2<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, computeV: Bool = true) -> (e: Tensor<T>, v: Tensor<T>)
```

### `batchSvd(_:computeUv:fullMatrices:)`

``` swift
@inlinable @inline(__always) public static func batchSvd<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, computeUv: Bool = true, fullMatrices: Bool = false) -> (s: Tensor<T>, u: Tensor<T>, v: Tensor<T>)
```

### `batchToSpace(_:crops:blockSize:)`

BatchToSpace for 4-D tensors of type T.

``` swift
@inlinable @inline(__always) public static func batchToSpace<T: TensorFlowScalar, Tidx: TensorFlowIndex>(_ input: Tensor<T>, crops: Tensor<Tidx>, blockSize: Int64) -> Tensor<T>
```

This is a legacy version of the more general BatchToSpaceND.

Rearranges (permutes) data from batch into blocks of spatial data, followed by
cropping. This is the reverse transformation of SpaceToBatch. More specifically,
this op outputs a copy of the input tensor where values from the `batch`
dimension are moved in spatial blocks to the `height` and `width` dimensions,
followed by cropping along the `height` and `width` dimensions.

#### Parameters

  - input: - input: 4-D tensor with shape `[batch*block_size*block_size, height_pad/block_size, width_pad/block_size, depth]`. Note that the batch size of the input tensor must be divisible by `block_size * block_size`.
  - crops: - crops: \`\`\`
    crops = \[\[crop\_top, crop\_bottom\], \[crop\_left, crop\_right\]\]
    ``` 
    ```

### `batchToSpaceND(_:blockShape:crops:)`

BatchToSpace for N-D tensors of type T.

``` swift
@inlinable @inline(__always) public static func batchToSpaceND<T: TensorFlowScalar, TblockShape: TensorFlowIndex, Tcrops: TensorFlowIndex>(_ input: Tensor<T>, blockShape: Tensor<TblockShape>, crops: Tensor<Tcrops>) -> Tensor<T>
```

This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
`block_shape + [batch]`, interleaves these blocks back into the grid defined by
the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
the input.  The spatial dimensions of this intermediate result are then
optionally cropped according to `crops` to produce the output.  This is the
reverse of SpaceToBatch.  See below for a precise description.

#### Parameters

  - input: - input: N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`, where spatial\_shape has M dimensions.
  - crops: - crops: This operation is equivalent to the following steps:
    1.  Reshape `input` to `reshaped` of shape:
        \[block\_shape\[0\], ..., block\_shape\[M-1\],
        batch / prod(block\_shape),
        input\_shape\[1\], ..., input\_shape\[N-1\]\]
    
    2.  Permute dimensions of `reshaped` to produce `permuted` of shape
        \[batch / prod(block\_shape),
        
        input\_shape\[1\], block\_shape\[0\],
        ...,
        input\_shape\[M\], block\_shape\[M-1\],
        
        input\_shape\[M+1\], ..., input\_shape\[N-1\]\]
    
    3.  Reshape `permuted` to produce `reshaped_permuted` of shape
        \[batch / prod(block\_shape),
        
        input\_shape\[1\] \* block\_shape\[0\],
        ...,
        input\_shape\[M\] \* block\_shape\[M-1\],
        
        input\_shape\[M+1\],
        ...,
        input\_shape\[N-1\]\]
    
    4.  Crop the start and end of dimensions `[1, ..., M]` of
        `reshaped_permuted` according to `crops` to produce the output of shape:
        \[batch / prod(block\_shape),
        
        input\_shape\[1\] \* block\_shape\[0\] - crops\[0,0\] - crops\[0,1\],
        ...,
        input\_shape\[M\] \* block\_shape\[M-1\] - crops\[M-1,0\] - crops\[M-1,1\],
        
        input\_shape\[M+1\], ..., input\_shape\[N-1\]\]
    Some examples:
    (1) For the following input of shape `[4, 1, 1, 1]`, `block_shape = [2, 2]`, and
    `crops = [[0, 0], [0, 0]]`:
    ``` 
    [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    ```
    The output tensor has shape `[1, 2, 2, 1]` and value:
    ``` 
    x = [[[[1], [2]], [[3], [4]]]]
    ```
    (2) For the following input of shape `[4, 1, 1, 3]`, `block_shape = [2, 2]`, and
    `crops = [[0, 0], [0, 0]]`:
    ``` 
    [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
    ```
    The output tensor has shape `[1, 2, 2, 3]` and value:
    ``` 
    x = [[[[1, 2, 3], [4, 5, 6]],
          [[7, 8, 9], [10, 11, 12]]]]
    ```
    (3) For the following input of shape `[4, 2, 2, 1]`, `block_shape = [2, 2]`, and
    `crops = [[0, 0], [0, 0]]`:
    ``` 
    x = [[[[1], [3]], [[9], [11]]],
         [[[2], [4]], [[10], [12]]],
         [[[5], [7]], [[13], [15]]],
         [[[6], [8]], [[14], [16]]]]
    ```
    The output tensor has shape `[1, 4, 4, 1]` and value:
    ``` 
    x = [[[[1],   [2],  [3],  [4]],
         [[5],   [6],  [7],  [8]],
         [[9],  [10], [11],  [12]],
         [[13], [14], [15],  [16]]]]
    ```
    (4) For the following input of shape `[8, 1, 3, 1]`, `block_shape = [2, 2]`, and
    `crops = [[0, 0], [2, 0]]`:
    ``` 
    x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
         [[[0], [2], [4]]], [[[0], [10], [12]]],
         [[[0], [5], [7]]], [[[0], [13], [15]]],
         [[[0], [6], [8]]], [[[0], [14], [16]]]]
    ```
    The output tensor has shape `[2, 2, 4, 1]` and value:
    ``` 
    x = [[[[1],   [2],  [3],  [4]],
          [[5],   [6],  [7],  [8]]],
         [[[9],  [10], [11],  [12]],
          [[13], [14], [15],  [16]]]]
    ```

### `besselI0e(_:)`

Computes the Bessel i0e function of `x` element-wise.

``` swift
@inlinable @inline(__always) public static func besselI0e<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Exponentially scaled modified Bessel function of order 0 defined as
`bessel_i0e(x) = exp(-abs(x)) bessel_i0(x)`.

This function is faster and numerically stabler than `bessel_i0(x)`.

### `besselI1e(_:)`

Computes the Bessel i1e function of `x` element-wise.

``` swift
@inlinable @inline(__always) public static func besselI1e<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Exponentially scaled modified Bessel function of order 0 defined as
`bessel_i1e(x) = exp(-abs(x)) bessel_i1(x)`.

This function is faster and numerically stabler than `bessel_i1(x)`.

### `betainc(_:_:_:)`

Compute the regularized incomplete beta integral \\(I\_x(a, b)\\).

``` swift
@inlinable @inline(__always) public static func betainc<T: FloatingPoint & TensorFlowScalar>(_ a: Tensor<T>, _ b: Tensor<T>, _ x: Tensor<T>) -> Tensor<T>
```

The regularized incomplete beta integral is defined as:

\\(I\_x(a, b) = \\frac{B(x; a, b)}{B(a, b)}\\)

where

\\(B(x; a, b) = \\int\_0^x t^{a-1} (1 - t)^{b-1} dt\\)

is the incomplete beta function and \\(B(a, b)\\) is the *complete*
beta function.

### `biasAdd(value:bias:dataFormat:)`

Adds `bias` to `value`.

``` swift
@inlinable @inline(__always) public static func biasAdd<T: TensorFlowNumeric>(value: Tensor<T>, bias: Tensor<T>, dataFormat: DataFormat = .nhwc) -> Tensor<T>
```

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.

#### Parameters

  - value: - value: Any number of dimensions.
  - bias: - bias: 1-D with size the last dimension of `value`.

### `biasAddGrad(outBackprop:dataFormat:)`

The backward operation for "BiasAdd" on the "bias" tensor.

``` swift
@inlinable @inline(__always) public static func biasAddGrad<T: TensorFlowNumeric>(outBackprop: Tensor<T>, dataFormat: DataFormat = .nhwc) -> Tensor<T>
```

It accumulates all the values from out\_backprop into the feature dimension.
For NHWC data format, the feature dimension is the last. For NCHW data format,
the feature dimension is the third-to-last.

### `biasAddV1(value:bias:)`

Adds `bias` to `value`.

``` swift
@inlinable @inline(__always) public static func biasAddV1<T: TensorFlowNumeric>(value: Tensor<T>, bias: Tensor<T>) -> Tensor<T>
```

This is a deprecated version of BiasAdd and will be soon removed.

This is a special case of `tf.add` where `bias` is restricted to be 1-D.
Broadcasting is supported, so `value` may have any number of dimensions.

#### Parameters

  - value: - value: Any number of dimensions.
  - bias: - bias: 1-D with size the last dimension of `value`.

### `binary(_:_:)`

``` swift
@inlinable @inline(__always) public static func binary<T: TensorFlowScalar>(_ a: Tensor<T>, _ b: Tensor<T>) -> Tensor<T>
```

### `bincount(arr:size:weights:)`

Counts the number of occurrences of each value in an integer array.

``` swift
@inlinable @inline(__always) public static func bincount<T: TensorFlowNumeric>(arr: Tensor<Int32>, size: Tensor<Int32>, weights: Tensor<T>) -> Tensor<T>
```

Outputs a vector with length `size` and the same dtype as `weights`. If
`weights` are empty, then index `i` stores the number of times the value `i` is
counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
the value in `weights` at each index where the corresponding value in `arr` is
`i`.

Values in `arr` outside of the range \[0, size) are ignored.

#### Parameters

  - arr: - arr: int32 `Tensor`.
  - size: - size: non-negative int32 scalar `Tensor`.
  - weights: - weights: is an int32, int64, float32, or float64 `Tensor` with the same shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights equal to 1.

### `bitcast(_:)`

Bitcasts a tensor from one type to another without copying data.

``` swift
@inlinable @inline(__always) public static func bitcast<T: TensorFlowNumeric, Type: TensorFlowNumeric>(_ input: Tensor<T>) -> Tensor<Type>
```

Given a tensor `input`, this operation returns a tensor that has the same buffer
data as `input` with datatype `type`.

If the input datatype `T` is larger than the output datatype `type` then the
shape changes from \[...\] to \[..., sizeof(`T`)/sizeof(`type`)\].

If `T` is smaller than `type`, the operator requires that the rightmost
dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
\[..., sizeof(`type`)/sizeof(`T`)\] to \[...\].

tf.bitcast() and tf.cast() work differently when real dtype is casted as a complex dtype
(e.g. tf.complex64 or tf.complex128) as tf.cast() make imaginary part 0 while tf.bitcast()
gives module error.
For example,

Example 1:

> > > a = \[1., 2., 3.\]
> > > equality\_bitcast = tf.bitcast(a, tf.complex128)
> > > Traceback (most recent call last):
> > > ...
> > > InvalidArgumentError: Cannot bitcast from 1 to 18 \[Op:Bitcast\]
> > > equality\_cast = tf.cast(a, tf.complex128)
> > > print(equality\_cast)
> > > tf.Tensor(\[1.+0.j 2.+0.j 3.+0.j\], shape=(3,), dtype=complex128)

Example 2:

> > > tf.bitcast(tf.constant(0xffffffff, dtype=tf.uint32), tf.uint8)
> > > \<tf.Tensor: shape=(4,), dtype=uint8, numpy=array(\[255, 255, 255, 255\], dtype=uint8)\>

Example 3:

> > > x = \[1., 2., 3.\]
> > > y = \[0., 2., 3.\]
> > > equality= tf.equal(x,y)
> > > equality\_cast = tf.cast(equality,tf.float32)
> > > equality\_bitcast = tf.bitcast(equality\_cast,tf.uint8)
> > > print(equality)
> > > tf.Tensor(\[False True True\], shape=(3,), dtype=bool)
> > > print(equality\_cast)
> > > tf.Tensor(\[0. 1. 1.\], shape=(3,), dtype=float32)
> > > print(equality\_bitcast)
> > > tf.Tensor(
> > > \[\[  0   0   0   0\]
> > > \[  0   0 128  63\]
> > > \[  0   0 128  63\]\], shape=(3, 4), dtype=uint8)

*NOTE*: Bitcast is implemented as a low-level cast, so machines with different
endian orderings will give different results.

### `bitwiseAnd(_:_:)`

Elementwise computes the bitwise AND of `x` and `y`.

``` swift
@inlinable @inline(__always) public static func bitwiseAnd<T: TensorFlowInteger>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

The result will have those bits set, that are set in both `x` and `y`. The
computation is performed on the underlying representations of `x` and `y`.

For example:

``` python
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64,
              tf.uint8, tf.uint16, tf.uint32, tf.uint64]
 
for dtype in dtype_list:
  lhs = tf.constant([0, 5, 3, 14], dtype=dtype)
  rhs = tf.constant([5, 0, 7, 11], dtype=dtype)
  exp = tf.constant([0, 0, 3, 10], dtype=tf.float32)
 
  res = bitwise_ops.bitwise_and(lhs, rhs)
  tf.assert_equal(tf.cast(res, tf.float32), exp) # TRUE
```

### `bitwiseOr(_:_:)`

Elementwise computes the bitwise OR of `x` and `y`.

``` swift
@inlinable @inline(__always) public static func bitwiseOr<T: TensorFlowInteger>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

The result will have those bits set, that are set in `x`, `y` or both. The
computation is performed on the underlying representations of `x` and `y`.

For example:

``` python
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64,
              tf.uint8, tf.uint16, tf.uint32, tf.uint64]
 
for dtype in dtype_list:
  lhs = tf.constant([0, 5, 3, 14], dtype=dtype)
  rhs = tf.constant([5, 0, 7, 11], dtype=dtype)
  exp = tf.constant([5, 5, 7, 15], dtype=tf.float32)
 
  res = bitwise_ops.bitwise_or(lhs, rhs)
  tf.assert_equal(tf.cast(res,  tf.float32), exp)  # TRUE
```

### `bitwiseXor(_:_:)`

Elementwise computes the bitwise XOR of `x` and `y`.

``` swift
@inlinable @inline(__always) public static func bitwiseXor<T: TensorFlowInteger>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

The result will have those bits set, that are different in `x` and `y`. The
computation is performed on the underlying representations of `x` and `y`.

For example:

``` python
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64,
              tf.uint8, tf.uint16, tf.uint32, tf.uint64]
 
for dtype in dtype_list:
  lhs = tf.constant([0, 5, 3, 14], dtype=dtype)
  rhs = tf.constant([5, 0, 7, 11], dtype=dtype)
  exp = tf.constant([5, 5, 4, 5],  dtype=tf.float32)
 
  res = bitwise_ops.bitwise_xor(lhs, rhs)
  tf.assert_equal(tf.cast(res, tf.float32), exp) # TRUE
```

### `blockLSTM(seqLenMax:_:csPrev:hPrev:w:wci:wcf:wco:_:forgetBias:cellClip:usePeephole:)`

Computes the LSTM cell forward propagation for all the time steps.

``` swift
@inlinable @inline(__always) public static func blockLSTM<T: FloatingPoint & TensorFlowScalar>(seqLenMax: Tensor<Int64>, _ x: Tensor<T>, csPrev: Tensor<T>, hPrev: Tensor<T>, w: Tensor<T>, wci: Tensor<T>, wcf: Tensor<T>, wco: Tensor<T>, _ b: Tensor<T>, forgetBias: Double = 1, cellClip: Double = 3, usePeephole: Bool = false) -> (
    i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>,
    h: Tensor<T>
  )
```

This is equivalent to applying LSTMBlockCell in a loop, like so:

``` python
for x1 in unpack(x):
  i1, cs1, f1, o1, ci1, co1, h1 = LSTMBlock(
    x1, cs_prev, h_prev, w, wci, wcf, wco, b)
  cs_prev = cs1
  h_prev = h1
  i.append(i1)
  cs.append(cs1)
  f.append(f1)
  o.append(o1)
  ci.append(ci1)
  co.append(co1)
  h.append(h1)
return pack(i), pack(cs), pack(f), pack(o), pack(ci), pack(ch), pack(h)
```

#### Parameters

  - x: - x: The sequence input to the LSTM, shape (timelen, batch\_size, num\_inputs).
  - w: - w: The weight matrix.
  - wci: - wci: The weight matrix for input gate peephole connection.
  - wcf: - wcf: The weight matrix for forget gate peephole connection.
  - wco: - wco: The weight matrix for output gate peephole connection.
  - b: - b: The bias vector.

### `blockLSTMGrad(seqLenMax:_:csPrev:hPrev:w:wci:wcf:wco:_:i:cs:f:o:ci:co:h:csGrad:hGrad:usePeephole:)`

Computes the LSTM cell backward propagation for the entire time sequence.

``` swift
@inlinable @inline(__always) public static func blockLSTMGrad<T: FloatingPoint & TensorFlowScalar>(seqLenMax: Tensor<Int64>, _ x: Tensor<T>, csPrev: Tensor<T>, hPrev: Tensor<T>, w: Tensor<T>, wci: Tensor<T>, wcf: Tensor<T>, wco: Tensor<T>, _ b: Tensor<T>, i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>, h: Tensor<T>, csGrad: Tensor<T>, hGrad: Tensor<T>, usePeephole: Bool) -> (
    xGrad: Tensor<T>, csPrevGrad: Tensor<T>, hPrevGrad: Tensor<T>, wGrad: Tensor<T>,
    wciGrad: Tensor<T>, wcfGrad: Tensor<T>, wcoGrad: Tensor<T>, bGrad: Tensor<T>
  )
```

This implementation is to be used in conjunction of LSTMBlock.

#### Parameters

  - x: - x: The sequence input to the LSTM, shape (timelen, batch\_size, num\_inputs).
  - w: - w: The weight matrix.
  - wci: - wci: The weight matrix for input gate peephole connection.
  - wcf: - wcf: The weight matrix for forget gate peephole connection.
  - wco: - wco: The weight matrix for output gate peephole connection.
  - b: - b: The bias vector.
  - i: - i: The input gate over the whole time sequence.
  - cs: - cs: The cell state before the tanh over the whole time sequence.
  - f: - f: The forget gate over the whole time sequence.
  - o: - o: The output gate over the whole time sequence.
  - ci: - ci: The cell input over the whole time sequence.
  - co: - co: The cell after the tanh over the whole time sequence.
  - h: - h: The output h vector over the whole time sequence.

### `blockLSTMGradV2(seqLenMax:_:csPrev:hPrev:w:wci:wcf:wco:_:i:cs:f:o:ci:co:h:csGrad:hGrad:usePeephole:)`

Computes the LSTM cell backward propagation for the entire time sequence.

``` swift
@inlinable @inline(__always) public static func blockLSTMGradV2<T: FloatingPoint & TensorFlowScalar>(seqLenMax: Tensor<Int64>, _ x: Tensor<T>, csPrev: Tensor<T>, hPrev: Tensor<T>, w: Tensor<T>, wci: Tensor<T>, wcf: Tensor<T>, wco: Tensor<T>, _ b: Tensor<T>, i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>, h: Tensor<T>, csGrad: Tensor<T>, hGrad: Tensor<T>, usePeephole: Bool) -> (
    xGrad: Tensor<T>, csPrevGrad: Tensor<T>, hPrevGrad: Tensor<T>, wGrad: Tensor<T>,
    wciGrad: Tensor<T>, wcfGrad: Tensor<T>, wcoGrad: Tensor<T>, bGrad: Tensor<T>
  )
```

This implementation is to be used in conjunction of BlockLSTMV2.

#### Parameters

  - x: - x: The sequence input to the LSTM, shape (timelen, batch\_size, num\_inputs).
  - w: - w: The weight matrix.
  - wci: - wci: The weight matrix for input gate peephole connection.
  - wcf: - wcf: The weight matrix for forget gate peephole connection.
  - wco: - wco: The weight matrix for output gate peephole connection.
  - b: - b: The bias vector.
  - i: - i: The input gate over the whole time sequence.
  - cs: - cs: The cell state before the tanh over the whole time sequence.
  - f: - f: The forget gate over the whole time sequence.
  - o: - o: The output gate over the whole time sequence.
  - ci: - ci: The cell input over the whole time sequence.
  - co: - co: The cell after the tanh over the whole time sequence.
  - h: - h: The output h vector over the whole time sequence.

### `blockLSTMV2(seqLenMax:_:csPrev:hPrev:w:wci:wcf:wco:_:cellClip:usePeephole:)`

Computes the LSTM cell forward propagation for all the time steps.

``` swift
@inlinable @inline(__always) public static func blockLSTMV2<T: FloatingPoint & TensorFlowScalar>(seqLenMax: Tensor<Int64>, _ x: Tensor<T>, csPrev: Tensor<T>, hPrev: Tensor<T>, w: Tensor<T>, wci: Tensor<T>, wcf: Tensor<T>, wco: Tensor<T>, _ b: Tensor<T>, cellClip: Double = 0, usePeephole: Bool = false) -> (
    i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>,
    h: Tensor<T>
  )
```

This is equivalent to applying LSTMBlockCell in a loop, like so:

``` python
for x1 in unpack(x):
  i1, cs1, f1, o1, ci1, co1, h1 = LSTMBlock(
    x1, cs_prev, h_prev, w, wci, wcf, wco, b)
  cs_prev = cs1
  h_prev = h1
  i.append(i1)
  cs.append(cs1)
  f.append(f1)
  o.append(o1)
  ci.append(ci1)
  co.append(co1)
  h.append(h1)
return pack(i), pack(cs), pack(f), pack(o), pack(ci), pack(ch), pack(h)
 
Note that unlike LSTMBlockCell (and BlockLSTM) which uses ICFO gate layout,
this op uses IFCO. So in order for the following snippet to be equivalent
all gate-related outputs should be reordered.
```

#### Parameters

  - x: - x: The sequence input to the LSTM, shape (timelen, batch\_size, num\_inputs).
  - w: - w: The weight matrix.
  - wci: - wci: The weight matrix for input gate peephole connection.
  - wcf: - wcf: The weight matrix for forget gate peephole connection.
  - wco: - wco: The weight matrix for output gate peephole connection.
  - b: - b: The bias vector.

### `boostedTreesAggregateStats(nodeIds:gradients:hessians:feature:maxSplits:numBuckets:)`

Aggregates the summary of accumulated stats for the batch.

``` swift
@inlinable @inline(__always) public static func boostedTreesAggregateStats(nodeIds: Tensor<Int32>, gradients: Tensor<Float>, hessians: Tensor<Float>, feature: Tensor<Int32>, maxSplits: Int64, numBuckets: Int64) -> Tensor<Float>
```

The summary stats contains gradients and hessians accumulated for each node, feature dimension id and bucket.

#### Parameters

  - gradients: - gradients: float32; Rank 2 Tensor (shape=\[batch\_size, logits\_dimension\]) with gradients for each example.
  - hessians: - hessians: float32; Rank 2 Tensor (shape=\[batch\_size, hessian\_dimension\]) with hessians for each example.
  - feature: - feature: int32; Rank 2 feature Tensors (shape=\[batch\_size, feature\_dimension\]).

### `boostedTreesBucketize(floatValues:bucketBoundaries:)`

Bucketize each feature based on bucket boundaries.

``` swift
@inlinable @inline(__always) public static func boostedTreesBucketize(floatValues: [Tensor<Float>], bucketBoundaries: [Tensor<Float>]) -> [Tensor<Int32>]
```

An op that returns a list of float tensors, where each tensor represents the
bucketized values for a single feature.

### `boostedTreesCalculateBestFeatureSplit(nodeIdRange:statsSummary:l1:l2:treeComplexity:minNodeWeight:logitsDimension:splitType:)`

Calculates gains for each feature and returns the best possible split information for the feature.

``` swift
@inlinable @inline(__always) public static func boostedTreesCalculateBestFeatureSplit(nodeIdRange: Tensor<Int32>, statsSummary: Tensor<Float>, l1: Tensor<Float>, l2: Tensor<Float>, treeComplexity: Tensor<Float>, minNodeWeight: Tensor<Float>, logitsDimension: Int64, splitType: SplitType = .inequality) -> (
    nodeIds: Tensor<Int32>, gains: Tensor<Float>, featureDimensions: Tensor<Int32>,
    thresholds: Tensor<Int32>, leftNodeContribs: Tensor<Float>,
    rightNodeContribs: Tensor<Float>, splitWithDefaultDirections: StringTensor
  )
```

The split information is the best threshold (bucket id), gains and left/right node contributions per node for each feature.

It is possible that not all nodes can be split on each feature. Hence, the list of possible nodes can differ between the features. Therefore, we return `node_ids_list` for each feature, containing the list of nodes that this feature can be used to split.

In this manner, the output is the best split per features and per node, so that it needs to be combined later to produce the best split for each node (among all possible features).

The output shapes are compatible in a way that the first dimension of all tensors are the same and equal to the number of possible split nodes for each feature.

#### Parameters

  - l1: - l1: l1 regularization factor on leaf weights, per instance based.
  - l2: - l2: l2 regularization factor on leaf weights, per instance based.

### `boostedTreesCalculateBestGainsPerFeature(nodeIdRange:statsSummaryList:l1:l2:treeComplexity:minNodeWeight:maxSplits:)`

Calculates gains for each feature and returns the best possible split information for the feature.

``` swift
@inlinable @inline(__always) public static func boostedTreesCalculateBestGainsPerFeature(nodeIdRange: Tensor<Int32>, statsSummaryList: [Tensor<Float>], l1: Tensor<Float>, l2: Tensor<Float>, treeComplexity: Tensor<Float>, minNodeWeight: Tensor<Float>, maxSplits: Int64) -> (
    nodeIdsList: [Tensor<Int32>], gainsList: [Tensor<Float>], thresholdsList: [Tensor<Int32>],
    leftNodeContribsList: [Tensor<Float>], rightNodeContribsList: [Tensor<Float>]
  )
```

The split information is the best threshold (bucket id), gains and left/right node contributions per node for each feature.

It is possible that not all nodes can be split on each feature. Hence, the list of possible nodes can differ between the features. Therefore, we return `node_ids_list` for each feature, containing the list of nodes that this feature can be used to split.

In this manner, the output is the best split per features and per node, so that it needs to be combined later to produce the best split for each node (among all possible features).

The length of output lists are all of the same length, `num_features`.
The output shapes are compatible in a way that the first dimension of all tensors of all lists are the same and equal to the number of possible split nodes for each feature.

#### Parameters

  - l1: - l1: l1 regularization factor on leaf weights, per instance based.
  - l2: - l2: l2 regularization factor on leaf weights, per instance based.

### `boostedTreesCenterBias(treeEnsembleHandle:meanGradients:meanHessians:l1:l2:)`

Calculates the prior from the training data (the bias) and fills in the first node with the logits' prior. Returns a boolean indicating whether to continue centering.

``` swift
@inlinable @inline(__always) public static func boostedTreesCenterBias(treeEnsembleHandle: ResourceHandle, meanGradients: Tensor<Float>, meanHessians: Tensor<Float>, l1: Tensor<Float>, l2: Tensor<Float>) -> Tensor<Bool>
```

#### Parameters

  - l1: - l1: l1 regularization factor on leaf weights, per instance based.
  - l2: - l2: l2 regularization factor on leaf weights, per instance based.

### `boostedTreesCreateEnsemble(treeEnsembleHandle:stampToken:treeEnsembleSerialized:)`

Creates a tree ensemble model and returns a handle to it.

``` swift
@inlinable @inline(__always) public static func boostedTreesCreateEnsemble(treeEnsembleHandle: ResourceHandle, stampToken: Tensor<Int64>, treeEnsembleSerialized: StringTensor)
```

### `boostedTreesCreateQuantileStreamResource(quantileStreamResourceHandle:epsilon:numStreams:maxElements:)`

Create the Resource for Quantile Streams.

``` swift
@inlinable @inline(__always) public static func boostedTreesCreateQuantileStreamResource(quantileStreamResourceHandle: ResourceHandle, epsilon: Tensor<Float>, numStreams: Tensor<Int64>, maxElements: Int64 = 1_099_511_627_776)
```

#### Parameters

  - epsilon: - epsilon: float; The required approximation error of the stream resource.

### `boostedTreesDeserializeEnsemble(treeEnsembleHandle:stampToken:treeEnsembleSerialized:)`

Deserializes a serialized tree ensemble config and replaces current tree

``` swift
@inlinable @inline(__always) public static func boostedTreesDeserializeEnsemble(treeEnsembleHandle: ResourceHandle, stampToken: Tensor<Int64>, treeEnsembleSerialized: StringTensor)
```

ensemble.

### `boostedTreesEnsembleResourceHandleOp(container:sharedName:)`

Creates a handle to a BoostedTreesEnsembleResource

``` swift
@inlinable @inline(__always) public static func boostedTreesEnsembleResourceHandleOp(container: String, sharedName: String) -> ResourceHandle
```

### `boostedTreesExampleDebugOutputs(treeEnsembleHandle:bucketizedFeatures:logitsDimension:)`

Debugging/model interpretability outputs for each example.

``` swift
@inlinable @inline(__always) public static func boostedTreesExampleDebugOutputs(treeEnsembleHandle: ResourceHandle, bucketizedFeatures: [Tensor<Int32>], logitsDimension: Int64) -> StringTensor
```

It traverses all the trees and computes debug metrics for individual examples,
such as getting split feature ids and logits after each split along the decision
path used to compute directional feature contributions.

### `boostedTreesFlushQuantileSummaries(quantileStreamResourceHandle:numFeatures:)`

Flush the quantile summaries from each quantile stream resource.

``` swift
@inlinable @inline(__always) public static func boostedTreesFlushQuantileSummaries(quantileStreamResourceHandle: ResourceHandle, numFeatures: Int64) -> [Tensor<Float>]
```

An op that outputs a list of quantile summaries of a quantile stream resource.
Each summary Tensor is rank 2, containing summaries (value, weight, min\_rank,
max\_rank) for a single feature.

### `boostedTreesGetEnsembleStates(treeEnsembleHandle:)`

Retrieves the tree ensemble resource stamp token, number of trees and growing statistics.

``` swift
@inlinable @inline(__always) public static func boostedTreesGetEnsembleStates(treeEnsembleHandle: ResourceHandle) -> (
    stampToken: Tensor<Int64>, numTrees: Tensor<Int32>, numFinalizedTrees: Tensor<Int32>,
    numAttemptedLayers: Tensor<Int32>, lastLayerNodesRange: Tensor<Int32>
  )
```

### `boostedTreesMakeQuantileSummaries(floatValues:exampleWeights:epsilon:)`

Makes the summary of quantiles for the batch.

``` swift
@inlinable @inline(__always) public static func boostedTreesMakeQuantileSummaries(floatValues: [Tensor<Float>], exampleWeights: Tensor<Float>, epsilon: Tensor<Float>) -> [Tensor<Float>]
```

An op that takes a list of tensors (one tensor per feature) and outputs the
quantile summaries for each tensor.

#### Parameters

  - epsilon: - epsilon: float; The required maximum approximation error.

### `boostedTreesMakeStatsSummary(nodeIds:gradients:hessians:bucketizedFeaturesList:maxSplits:numBuckets:)`

Makes the summary of accumulated stats for the batch.

``` swift
@inlinable @inline(__always) public static func boostedTreesMakeStatsSummary(nodeIds: Tensor<Int32>, gradients: Tensor<Float>, hessians: Tensor<Float>, bucketizedFeaturesList: [Tensor<Int32>], maxSplits: Int64, numBuckets: Int64) -> Tensor<Float>
```

The summary stats contains gradients and hessians accumulated into the corresponding node and bucket for each example.

#### Parameters

  - gradients: - gradients: float32; Rank 2 Tensor (shape=\[\#examples, 1\]) for gradients.
  - hessians: - hessians: float32; Rank 2 Tensor (shape=\[\#examples, 1\]) for hessians.

### `boostedTreesPredict(treeEnsembleHandle:bucketizedFeatures:logitsDimension:)`

Runs multiple additive regression ensemble predictors on input instances and

``` swift
@inlinable @inline(__always) public static func boostedTreesPredict(treeEnsembleHandle: ResourceHandle, bucketizedFeatures: [Tensor<Int32>], logitsDimension: Int64) -> Tensor<Float>
```

computes the logits. It is designed to be used during prediction.
It traverses all the trees and calculates the final score for each instance.

### `boostedTreesQuantileStreamResourceAddSummaries(quantileStreamResourceHandle:summaries:)`

Add the quantile summaries to each quantile stream resource.

``` swift
@inlinable @inline(__always) public static func boostedTreesQuantileStreamResourceAddSummaries(quantileStreamResourceHandle: ResourceHandle, summaries: [Tensor<Float>])
```

An op that adds a list of quantile summaries to a quantile stream resource. Each
summary Tensor is rank 2, containing summaries (value, weight, min\_rank, max\_rank)
for a single feature.

#### Parameters

  - summaries: - summaries: string; List of Rank 2 Tensor each containing the summaries for a single feature.

### `boostedTreesQuantileStreamResourceDeserialize(quantileStreamResourceHandle:bucketBoundaries:)`

Deserialize bucket boundaries and ready flag into current QuantileAccumulator.

``` swift
@inlinable @inline(__always) public static func boostedTreesQuantileStreamResourceDeserialize(quantileStreamResourceHandle: ResourceHandle, bucketBoundaries: [Tensor<Float>])
```

An op that deserializes bucket boundaries and are boundaries ready flag into current QuantileAccumulator.

### `boostedTreesQuantileStreamResourceFlush(quantileStreamResourceHandle:numBuckets:generateQuantiles:)`

Flush the summaries for a quantile stream resource.

``` swift
@inlinable @inline(__always) public static func boostedTreesQuantileStreamResourceFlush(quantileStreamResourceHandle: ResourceHandle, numBuckets: Tensor<Int64>, generateQuantiles: Bool = false)
```

An op that flushes the summaries for a quantile stream resource.

### `boostedTreesQuantileStreamResourceGetBucketBoundaries(quantileStreamResourceHandle:numFeatures:)`

Generate the bucket boundaries for each feature based on accumulated summaries.

``` swift
@inlinable @inline(__always) public static func boostedTreesQuantileStreamResourceGetBucketBoundaries(quantileStreamResourceHandle: ResourceHandle, numFeatures: Int64) -> [Tensor<Float>]
```

An op that returns a list of float tensors for a quantile stream resource. Each
tensor is Rank 1 containing bucket boundaries for a single feature.

### `boostedTreesQuantileStreamResourceHandleOp(container:sharedName:)`

Creates a handle to a BoostedTreesQuantileStreamResource.

``` swift
@inlinable @inline(__always) public static func boostedTreesQuantileStreamResourceHandleOp(container: String, sharedName: String) -> ResourceHandle
```

### `boostedTreesSerializeEnsemble(treeEnsembleHandle:)`

Serializes the tree ensemble to a proto.

``` swift
@inlinable @inline(__always) public static func boostedTreesSerializeEnsemble(treeEnsembleHandle: ResourceHandle) -> (stampToken: Tensor<Int64>, treeEnsembleSerialized: StringTensor)
```

### `boostedTreesSparseAggregateStats(nodeIds:gradients:hessians:featureIndices:featureValues:featureShape:maxSplits:numBuckets:)`

Aggregates the summary of accumulated stats for the batch.

``` swift
@inlinable @inline(__always) public static func boostedTreesSparseAggregateStats(nodeIds: Tensor<Int32>, gradients: Tensor<Float>, hessians: Tensor<Float>, featureIndices: Tensor<Int32>, featureValues: Tensor<Int32>, featureShape: Tensor<Int32>, maxSplits: Int64, numBuckets: Int64) -> (
    statsSummaryIndices: Tensor<Int32>, statsSummaryValues: Tensor<Float>,
    statsSummaryShape: Tensor<Int32>
  )
```

The summary stats contains gradients and hessians accumulated for each node, bucket and dimension id.

#### Parameters

  - gradients: - gradients: float32; Rank 2 Tensor (shape=\[batch\_size, logits\_dimension\]) with gradients for each example.
  - hessians: - hessians: float32; Rank 2 Tensor (shape=\[batch\_size, hessian\_dimension\]) with hessians for each example.

### `boostedTreesSparseCalculateBestFeatureSplit(nodeIdRange:statsSummaryIndices:statsSummaryValues:statsSummaryShape:l1:l2:treeComplexity:minNodeWeight:logitsDimension:splitType:)`

Calculates gains for each feature and returns the best possible split information for the feature.

``` swift
@inlinable @inline(__always) public static func boostedTreesSparseCalculateBestFeatureSplit(nodeIdRange: Tensor<Int32>, statsSummaryIndices: Tensor<Int32>, statsSummaryValues: Tensor<Float>, statsSummaryShape: Tensor<Int32>, l1: Tensor<Float>, l2: Tensor<Float>, treeComplexity: Tensor<Float>, minNodeWeight: Tensor<Float>, logitsDimension: Int64, splitType: SplitType2 = .inequality) -> (
    nodeIds: Tensor<Int32>, gains: Tensor<Float>, featureDimensions: Tensor<Int32>,
    thresholds: Tensor<Int32>, leftNodeContribs: Tensor<Float>,
    rightNodeContribs: Tensor<Float>, splitWithDefaultDirections: StringTensor
  )
```

The split information is the best threshold (bucket id), gains and left/right node contributions per node for each feature.

It is possible that not all nodes can be split on each feature. Hence, the list of possible nodes can differ between the features. Therefore, we return `node_ids_list` for each feature, containing the list of nodes that this feature can be used to split.

In this manner, the output is the best split per features and per node, so that it needs to be combined later to produce the best split for each node (among all possible features).

The output shapes are compatible in a way that the first dimension of all tensors are the same and equal to the number of possible split nodes for each feature.

#### Parameters

  - l1: - l1: l1 regularization factor on leaf weights, per instance based.
  - l2: - l2: l2 regularization factor on leaf weights, per instance based.

### `boostedTreesTrainingPredict(treeEnsembleHandle:cachedTreeIds:cachedNodeIds:bucketizedFeatures:logitsDimension:)`

Runs multiple additive regression ensemble predictors on input instances and

``` swift
@inlinable @inline(__always) public static func boostedTreesTrainingPredict(treeEnsembleHandle: ResourceHandle, cachedTreeIds: Tensor<Int32>, cachedNodeIds: Tensor<Int32>, bucketizedFeatures: [Tensor<Int32>], logitsDimension: Int64) -> (partialLogits: Tensor<Float>, treeIds: Tensor<Int32>, nodeIds: Tensor<Int32>)
```

computes the update to cached logits. It is designed to be used during training.
It traverses the trees starting from cached tree id and cached node id and
calculates the updates to be pushed to the cache.

### `boostedTreesUpdateEnsemble(treeEnsembleHandle:featureIds:nodeIds:gains:thresholds:leftNodeContribs:rightNodeContribs:maxDepth:learningRate:pruningMode:)`

Updates the tree ensemble by either adding a layer to the last tree being grown

``` swift
@inlinable @inline(__always) public static func boostedTreesUpdateEnsemble(treeEnsembleHandle: ResourceHandle, featureIds: Tensor<Int32>, nodeIds: [Tensor<Int32>], gains: [Tensor<Float>], thresholds: [Tensor<Int32>], leftNodeContribs: [Tensor<Float>], rightNodeContribs: [Tensor<Float>], maxDepth: Tensor<Int32>, learningRate: Tensor<Float>, pruningMode: Int64)
```

or by starting a new tree.

#### Parameters

  - gains: - gains: List of rank 1 tensors representing the gains for each of the feature's split.
  - thresholds: - thresholds: List of rank 1 tensors representing the thesholds for each of the feature's split.

### `boostedTreesUpdateEnsembleV2(treeEnsembleHandle:featureIds:dimensionIds:nodeIds:gains:thresholds:leftNodeContribs:rightNodeContribs:splitTypes:maxDepth:learningRate:pruningMode:logitsDimension:)`

Updates the tree ensemble by adding a layer to the last tree being grown

``` swift
@inlinable @inline(__always) public static func boostedTreesUpdateEnsembleV2(treeEnsembleHandle: ResourceHandle, featureIds: Tensor<Int32>, dimensionIds: [Tensor<Int32>], nodeIds: [Tensor<Int32>], gains: [Tensor<Float>], thresholds: [Tensor<Int32>], leftNodeContribs: [Tensor<Float>], rightNodeContribs: [Tensor<Float>], splitTypes: [StringTensor], maxDepth: Tensor<Int32>, learningRate: Tensor<Float>, pruningMode: Tensor<Int32>, logitsDimension: Int64 = 1)
```

or by starting a new tree.

#### Parameters

  - gains: - gains: List of rank 1 tensors representing the gains for each of the feature's split.
  - thresholds: - thresholds: List of rank 1 tensors representing the thesholds for each of the feature's split.

### `broadcastArgs(s0:s1:)`

Return the shape of s0 op s1 with broadcast.

``` swift
@inlinable @inline(__always) public static func broadcastArgs<T: TensorFlowIndex>(s0: Tensor<T>, s1: Tensor<T>) -> Tensor<T>
```

Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.

### `broadcastGradientArgs(s0:s1:)`

Return the reduction indices for computing gradients of s0 op s1 with broadcast.

``` swift
@inlinable @inline(__always) public static func broadcastGradientArgs<T: TensorFlowIndex>(s0: Tensor<T>, s1: Tensor<T>) -> (r0: Tensor<T>, r1: Tensor<T>)
```

This is typically used by gradient computations for a broadcasting operation.

### `broadcastTo(_:shape:)`

Broadcast an array for a compatible shape.

``` swift
@inlinable @inline(__always) public static func broadcastTo<T: TensorFlowScalar, Tidx: TensorFlowIndex>(_ input: Tensor<T>, shape: Tensor<Tidx>) -> Tensor<T>
```

Broadcasting is the process of making arrays to have compatible shapes
for arithmetic operations. Two shapes are compatible if for each
dimension pair they are either equal or one of them is one. When trying
to broadcast a Tensor to a shape, it starts with the trailing dimensions,
and works its way forward.

For example,

> > > x = tf.constant(\[1, 2, 3\])
> > > y = tf.broadcast\_to(x, \[3, 3\])
> > > print(y)
> > > tf.Tensor(
> > > \[\[1 2 3\]
> > > \[1 2 3\]
> > > \[1 2 3\]\], shape=(3, 3), dtype=int32)

In the above example, the input Tensor with the shape of `[1, 3]`
is broadcasted to output Tensor with shape of `[3, 3]`.

#### Parameters

  - input: - input: A Tensor to broadcast.
  - shape: - shape: An 1-D `int` Tensor. The shape of the desired output.

### `bucketize(_:boundaries:)`

Bucketizes 'input' based on 'boundaries'.

``` swift
@inlinable @inline(__always) public static func bucketize<T: TensorFlowNumeric>(_ input: Tensor<T>, boundaries: [Double]) -> Tensor<Int32>
```

For example, if the inputs are
boundaries = \[0, 10, 100\]
input = \[\[-5, 10000\]
\[150,   10\]
\[5,    100\]\]

then the output will be
output = \[\[0, 3\]
\[3, 2\]
\[1, 3\]\]

#### Parameters

  - input: - input: Any shape of Tensor contains with int or float type.

### `bytesProducedStatsDataset(inputDataset:tag:outputTypes:outputShapes:)`

Records the bytes size of each element of `input_dataset` in a StatsAggregator.

``` swift
@inlinable @inline(__always) public static func bytesProducedStatsDataset(inputDataset: VariantHandle, tag: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `cSRSparseMatrixComponents(csrSparseMatrix:index:)`

Reads out the CSR components at batch `index`.

``` swift
@inlinable @inline(__always) public static func cSRSparseMatrixComponents<Type: FloatingPoint & TensorFlowScalar>(csrSparseMatrix: VariantHandle, index: Tensor<Int32>) -> (rowPtrs: Tensor<Int32>, colInds: Tensor<Int32>, values: Tensor<Type>)
```

This op is meant only for debugging / testing, and its interface is not expected
to be stable.

#### Parameters

  - index: - index: The index in `csr_sparse_matrix`'s batch.

### `cSRSparseMatrixToDense(sparseInput:)`

Convert a (possibly batched) CSRSparseMatrix to dense.

``` swift
@inlinable @inline(__always) public static func cSRSparseMatrixToDense<Type: FloatingPoint & TensorFlowScalar>(sparseInput: VariantHandle) -> Tensor<Type>
```

### `cSRSparseMatrixToSparseTensor(sparseMatrix:)`

Converts a (possibly batched) CSRSparesMatrix to a SparseTensor.

``` swift
@inlinable @inline(__always) public static func cSRSparseMatrixToSparseTensor<Type: FloatingPoint & TensorFlowScalar>(sparseMatrix: VariantHandle) -> (indices: Tensor<Int64>, values: Tensor<Type>, denseShape: Tensor<Int64>)
```

### `cSVDataset(filenames:compressionType:bufferSize:header:fieldDelim:useQuoteDelim:naValue:selectCols:recordDefaults:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func cSVDataset<OutputTypes: TensorArrayProtocol>(filenames: StringTensor, compressionType: StringTensor, bufferSize: Tensor<Int64>, header: Tensor<Bool>, fieldDelim: StringTensor, useQuoteDelim: Tensor<Bool>, naValue: StringTensor, selectCols: Tensor<Int64>, recordDefaults: OutputTypes, outputShapes: [TensorShape?]) -> VariantHandle
```

### `cTCBeamSearchDecoder(inputs:sequenceLength:beamWidth:topPaths:mergeRepeated:)`

Performs beam search decoding on the logits given in input.

``` swift
@inlinable @inline(__always) public static func cTCBeamSearchDecoder<T: FloatingPoint & TensorFlowScalar>(inputs: Tensor<T>, sequenceLength: Tensor<Int32>, beamWidth: Int64, topPaths: Int64, mergeRepeated: Bool = true) -> (
    decodedIndices: [Tensor<Int64>], decodedValues: [Tensor<Int64>],
    decodedShape: [Tensor<Int64>], logProbability: Tensor<T>
  )
```

A note about the attribute merge\_repeated: For the beam search decoder,
this means that if consecutive entries in a beam are the same, only
the first of these is emitted.  That is, when the top path is "A B B B B",
"A B" is returned if merge\_repeated = True but "A B B B B" is
returned if merge\_repeated = False.

#### Parameters

  - inputs: - inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.

### `cTCGreedyDecoder(inputs:sequenceLength:mergeRepeated:)`

Performs greedy decoding on the logits given in inputs.

``` swift
@inlinable @inline(__always) public static func cTCGreedyDecoder<T: FloatingPoint & TensorFlowScalar>(inputs: Tensor<T>, sequenceLength: Tensor<Int32>, mergeRepeated: Bool = false) -> (
    decodedIndices: Tensor<Int64>, decodedValues: Tensor<Int64>, decodedShape: Tensor<Int64>,
    logProbability: Tensor<T>
  )
```

A note about the attribute merge\_repeated: if enabled, when
consecutive logits' maximum indices are the same, only the first of
these is emitted.  Labeling the blank '\*', the sequence "A B B \* B B"
becomes "A B B" if merge\_repeated = True and "A B B B B" if
merge\_repeated = False.

Regardless of the value of merge\_repeated, if the maximum index of a given
time and batch corresponds to the blank, index `(num_classes - 1)`, no new
element is emitted.

#### Parameters

  - inputs: - inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.

### `cTCLoss(inputs:labelsIndices:labelsValues:sequenceLength:preprocessCollapseRepeated:ctcMergeRepeated:ignoreLongerOutputsThanInputs:)`

Calculates the CTC Loss (log probability) for each batch entry.  Also calculates

``` swift
@inlinable @inline(__always) public static func cTCLoss<T: FloatingPoint & TensorFlowScalar>(inputs: Tensor<T>, labelsIndices: Tensor<Int64>, labelsValues: Tensor<Int32>, sequenceLength: Tensor<Int32>, preprocessCollapseRepeated: Bool = false, ctcMergeRepeated: Bool = true, ignoreLongerOutputsThanInputs: Bool = false) -> (loss: Tensor<T>, gradient: Tensor<T>)
```

the gradient.  This class performs the softmax operation for you, so inputs
should be e.g. linear projections of outputs by an LSTM.

#### Parameters

  - inputs: - inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.

### `cacheDataset(inputDataset:filename:outputTypes:outputShapes:)`

Creates a dataset that caches elements from `input_dataset`.

``` swift
@inlinable @inline(__always) public static func cacheDataset(inputDataset: VariantHandle, filename: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

A CacheDataset will iterate over the input\_dataset, and store tensors. If the
cache already exists, the cache will be used. If the cache is inappropriate
(e.g. cannot be opened, contains tensors of the wrong shape / size), an error
will the returned when used.

#### Parameters

  - filename: - filename: A path on the filesystem where we should cache the dataset. Note: this will be a directory.

### `cacheDatasetV2(inputDataset:filename:cache:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func cacheDatasetV2(inputDataset: VariantHandle, filename: StringTensor, cache: ResourceHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `cast(_:truncate:)`

Cast x of type SrcT to y of DstT.

``` swift
@inlinable @inline(__always) public static func cast<Srct: TensorFlowScalar, Dstt: TensorFlowScalar>(_ x: Tensor<Srct>, truncate: Bool = false) -> Tensor<Dstt>
```

### `ceil(_:)`

Returns element-wise smallest integer not less than x.

``` swift
@inlinable @inline(__always) public static func ceil<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

### `checkNumerics(_:message:)`

Checks a tensor for NaN and Inf values.

``` swift
@inlinable @inline(__always) public static func checkNumerics<T: FloatingPoint & TensorFlowScalar>(_ tensor: Tensor<T>, message: String) -> Tensor<T>
```

When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

### `cholesky(_:)`

Computes the Cholesky decomposition of one or more square matrices.

``` swift
@inlinable @inline(__always) public static func cholesky<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices.

The input has to be symmetric and positive definite. Only the lower-triangular
part of the input will be used for this operation. The upper-triangular part
will not be read.

The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

**Note**: The gradient computation on GPU is faster for large matrices but
not for large batch dimensions when the submatrices are small. In this
case it might be faster to use the CPU.

#### Parameters

  - input: - input: Shape is `[..., M, M]`.

### `choleskyGrad(l:grad:)`

Computes the reverse mode backpropagated gradient of the Cholesky algorithm.

``` swift
@inlinable @inline(__always) public static func choleskyGrad<T: FloatingPoint & TensorFlowScalar>(l: Tensor<T>, grad: Tensor<T>) -> Tensor<T>
```

For an explanation see "Differentiation of the Cholesky algorithm" by
Iain Murray http://arxiv.org/abs/1602.07527.

#### Parameters

  - l: - l: Output of batch Cholesky algorithm l = cholesky(A). Shape is `[..., M, M]`. Algorithm depends only on lower triangular part of the innermost matrices of this tensor.
  - grad: - grad: df/dl where f is some scalar function. Shape is `[..., M, M]`. Algorithm depends only on lower triangular part of the innermost matrices of this tensor.

### `chooseFastestDataset(inputDatasets:numExperiments:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func chooseFastestDataset(inputDatasets: [VariantHandle], numExperiments: Int64, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `clipByValue(t:clipValueMin:clipValueMax:)`

Clips tensor values to a specified min and max.

``` swift
@inlinable @inline(__always) public static func clipByValue<T: TensorFlowNumeric>(t: Tensor<T>, clipValueMin: Tensor<T>, clipValueMax: Tensor<T>) -> Tensor<T>
```

Given a tensor `t`, this operation returns a tensor of the same type and
shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
Any values less than `clip_value_min` are set to `clip_value_min`. Any values
greater than `clip_value_max` are set to `clip_value_max`.

#### Parameters

  - t: - t: A `Tensor`.

### `closeSummaryWriter(writer:)`

``` swift
@inlinable @inline(__always) public static func closeSummaryWriter(writer: ResourceHandle)
```

### `collectiveBcastRecv(groupSize:groupKey:instanceKey:shape:communicationHint:)`

Receives a tensor value broadcast from another device.

``` swift
@inlinable @inline(__always) public static func collectiveBcastRecv<T: TensorFlowNumeric>(groupSize: Int64, groupKey: Int64, instanceKey: Int64, shape: TensorShape?, communicationHint: String = "auto") -> Tensor<T>
```

### `collectiveBcastSend(_:groupSize:groupKey:instanceKey:shape:communicationHint:)`

Broadcasts a tensor value to one or more other devices.

``` swift
@inlinable @inline(__always) public static func collectiveBcastSend<T: TensorFlowNumeric>(_ input: Tensor<T>, groupSize: Int64, groupKey: Int64, instanceKey: Int64, shape: TensorShape?, communicationHint: String = "auto") -> Tensor<T>
```

### `collectiveGather(_:groupSize:groupKey:instanceKey:shape:communicationHint:)`

Mutually accumulates multiple tensors of identical type and shape.

``` swift
@inlinable @inline(__always) public static func collectiveGather<T: TensorFlowNumeric>(_ input: Tensor<T>, groupSize: Int64, groupKey: Int64, instanceKey: Int64, shape: TensorShape?, communicationHint: String = "auto") -> Tensor<T>
```

### `collectivePermute(_:sourceTargetPairs:)`

An Op to permute tensors across replicated TPU instances.

``` swift
@inlinable @inline(__always) public static func collectivePermute<T: TensorFlowNumeric>(_ input: Tensor<T>, sourceTargetPairs: Tensor<Int32>) -> Tensor<T>
```

Each instance supplies its own input.

For example, suppose there are 4 TPU instances: `[A, B, C, D]`. Passing
source\_target\_pairs=`[[0,1],[1,2],[2,3],[3,0]]` gets the outputs:
`[D, A, B, C]`.

#### Parameters

  - input: - input: The local input to be permuted. Currently only supports float and bfloat16.

### `collectiveReduce(_:groupSize:groupKey:instanceKey:mergeOp:finalOp:subdivOffsets:waitFor:communicationHint:)`

Mutually reduces multiple tensors of identical type and shape.

``` swift
@inlinable @inline(__always) public static func collectiveReduce<T: TensorFlowNumeric>(_ input: Tensor<T>, groupSize: Int64, groupKey: Int64, instanceKey: Int64, mergeOp: MergeOp, finalOp: FinalOp, subdivOffsets: [Int32], waitFor: [Int32], communicationHint: String = "auto") -> Tensor<T>
```

### `combinedNonMaxSuppression(boxes:scores:maxOutputSizePerClass:maxTotalSize:iouThreshold:scoreThreshold:padPerClass:clipBoxes:)`

Greedily selects a subset of bounding boxes in descending order of score,

``` swift
@inlinable @inline(__always) public static func combinedNonMaxSuppression(boxes: Tensor<Float>, scores: Tensor<Float>, maxOutputSizePerClass: Tensor<Int32>, maxTotalSize: Tensor<Int32>, iouThreshold: Tensor<Float>, scoreThreshold: Tensor<Float>, padPerClass: Bool = false, clipBoxes: Bool = true) -> (
    nmsedBoxes: Tensor<Float>, nmsedScores: Tensor<Float>, nmsedClasses: Tensor<Float>,
    validDetections: Tensor<Int32>
  )
```

This operation performs non\_max\_suppression on the inputs per batch, across
all classes.
Prunes away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes are supplied as
\[y1, x1, y2, x2\], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval \[0, 1\]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system. Also note that
this algorithm is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is the final boxes, scores and classes tensor
returned after performing non\_max\_suppression.

#### Parameters

  - boxes: - boxes: A 4-D float tensor of shape `[batch_size, num_boxes, q, 4]`. If `q` is 1 then same boxes are used for all classes otherwise, if `q` is equal to number of classes, class-specific boxes are used.
  - scores: - scores: A 3-D float tensor of shape `[batch_size, num_boxes, num_classes]` representing a single score corresponding to each box (each row of boxes).

### `compareAndBitpack(_:threshold:)`

Compare values of `input` to `threshold` and pack resulting bits into a `uint8`.

``` swift
@inlinable @inline(__always) public static func compareAndBitpack<T: TensorFlowScalar>(_ input: Tensor<T>, threshold: Tensor<T>) -> Tensor<UInt8>
```

Each comparison returns a boolean `true` (if `input_value > threshold`)
or and `false` otherwise.

This operation is useful for Locality-Sensitive-Hashing (LSH) and other
algorithms that use hashing approximations of cosine and `L2` distances;
codes can be generated from an input via:

``` python
codebook_size = 50
codebook_bits = codebook_size * 32
codebook = tf.get_variable('codebook', [x.shape[-1].value, codebook_bits],
                           dtype=x.dtype,
                           initializer=tf.orthogonal_initializer())
codes = compare_and_threshold(tf.matmul(x, codebook), threshold=0.)
codes = tf.bitcast(codes, tf.int32)  # go from uint8 to int32
# now codes has shape x.shape[:-1] + [codebook_size]
```

**NOTE**: Currently, the innermost dimension of the tensor must be divisible
by 8.

Given an `input` shaped `[s0, s1, ..., s_n]`, the output is
a `uint8` tensor shaped `[s0, s1, ..., s_n / 8]`.

#### Parameters

  - input: - input: Values to compare against `threshold` and bitpack.
  - threshold: - threshold: Threshold to compare against.

### `complex(real:imag:)`

Converts two real numbers to a complex number.

``` swift
@inlinable @inline(__always) public static func complex<T: FloatingPoint & TensorFlowScalar, Tout: TensorFlowScalar>(real: Tensor<T>, imag: Tensor<T>) -> Tensor<Tout>
```

Given a tensor `real` representing the real part of a complex number, and a
tensor `imag` representing the imaginary part of a complex number, this
operation returns complex numbers elementwise of the form \\(a + bj\\), where
*a* represents the `real` part and *b* represents the `imag` part.

The input tensors `real` and `imag` must have the same shape.

For example:

``` 
# tensor 'real' is [2.25, 3.25]
# tensor `imag` is [4.75, 5.75]
tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
```

### `complexAbs(_:)`

Computes the complex absolute value of a tensor.

``` swift
@inlinable @inline(__always) public static func complexAbs<T: TensorFlowScalar, Tout: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<Tout>
```

Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` or `double` that is the absolute value of each element in `x`. All
elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
value is computed as \\( \\sqrt{a^2 + b^2}\\).

### `complexStruct(nA:nB:)`

``` swift
@inlinable @inline(__always) public static func complexStruct<TC: TensorGroup>(nA: Int64, nB: Int64) -> (a: [Tensor<Int32>], b: [Tensor<Int64>], c: TC)
```

### `computeAccidentalHits(trueClasses:sampledCandidates:numTrue:seed:seed2:)`

Computes the ids of the positions in sampled\_candidates that match true\_labels.

``` swift
@inlinable @inline(__always) public static func computeAccidentalHits(trueClasses: Tensor<Int64>, sampledCandidates: Tensor<Int64>, numTrue: Int64, seed: Int64 = 0, seed2: Int64 = 0) -> (indices: Tensor<Int32>, ids: Tensor<Int64>, weights: Tensor<Float>)
```

When doing log-odds NCE, the result of this op should be passed through a
SparseToDense op, then added to the logits of the sampled candidates. This has
the effect of 'removing' the sampled labels that match the true labels by
making the classifier sure that they are sampled labels.

### `concat(concatDim:_:)`

Concatenates tensors along one dimension.

``` swift
@inlinable @inline(__always) public static func concat<T: TensorFlowScalar>(concatDim: Tensor<Int32>, _ values: [Tensor<T>]) -> Tensor<T>
```

#### Parameters

  - values: - values: The `N` Tensors to concatenate. Their ranks and types must match, and their sizes must match in all dimensions except `concat_dim`.

### `concatOffset(concatDim:shape:)`

Computes offsets of concat inputs within its output.

``` swift
@inlinable @inline(__always) public static func concatOffset(concatDim: Tensor<Int32>, shape: [Tensor<Int32>]) -> [Tensor<Int32>]
```

For example:

``` 
# 'x' is [2, 2, 7]
# 'y' is [2, 3, 7]
# 'z' is [2, 5, 7]
concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
```

This is typically used by gradient computations for a concat operation.

#### Parameters

  - shape: - shape: The `N` int32 vectors representing shape of tensors being concatenated.

### `concatV2(_:axis:)`

Concatenates tensors along one dimension.

``` swift
@inlinable @inline(__always) public static func concatV2<T: TensorFlowScalar, Tidx: TensorFlowIndex>(_ values: [Tensor<T>], axis: Tensor<Tidx>) -> Tensor<T>
```

#### Parameters

  - values: - values: List of `N` Tensors to concatenate. Their ranks and types must match, and their sizes must match in all dimensions except `concat_dim`.
  - axis: - axis: 0-D.  The dimension along which to concatenate.  Must be in the range \[-rank(values), rank(values)).

### `concatenateDataset(inputDataset:anotherDataset:outputTypes:outputShapes:)`

Creates a dataset that concatenates `input_dataset` with `another_dataset`.

``` swift
@inlinable @inline(__always) public static func concatenateDataset(inputDataset: VariantHandle, anotherDataset: VariantHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `configureDistributedTPU(embeddingConfig:tpuEmbeddingConfig:isGlobalInit:enableWholeMeshCompilations:compilationFailureClosesChips:)`

Sets up the centralized structures for a distributed TPU system.

``` swift
@inlinable @inline(__always) public static func configureDistributedTPU(embeddingConfig: String, tpuEmbeddingConfig: String, isGlobalInit: Bool = false, enableWholeMeshCompilations: Bool = false, compilationFailureClosesChips: Bool = true) -> StringTensor
```

### `configureTPUEmbedding(config:)`

Sets up TPUEmbedding in a distributed TPU system.

``` swift
@inlinable @inline(__always) public static func configureTPUEmbedding(config: String)
```

### `conj(_:)`

Returns the complex conjugate of a complex number.

``` swift
@inlinable @inline(__always) public static func conj<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

Given a tensor `input` of complex numbers, this operation returns a tensor of
complex numbers that are the complex conjugate of each element in `input`. The
complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
real part and *b* is the imaginary part.

The complex conjugate returned by this operation is of the form \\(a - bj\\).

For example:

``` 
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
```

### `conjugateTranspose(_:perm:)`

Shuffle dimensions of x according to a permutation and conjugate the result.

``` swift
@inlinable @inline(__always) public static func conjugateTranspose<T: TensorFlowScalar, Tperm: TensorFlowIndex>(_ x: Tensor<T>, perm: Tensor<Tperm>) -> Tensor<T>
```

The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
`y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
`y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])`

### `constructionFails()`

``` swift
@inlinable @inline(__always) public static func constructionFails()
```

### `consumeMutexLock(mutexLock:)`

This op consumes a lock created by `MutexLock`.

``` swift
@inlinable @inline(__always) public static func consumeMutexLock(mutexLock: VariantHandle)
```

This op exists to consume a tensor created by `MutexLock` (other than
direct control dependencies).  It should be the only that consumes the tensor,
and will raise an error if it is not.  Its only purpose is to keep the
mutex lock tensor alive until it is consumed by this op.

**NOTE**: This operation must run on the same device as its input.  This may
be enforced via the `colocate_with` mechanism.

### `controlTrigger()`

Does nothing. Serves as a control trigger for scheduling.

``` swift
@inlinable @inline(__always) public static func controlTrigger()
```

Only useful as a placeholder for control edges.

### `conv2D(_:filter:strides:useCudnnOnGpu:padding:explicitPaddings:dataFormat:dilations:)`

Computes a 2-D convolution given 4-D `input` and `filter` tensors.

``` swift
@inlinable @inline(__always) public static func conv2D<T: TensorFlowNumeric>(_ input: Tensor<T>, filter: Tensor<T>, strides: [Int32], useCudnnOnGpu: Bool = true, padding: Padding2, explicitPaddings: [Int32], dataFormat: DataFormat = .nhwc, dilations: [Int32] = [1, 1, 1, 1]) -> Tensor<T>
```

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`, this op
performs the following:

1.  Flattens the filter to a 2-D matrix with shape
    `[filter_height * filter_width * in_channels, output_channels]`.
2.  Extracts image patches from the input tensor to form a *virtual*
    tensor of shape `[batch, out_height, out_width, filter_height * filter_width * in_channels]`.
3.  For each patch, right-multiplies the filter matrix and the image patch
    vector.

In detail, with the default NHWC format,

``` 
output[b, i, j, k] =
    sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                    filter[di, dj, q, k]
```

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

#### Parameters

  - input: - input: A 4-D tensor. The dimension order is interpreted according to the value of `data_format`, see below for details.
  - filter: - filter: A 4-D tensor of shape `[filter_height, filter_width, in_channels, out_channels]`

### `conv2DBackpropFilter(_:filterSizes:outBackprop:strides:useCudnnOnGpu:padding:explicitPaddings:dataFormat:dilations:)`

Computes the gradients of convolution with respect to the filter.

``` swift
@inlinable @inline(__always) public static func conv2DBackpropFilter<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, filterSizes: Tensor<Int32>, outBackprop: Tensor<T>, strides: [Int32], useCudnnOnGpu: Bool = true, padding: Padding2, explicitPaddings: [Int32], dataFormat: DataFormat = .nhwc, dilations: [Int32] = [1, 1, 1, 1]) -> Tensor<T>
```

#### Parameters

  - input: - input: 4-D with shape `[batch, in_height, in_width, in_channels]`.

### `conv2DBackpropInput(inputSizes:filter:outBackprop:strides:useCudnnOnGpu:padding:explicitPaddings:dataFormat:dilations:)`

Computes the gradients of convolution with respect to the input.

``` swift
@inlinable @inline(__always) public static func conv2DBackpropInput<T: TensorFlowNumeric>(inputSizes: Tensor<Int32>, filter: Tensor<T>, outBackprop: Tensor<T>, strides: [Int32], useCudnnOnGpu: Bool = true, padding: Padding2, explicitPaddings: [Int32], dataFormat: DataFormat = .nhwc, dilations: [Int32] = [1, 1, 1, 1]) -> Tensor<T>
```

#### Parameters

  - filter: - filter: 4-D with shape `[filter_height, filter_width, in_channels, out_channels]`.

### `conv3D(_:filter:strides:padding:dataFormat:dilations:)`

Computes a 3-D convolution given 5-D `input` and `filter` tensors.

``` swift
@inlinable @inline(__always) public static func conv3D<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, filter: Tensor<T>, strides: [Int32], padding: Padding, dataFormat: DataFormat1 = .ndhwc, dilations: [Int32] = [1, 1, 1, 1, 1]) -> Tensor<T>
```

In signal processing, cross-correlation is a measure of similarity of
two waveforms as a function of a time-lag applied to one of them. This
is also known as a sliding dot product or sliding inner-product.

Our Conv3D implements a form of cross-correlation.

#### Parameters

  - input: - input: Shape `[batch, in_depth, in_height, in_width, in_channels]`.
  - filter: - filter: Shape `[filter_depth, filter_height, filter_width, in_channels, out_channels]`. `in_channels` must match between `input` and `filter`.

### `conv3DBackpropFilter(_:filter:outBackprop:strides:padding:dilations:)`

Computes the gradients of 3-D convolution with respect to the filter.

``` swift
@inlinable @inline(__always) public static func conv3DBackpropFilter<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, filter: Tensor<T>, outBackprop: Tensor<T>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1, 1]) -> Tensor<T>
```

#### Parameters

  - input: - input: Shape `[batch, depth, rows, cols, in_channels]`.
  - filter: - filter: Shape `[depth, rows, cols, in_channels, out_channels]`. `in_channels` must match between `input` and `filter`.

### `conv3DBackpropFilterV2(_:filterSizes:outBackprop:strides:padding:dataFormat:dilations:)`

Computes the gradients of 3-D convolution with respect to the filter.

``` swift
@inlinable @inline(__always) public static func conv3DBackpropFilterV2<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, filterSizes: Tensor<Int32>, outBackprop: Tensor<T>, strides: [Int32], padding: Padding, dataFormat: DataFormat1 = .ndhwc, dilations: [Int32] = [1, 1, 1, 1, 1]) -> Tensor<T>
```

#### Parameters

  - input: - input: Shape `[batch, depth, rows, cols, in_channels]`.

### `conv3DBackpropInput(_:filter:outBackprop:strides:padding:dilations:)`

Computes the gradients of 3-D convolution with respect to the input.

``` swift
@inlinable @inline(__always) public static func conv3DBackpropInput<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, filter: Tensor<T>, outBackprop: Tensor<T>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1, 1]) -> Tensor<T>
```

#### Parameters

  - input: - input: Shape `[batch, depth, rows, cols, in_channels]`.
  - filter: - filter: Shape `[depth, rows, cols, in_channels, out_channels]`. `in_channels` must match between `input` and `filter`.

### `conv3DBackpropInputV2(inputSizes:filter:outBackprop:strides:padding:dataFormat:dilations:)`

Computes the gradients of 3-D convolution with respect to the input.

``` swift
@inlinable @inline(__always) public static func conv3DBackpropInputV2<T: FloatingPoint & TensorFlowScalar, Tshape: TensorFlowIndex>(inputSizes: Tensor<Tshape>, filter: Tensor<T>, outBackprop: Tensor<T>, strides: [Int32], padding: Padding, dataFormat: DataFormat1 = .ndhwc, dilations: [Int32] = [1, 1, 1, 1, 1]) -> Tensor<T>
```

#### Parameters

  - filter: - filter: Shape `[depth, rows, cols, in_channels, out_channels]`. `in_channels` must match between `input` and `filter`.

### `copy(_:tensorName:debugOpsSpec:)`

Copy a tensor from CPU-to-CPU or GPU-to-GPU.

``` swift
@inlinable @inline(__always) public static func copy<T: TensorFlowScalar>(_ input: Tensor<T>, tensorName: String, debugOpsSpec: [String]) -> Tensor<T>
```

Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
device on which the tensor is allocated.
N.B.: If the all downstream attached debug ops are disabled given the current
gRPC gating status, the output will simply forward the input tensor without
deep-copying. See the documentation of Debug\* ops for more details.

Unlike the CopyHost Op, this op does not have HostMemory constraint on its
input or output.

#### Parameters

  - input: - input: Input tensor.

### `copyHost(_:tensorName:debugOpsSpec:)`

Copy a tensor to host.

``` swift
@inlinable @inline(__always) public static func copyHost<T: TensorFlowScalar>(_ input: Tensor<T>, tensorName: String, debugOpsSpec: [String]) -> Tensor<T>
```

Performs CPU-to-CPU deep-copying of tensor.
N.B.: If the all downstream attached debug ops are disabled given the current
gRPC gating status, the output will simply forward the input tensor without
deep-copying. See the documentation of Debug\* ops for more details.

Unlike the Copy Op, this op has HostMemory constraint on its input or output.

#### Parameters

  - input: - input: Input tensor.

### `copyOp(_:)`

``` swift
@inlinable @inline(__always) public static func copyOp<T: TensorFlowScalar>(_ a: Tensor<T>) -> Tensor<T>
```

### `cos(_:)`

Computes cos of x element-wise.

``` swift
@inlinable @inline(__always) public static func cos<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Given an input tensor, this function computes cosine of every
element in the tensor. Input range is `(-inf, inf)` and
output range is `[-1,1]`. If input lies outside the boundary, `nan`
is returned.

``` python
x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
tf.math.cos(x) ==> [nan -0.91113025 0.87758255 0.5403023 0.36235774 0.48718765 -0.95215535 nan]
```

### `cosh(_:)`

Computes hyperbolic cosine of x element-wise.

``` swift
@inlinable @inline(__always) public static func cosh<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Given an input tensor, this function computes hyperbolic cosine of every
element in the tensor. Input range is `[-inf, inf]` and output range
is `[1, inf]`.

``` python
x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
tf.math.cosh(x) ==> [inf 4.0515420e+03 1.1276259e+00 1.5430807e+00 1.8106556e+00 3.7621956e+00 1.1013233e+04 inf]
```

### `createSummaryDbWriter(writer:dbUri:experimentName:runName:userName:)`

``` swift
@inlinable @inline(__always) public static func createSummaryDbWriter(writer: ResourceHandle, dbUri: StringTensor, experimentName: StringTensor, runName: StringTensor, userName: StringTensor)
```

### `createSummaryFileWriter(writer:logdir:maxQueue:flushMillis:filenameSuffix:)`

``` swift
@inlinable @inline(__always) public static func createSummaryFileWriter(writer: ResourceHandle, logdir: StringTensor, maxQueue: Tensor<Int32>, flushMillis: Tensor<Int32>, filenameSuffix: StringTensor)
```

### `cropAndResize(image:boxes:boxInd:cropSize:method:extrapolationValue:)`

Extracts crops from the input image tensor and resizes them.

``` swift
@inlinable @inline(__always) public static func cropAndResize<T: TensorFlowNumeric>(image: Tensor<T>, boxes: Tensor<Float>, boxInd: Tensor<Int32>, cropSize: Tensor<Int32>, method: Method = .bilinear, extrapolationValue: Double = 0) -> Tensor<Float>
```

Extracts crops from the input image tensor and resizes them using bilinear
sampling or nearest neighbor sampling (possibly with aspect ratio change) to a
common output size specified by `crop_size`. This is more general than the
`crop_to_bounding_box` op which extracts a fixed size slice from the input image
and does not allow resizing or aspect ratio change.

Returns a tensor with `crops` from the input `image` at positions defined at the
bounding box locations in `boxes`. The cropped boxes are all resized (with
bilinear or nearest neighbor interpolation) to a fixed
`size = [crop_height, crop_width]`. The result is a 4-D tensor
`[num_boxes, crop_height, crop_width, depth]`. The resizing is corner aligned.
In particular, if `boxes = [[0, 0, 1, 1]]`, the method will give identical
results to using `tf.image.resize_bilinear()` or
`tf.image.resize_nearest_neighbor()`(depends on the `method` argument) with
`align_corners=True`.

#### Parameters

  - image: - image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`. Both `image_height` and `image_width` need to be positive.
  - boxes: - boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor specifies the coordinates of a box in the `box_ind[i]` image and is specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image height is mapped to `[0, image_height - 1]` in image height coordinates. We do allow `y1` \> `y2`, in which case the sampled crop is an up-down flipped version of the original image. The width dimension is treated similarly. Normalized coordinates outside the `[0, 1]` range are allowed, in which case we use `extrapolation_value` to extrapolate the input image values.

### `cropAndResizeGradBoxes(grads:image:boxes:boxInd:method:)`

Computes the gradient of the crop\_and\_resize op wrt the input boxes tensor.

``` swift
@inlinable @inline(__always) public static func cropAndResizeGradBoxes<T: TensorFlowNumeric>(grads: Tensor<Float>, image: Tensor<T>, boxes: Tensor<Float>, boxInd: Tensor<Int32>, method: Method4 = .bilinear) -> Tensor<Float>
```

#### Parameters

  - grads: - grads: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
  - image: - image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`. Both `image_height` and `image_width` need to be positive.
  - boxes: - boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor specifies the coordinates of a box in the `box_ind[i]` image and is specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image height is mapped to `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in which case the sampled crop is an up-down flipped version of the original image. The width dimension is treated similarly. Normalized coordinates outside the `\[0, 1\]`range are allowed, in which case we use`extrapolation\_value\` to extrapolate the input image values.

### `cropAndResizeGradImage(grads:boxes:boxInd:imageSize:method:)`

Computes the gradient of the crop\_and\_resize op wrt the input image tensor.

``` swift
@inlinable @inline(__always) public static func cropAndResizeGradImage<T: FloatingPoint & TensorFlowScalar>(grads: Tensor<Float>, boxes: Tensor<Float>, boxInd: Tensor<Int32>, imageSize: Tensor<Int32>, method: Method = .bilinear) -> Tensor<T>
```

#### Parameters

  - grads: - grads: A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
  - boxes: - boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor specifies the coordinates of a box in the `box_ind[i]` image and is specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image height is mapped to `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in which case the sampled crop is an up-down flipped version of the original image. The width dimension is treated similarly. Normalized coordinates outside the `\[0, 1\]`range are allowed, in which case we use`extrapolation\_value\` to extrapolate the input image values.

### `cross(_:_:)`

Compute the pairwise cross product.

``` swift
@inlinable @inline(__always) public static func cross<T: TensorFlowNumeric>(_ a: Tensor<T>, _ b: Tensor<T>) -> Tensor<T>
```

`a` and `b` must be the same shape; they can either be simple 3-element vectors,
or any shape where the innermost dimension is 3. In the latter case, each pair
of corresponding 3-element vectors is cross-multiplied independently.

#### Parameters

  - a: - a: A tensor containing 3-element vectors.
  - b: - b: Another tensor, of same type and shape as `a`.

### `crossReplicaSum(_:groupAssignment:)`

An Op to sum inputs across replicated TPU instances.

``` swift
@inlinable @inline(__always) public static func crossReplicaSum<T: TensorFlowNumeric>(_ input: Tensor<T>, groupAssignment: Tensor<Int32>) -> Tensor<T>
```

Each instance supplies its own input.

For example, suppose there are 8 TPU instances: `[A, B, C, D, E, F, G, H]`.
Passing group\_assignment=`[[0,2,4,6],[1,3,5,7]]` sets `A, C, E, G` as group 0,
and `B, D, F, H` as group 1. Thus we get the outputs:
`[A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H]`.

#### Parameters

  - input: - input: The local input to the sum.

### `cudnnRNN(_:inputH:inputC:params:rnnMode:inputMode:direction:dropout:seed:seed2:isTraining:)`

A RNN backed by cuDNN.

``` swift
@inlinable @inline(__always) public static func cudnnRNN<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, inputH: Tensor<T>, inputC: Tensor<T>, params: Tensor<T>, rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0, isTraining: Bool = true) -> (output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, reserveSpace: Tensor<T>)
```

Computes the RNN from the input and initial states, with respect to the params
buffer.

rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicate whether there is a linear projection between the input and
the actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used. Should be
"unidirectional" or "bidirectional".
dropout: Dropout probability. When set to 0., dropout is disabled.
seed: The 1st part of a seed to initialize dropout.
seed2: The 2nd part of a seed to initialize dropout.
input: A 3-D tensor with the shape of \[seq\_length, batch\_size, input\_size\].
input\_h: A 3-D tensor with the shape of \[num\_layer \* dir, batch\_size,
num\_units\].
input\_c: For LSTM, a 3-D tensor with the shape of
\[num\_layer \* dir, batch, num\_units\]. For other models, it is ignored.
params: A 1-D tensor that contains the weights and biases in an opaque layout.
The size must be created through CudnnRNNParamsSize, and initialized
separately. Note that they might not be compatible across different
generations. So it is a good idea to save and restore
output: A 3-D tensor with the shape of \[seq\_length, batch\_size,
dir \* num\_units\].
output\_h: The same shape has input\_h.
output\_c: The same shape as input\_c for LSTM. An empty tensor for other models.
is\_training: Indicates whether this operation is used for inferenece or
training.
reserve\_space: An opaque tensor that can be used in backprop calculation. It
is only produced if is\_training is false.

### `cudnnRNNBackprop(_:inputH:inputC:params:output:outputH:outputC:outputBackprop:outputHBackprop:outputCBackprop:reserveSpace:rnnMode:inputMode:direction:dropout:seed:seed2:)`

Backprop step of CudnnRNN.

``` swift
@inlinable @inline(__always) public static func cudnnRNNBackprop<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, inputH: Tensor<T>, inputC: Tensor<T>, params: Tensor<T>, output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, outputBackprop: Tensor<T>, outputHBackprop: Tensor<T>, outputCBackprop: Tensor<T>, reserveSpace: Tensor<T>, rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0) -> (
    inputBackprop: Tensor<T>, inputHBackprop: Tensor<T>, inputCBackprop: Tensor<T>,
    paramsBackprop: Tensor<T>
  )
```

Compute the backprop of both data and weights in a RNN.

rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicate whether there is a linear projection between the input and
the actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used. Should be
"unidirectional" or "bidirectional".
dropout: Dropout probability. When set to 0., dropout is disabled.
seed: The 1st part of a seed to initialize dropout.
seed2: The 2nd part of a seed to initialize dropout.
input: A 3-D tensor with the shape of \[seq\_length, batch\_size, input\_size\].
input\_h: A 3-D tensor with the shape of \[num\_layer \* dir, batch\_size,
num\_units\].
input\_c: For LSTM, a 3-D tensor with the shape of
\[num\_layer \* dir, batch, num\_units\]. For other models, it is ignored.
params: A 1-D tensor that contains the weights and biases in an opaque layout.
The size must be created through CudnnRNNParamsSize, and initialized
separately. Note that they might not be compatible across different
generations. So it is a good idea to save and restore
output: A 3-D tensor with the shape of \[seq\_length, batch\_size,
dir \* num\_units\].
output\_h: The same shape has input\_h.
output\_c: The same shape as input\_c for LSTM. An empty tensor for other models.
output\_backprop: A 3-D tensor with the same shape as output in the forward pass.
output\_h\_backprop: A 3-D tensor with the same shape as output\_h in the forward
pass.
output\_c\_backprop: A 3-D tensor with the same shape as output\_c in the forward
pass.
reserve\_space: The same reserve\_space produced in for forward operation.
input\_backprop: The backprop to input in the forward pass. Has the same shape
as input.
input\_h\_backprop: The backprop to input\_h in the forward pass. Has the same
shape as input\_h.
input\_c\_backprop: The backprop to input\_c in the forward pass. Has the same
shape as input\_c.
params\_backprop: The backprop to the params buffer in the forward pass. Has the
same shape as params.

### `cudnnRNNBackpropV2(_:inputH:inputC:params:output:outputH:outputC:outputBackprop:outputHBackprop:outputCBackprop:reserveSpace:hostReserved:rnnMode:inputMode:direction:dropout:seed:seed2:)`

Backprop step of CudnnRNN.

``` swift
@inlinable @inline(__always) public static func cudnnRNNBackpropV2<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, inputH: Tensor<T>, inputC: Tensor<T>, params: Tensor<T>, output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, outputBackprop: Tensor<T>, outputHBackprop: Tensor<T>, outputCBackprop: Tensor<T>, reserveSpace: Tensor<T>, hostReserved: Tensor<Int8>, rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0) -> (
    inputBackprop: Tensor<T>, inputHBackprop: Tensor<T>, inputCBackprop: Tensor<T>,
    paramsBackprop: Tensor<T>
  )
```

Compute the backprop of both data and weights in a RNN. Takes an extra
"host\_reserved" inupt than CudnnRNNBackprop, which is used to determine RNN
cudnnRNNAlgo\_t and cudnnMathType\_t.

rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicates whether there is a linear projection between the input and
the actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used. Should be
"unidirectional" or "bidirectional".
dropout: Dropout probability. When set to 0., dropout is disabled.
seed: The 1st part of a seed to initialize dropout.
seed2: The 2nd part of a seed to initialize dropout.
input: A 3-D tensor with the shape of \[seq\_length, batch\_size, input\_size\].
input\_h: A 3-D tensor with the shape of \[num\_layer \* dir, batch\_size,
num\_units\].
input\_c: For LSTM, a 3-D tensor with the shape of
\[num\_layer \* dir, batch, num\_units\]. For other models, it is ignored.
params: A 1-D tensor that contains the weights and biases in an opaque layout.
The size must be created through CudnnRNNParamsSize, and initialized
separately. Note that they might not be compatible across different
generations. So it is a good idea to save and restore
output: A 3-D tensor with the shape of \[seq\_length, batch\_size,
dir \* num\_units\].
output\_h: The same shape has input\_h.
output\_c: The same shape as input\_c for LSTM. An empty tensor for other models.
output\_backprop: A 3-D tensor with the same shape as output in the forward pass.
output\_h\_backprop: A 3-D tensor with the same shape as output\_h in the forward
pass.
output\_c\_backprop: A 3-D tensor with the same shape as output\_c in the forward
pass.
reserve\_space: The same reserve\_space produced in the forward operation.
host\_reserved: The same host\_reserved produced in the forward operation.
input\_backprop: The backprop to input in the forward pass. Has the same shape
as input.
input\_h\_backprop: The backprop to input\_h in the forward pass. Has the same
shape as input\_h.
input\_c\_backprop: The backprop to input\_c in the forward pass. Has the same
shape as input\_c.
params\_backprop: The backprop to the params buffer in the forward pass. Has the
same shape as params.

### `cudnnRNNBackpropV3(_:inputH:inputC:params:sequenceLengths:output:outputH:outputC:outputBackprop:outputHBackprop:outputCBackprop:reserveSpace:hostReserved:rnnMode:inputMode:direction:dropout:seed:seed2:numProj:timeMajor:)`

Backprop step of CudnnRNNV3.

``` swift
@inlinable @inline(__always) public static func cudnnRNNBackpropV3<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, inputH: Tensor<T>, inputC: Tensor<T>, params: Tensor<T>, sequenceLengths: Tensor<Int32>, output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, outputBackprop: Tensor<T>, outputHBackprop: Tensor<T>, outputCBackprop: Tensor<T>, reserveSpace: Tensor<T>, hostReserved: Tensor<Int8>, rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0, numProj: Int64 = 0, timeMajor: Bool = true) -> (
    inputBackprop: Tensor<T>, inputHBackprop: Tensor<T>, inputCBackprop: Tensor<T>,
    paramsBackprop: Tensor<T>
  )
```

Compute the backprop of both data and weights in a RNN. Takes an extra
"sequence\_lengths" input than CudnnRNNBackprop.

rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicates whether there is a linear projection between the input and
the actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used. Should be
"unidirectional" or "bidirectional".
dropout: Dropout probability. When set to 0., dropout is disabled.
seed: The 1st part of a seed to initialize dropout.
seed2: The 2nd part of a seed to initialize dropout.
input: If time\_major is true, this is a 3-D tensor with the shape of
\[seq\_length, batch\_size, input\_size\]. If time\_major is false, the shape is
\[batch\_size, seq\_length, input\_size\].
input\_h: If time\_major is true, this is a 3-D tensor with the shape of
\[num\_layer \* dir, batch\_size, num\_units\]. If time\_major is false, the shape
is \[batch\_size, num\_layer \* dir, num\_units\].
input\_c: For LSTM, a 3-D tensor with the shape of
\[num\_layer \* dir, batch, num\_units\]. For other models, it is ignored.
params: A 1-D tensor that contains the weights and biases in an opaque layout.
The size must be created through CudnnRNNParamsSize, and initialized
separately. Note that they might not be compatible across different
generations. So it is a good idea to save and restore
sequence\_lengths: a vector of lengths of each input sequence.
output: If time\_major is true, this is a 3-D tensor with the shape of
\[seq\_length, batch\_size, dir \* num\_units\]. If time\_major is false, the
shape is \[batch\_size, seq\_length, dir \* num\_units\].
output\_h: The same shape has input\_h.
output\_c: The same shape as input\_c for LSTM. An empty tensor for other models.
output\_backprop: A 3-D tensor with the same shape as output in the forward pass.
output\_h\_backprop: A 3-D tensor with the same shape as output\_h in the forward
pass.
output\_c\_backprop: A 3-D tensor with the same shape as output\_c in the forward
pass.
time\_major: Indicates whether the input/output format is time major or batch
major.
reserve\_space: The same reserve\_space produced in the forward operation.
input\_backprop: The backprop to input in the forward pass. Has the same shape
as input.
input\_h\_backprop: The backprop to input\_h in the forward pass. Has the same
shape as input\_h.
input\_c\_backprop: The backprop to input\_c in the forward pass. Has the same
shape as input\_c.
params\_backprop: The backprop to the params buffer in the forward pass. Has the
same shape as params.

### `cudnnRNNCanonicalToParams(numLayers:numUnits:inputSize:weights:biases:rnnMode:inputMode:direction:dropout:seed:seed2:)`

Converts CudnnRNN params from canonical form to usable form.

``` swift
@inlinable @inline(__always) public static func cudnnRNNCanonicalToParams<T: FloatingPoint & TensorFlowScalar>(numLayers: Tensor<Int32>, numUnits: Tensor<Int32>, inputSize: Tensor<Int32>, weights: [Tensor<T>], biases: [Tensor<T>], rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<T>
```

Writes a set of weights into the opaque params buffer so they can be used in
upcoming training or inferences.

Note that the params buffer may not be compatible across different GPUs. So any
save and restoration should be converted to and from the canonical weights and
biases.

num\_layers: Specifies the number of layers in the RNN model.
num\_units: Specifies the size of the hidden state.
input\_size: Specifies the size of the input state.
weights: the canonical form of weights that can be used for saving
and restoration. They are more likely to be compatible across different
generations.
biases: the canonical form of biases that can be used for saving
and restoration. They are more likely to be compatible across different
generations.
num\_params: number of parameter sets for all layers.
Each layer may contain multiple parameter sets, with each set consisting of
a weight matrix and a bias vector.
rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicate whether there is a linear projection between the input and
The actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used.
dir = (direction == bidirectional) ? 2 : 1
dropout: dropout probability. When set to 0., dropout is disabled.
seed: the 1st part of a seed to initialize dropout.
seed2: the 2nd part of a seed to initialize dropout.

### `cudnnRNNCanonicalToParamsV2(numLayers:numUnits:inputSize:weights:biases:rnnMode:inputMode:direction:dropout:seed:seed2:numProj:)`

Converts CudnnRNN params from canonical form to usable form. It supports the projection in LSTM.

``` swift
@inlinable @inline(__always) public static func cudnnRNNCanonicalToParamsV2<T: FloatingPoint & TensorFlowScalar>(numLayers: Tensor<Int32>, numUnits: Tensor<Int32>, inputSize: Tensor<Int32>, weights: [Tensor<T>], biases: [Tensor<T>], rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0, numProj: Int64 = 0) -> Tensor<T>
```

Writes a set of weights into the opaque params buffer so they can be used in
upcoming training or inferences.

Note that the params buffer may not be compatible across different GPUs. So any
save and restoration should be converted to and from the canonical weights and
biases.

num\_layers: Specifies the number of layers in the RNN model.
num\_units: Specifies the size of the hidden state.
input\_size: Specifies the size of the input state.
weights: the canonical form of weights that can be used for saving
and restoration. They are more likely to be compatible across different
generations.
biases: the canonical form of biases that can be used for saving
and restoration. They are more likely to be compatible across different
generations.
num\_params\_weigths: number of weight parameter matrix for all layers.
num\_params\_biases: number of bias parameter vector for all layers.
rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicate whether there is a linear projection between the input and
The actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used.
dir = (direction == bidirectional) ? 2 : 1
dropout: dropout probability. When set to 0., dropout is disabled.
seed: the 1st part of a seed to initialize dropout.
seed2: the 2nd part of a seed to initialize dropout.
num\_proj: The output dimensionality for the projection matrices. If None or 0,
no projection is performed.

### `cudnnRNNParamsSize(numLayers:numUnits:inputSize:t:rnnMode:inputMode:direction:dropout:seed:seed2:numProj:)`

Computes size of weights that can be used by a Cudnn RNN model.

``` swift
@inlinable @inline(__always) public static func cudnnRNNParamsSize<S: TensorFlowIndex>(numLayers: Tensor<Int32>, numUnits: Tensor<Int32>, inputSize: Tensor<Int32>, t: TensorDataType, rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0, numProj: Int64 = 0) -> Tensor<S>
```

Return the params size that can be used by the Cudnn RNN model. Subsequent
weight allocation and initialization should use this size.

num\_layers: Specifies the number of layers in the RNN model.
num\_units: Specifies the size of the hidden state.
input\_size: Specifies the size of the input state.
rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicate whether there is a linear projection between the input and
The actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used.
dir = (direction == bidirectional) ? 2 : 1
dropout: dropout probability. When set to 0., dropout is disabled.
seed: the 1st part of a seed to initialize dropout.
seed2: the 2nd part of a seed to initialize dropout.
params\_size: The size of the params buffer that should be allocated and
initialized for this RNN model. Note that this params buffer may not be
compatible across GPUs. Please use CudnnRNNParamsWeights and
CudnnRNNParamsBiases to save and restore them in a way that is compatible
across different runs.

### `cudnnRNNParamsToCanonical(numLayers:numUnits:inputSize:params:numParams:rnnMode:inputMode:direction:dropout:seed:seed2:)`

Retrieves CudnnRNN params in canonical form.

``` swift
@inlinable @inline(__always) public static func cudnnRNNParamsToCanonical<T: FloatingPoint & TensorFlowScalar>(numLayers: Tensor<Int32>, numUnits: Tensor<Int32>, inputSize: Tensor<Int32>, params: Tensor<T>, numParams: Int64, rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0) -> (weights: [Tensor<T>], biases: [Tensor<T>])
```

Retrieves a set of weights from the opaque params buffer that can be saved and
restored in a way compatible with future runs.

Note that the params buffer may not be compatible across different GPUs. So any
save and restoration should be converted to and from the canonical weights and
biases.

num\_layers: Specifies the number of layers in the RNN model.
num\_units: Specifies the size of the hidden state.
input\_size: Specifies the size of the input state.
num\_params: number of parameter sets for all layers.
Each layer may contain multiple parameter sets, with each set consisting of
a weight matrix and a bias vector.
weights: the canonical form of weights that can be used for saving
and restoration. They are more likely to be compatible across different
generations.
biases: the canonical form of biases that can be used for saving
and restoration. They are more likely to be compatible across different
generations.
rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicate whether there is a linear projection between the input and
The actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used.
dir = (direction == bidirectional) ? 2 : 1
dropout: dropout probability. When set to 0., dropout is disabled.
seed: the 1st part of a seed to initialize dropout.
seed2: the 2nd part of a seed to initialize dropout.

### `cudnnRNNParamsToCanonicalV2(numLayers:numUnits:inputSize:params:numParamsWeights:numParamsBiases:rnnMode:inputMode:direction:dropout:seed:seed2:numProj:)`

Retrieves CudnnRNN params in canonical form. It supports the projection in LSTM.

``` swift
@inlinable @inline(__always) public static func cudnnRNNParamsToCanonicalV2<T: FloatingPoint & TensorFlowScalar>(numLayers: Tensor<Int32>, numUnits: Tensor<Int32>, inputSize: Tensor<Int32>, params: Tensor<T>, numParamsWeights: Int64, numParamsBiases: Int64, rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0, numProj: Int64 = 0) -> (weights: [Tensor<T>], biases: [Tensor<T>])
```

Retrieves a set of weights from the opaque params buffer that can be saved and
restored in a way compatible with future runs.

Note that the params buffer may not be compatible across different GPUs. So any
save and restoration should be converted to and from the canonical weights and
biases.

num\_layers: Specifies the number of layers in the RNN model.
num\_units: Specifies the size of the hidden state.
input\_size: Specifies the size of the input state.
num\_params\_weigths: number of weight parameter matrix for all layers.
num\_params\_biases: number of bias parameter vector for all layers.
weights: the canonical form of weights that can be used for saving
and restoration. They are more likely to be compatible across different
generations.
biases: the canonical form of biases that can be used for saving
and restoration. They are more likely to be compatible across different
generations.
rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicate whether there is a linear projection between the input and
The actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used.
dir = (direction == bidirectional) ? 2 : 1
dropout: dropout probability. When set to 0., dropout is disabled.
seed: the 1st part of a seed to initialize dropout.
seed2: the 2nd part of a seed to initialize dropout.
num\_proj: The output dimensionality for the projection matrices. If None or 0,
no projection is performed.

### `cudnnRNNV2(_:inputH:inputC:params:rnnMode:inputMode:direction:dropout:seed:seed2:isTraining:)`

A RNN backed by cuDNN.

``` swift
@inlinable @inline(__always) public static func cudnnRNNV2<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, inputH: Tensor<T>, inputC: Tensor<T>, params: Tensor<T>, rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0, isTraining: Bool = true) -> (
    output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, reserveSpace: Tensor<T>,
    hostReserved: Tensor<Int8>
  )
```

Computes the RNN from the input and initial states, with respect to the params
buffer. Produces one extra output "host\_reserved" than CudnnRNN.

rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicates whether there is a linear projection between the input and
the actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used. Should be
"unidirectional" or "bidirectional".
dropout: Dropout probability. When set to 0., dropout is disabled.
seed: The 1st part of a seed to initialize dropout.
seed2: The 2nd part of a seed to initialize dropout.
input: A 3-D tensor with the shape of \[seq\_length, batch\_size, input\_size\].
input\_h: A 3-D tensor with the shape of \[num\_layer \* dir, batch\_size,
num\_units\].
input\_c: For LSTM, a 3-D tensor with the shape of
\[num\_layer \* dir, batch, num\_units\]. For other models, it is ignored.
params: A 1-D tensor that contains the weights and biases in an opaque layout.
The size must be created through CudnnRNNParamsSize, and initialized
separately. Note that they might not be compatible across different
generations. So it is a good idea to save and restore
output: A 3-D tensor with the shape of \[seq\_length, batch\_size,
dir \* num\_units\].
output\_h: The same shape has input\_h.
output\_c: The same shape as input\_c for LSTM. An empty tensor for other models.
is\_training: Indicates whether this operation is used for inferenece or
training.
reserve\_space: An opaque tensor that can be used in backprop calculation. It
is only produced if is\_training is true.
host\_reserved: An opaque tensor that can be used in backprop calculation. It is
only produced if is\_training is true. It is output on host memory rather than
device memory.

### `cudnnRNNV3(_:inputH:inputC:params:sequenceLengths:rnnMode:inputMode:direction:dropout:seed:seed2:numProj:isTraining:timeMajor:)`

A RNN backed by cuDNN.

``` swift
@inlinable @inline(__always) public static func cudnnRNNV3<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, inputH: Tensor<T>, inputC: Tensor<T>, params: Tensor<T>, sequenceLengths: Tensor<Int32>, rnnMode: RnnMode = .lstm, inputMode: InputMode = .linearInput, direction: Direction = .unidirectional, dropout: Double = 0, seed: Int64 = 0, seed2: Int64 = 0, numProj: Int64 = 0, isTraining: Bool = true, timeMajor: Bool = true) -> (
    output: Tensor<T>, outputH: Tensor<T>, outputC: Tensor<T>, reserveSpace: Tensor<T>,
    hostReserved: Tensor<Int8>
  )
```

Computes the RNN from the input and initial states, with respect to the params
buffer. Accepts one extra input "sequence\_lengths" than CudnnRNN.

rnn\_mode: Indicates the type of the RNN model.
input\_mode: Indicates whether there is a linear projection between the input and
the actual computation before the first layer. 'skip\_input' is only allowed
when input\_size == num\_units; 'auto\_select' implies 'skip\_input' when
input\_size == num\_units; otherwise, it implies 'linear\_input'.
direction: Indicates whether a bidirectional model will be used. Should be
"unidirectional" or "bidirectional".
dropout: Dropout probability. When set to 0., dropout is disabled.
seed: The 1st part of a seed to initialize dropout.
seed2: The 2nd part of a seed to initialize dropout.
input: If time\_major is true, this is a 3-D tensor with the shape of
\[seq\_length, batch\_size, input\_size\]. If time\_major is false, the shape is
\[batch\_size, seq\_length, input\_size\].
input\_h: If time\_major is true, this is a 3-D tensor with the shape of
\[num\_layer \* dir, batch\_size, num\_units\]. If time\_major is false, the shape
is \[batch\_size, num\_layer \* dir, num\_units\].
input\_c: For LSTM, a 3-D tensor with the shape of
\[num\_layer \* dir, batch, num\_units\]. For other models, it is ignored.
params: A 1-D tensor that contains the weights and biases in an opaque layout.
The size must be created through CudnnRNNParamsSize, and initialized
separately. Note that they might not be compatible across different
generations. So it is a good idea to save and restore
sequence\_lengths: a vector of lengths of each input sequence.
output: If time\_major is true, this is a 3-D tensor with the shape of
\[seq\_length, batch\_size, dir \* num\_units\]. If time\_major is false, the
shape is \[batch\_size, seq\_length, dir \* num\_units\].
output\_h: The same shape has input\_h.
output\_c: The same shape as input\_c for LSTM. An empty tensor for other models.
is\_training: Indicates whether this operation is used for inferenece or
training.
time\_major: Indicates whether the input/output format is time major or batch
major.
reserve\_space: An opaque tensor that can be used in backprop calculation. It
is only produced if is\_training is true.

### `cumprod(_:axis:exclusive:reverse:)`

Compute the cumulative product of the tensor `x` along `axis`.

``` swift
@inlinable @inline(__always) public static func cumprod<T: TensorFlowNumeric, Tidx: TensorFlowIndex>(_ x: Tensor<T>, axis: Tensor<Tidx>, exclusive: Bool = false, reverse: Bool = false) -> Tensor<T>
```

By default, this op performs an inclusive cumprod, which means that the first
element of the input is identical to the first element of the output:

``` python
tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
performed instead:

``` python
tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
```

By setting the `reverse` kwarg to `True`, the cumprod is performed in the
opposite direction:

``` python
tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
```

This is more efficient than using separate `tf.reverse` ops.

The `reverse` and `exclusive` kwargs can also be combined:

``` python
tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
```

#### Parameters

  - x: - x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
  - axis: - axis: A `Tensor` of type `int32` (default: 0). Must be in the range `[-rank(x), rank(x))`.

### `cumsum(_:axis:exclusive:reverse:)`

Compute the cumulative sum of the tensor `x` along `axis`.

``` swift
@inlinable @inline(__always) public static func cumsum<T: TensorFlowNumeric, Tidx: TensorFlowIndex>(_ x: Tensor<T>, axis: Tensor<Tidx>, exclusive: Bool = false, reverse: Bool = false) -> Tensor<T>
```

By default, this op performs an inclusive cumsum, which means that the first
element of the input is identical to the first element of the output:

``` python
tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
performed instead:

``` python
tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
```

By setting the `reverse` kwarg to `True`, the cumsum is performed in the
opposite direction:

``` python
tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
```

This is more efficient than using separate `tf.reverse` ops.

The `reverse` and `exclusive` kwargs can also be combined:

``` python
tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
```

#### Parameters

  - x: - x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
  - axis: - axis: A `Tensor` of type `int32` (default: 0). Must be in the range `[-rank(x), rank(x))`.

### `cumulativeLogsumexp(_:axis:exclusive:reverse:)`

Compute the cumulative product of the tensor `x` along `axis`.

``` swift
@inlinable @inline(__always) public static func cumulativeLogsumexp<T: FloatingPoint & TensorFlowScalar, Tidx: TensorFlowIndex>(_ x: Tensor<T>, axis: Tensor<Tidx>, exclusive: Bool = false, reverse: Bool = false) -> Tensor<T>
```

By default, this op performs an inclusive cumulative log-sum-exp,
which means that the first
element of the input is identical to the first element of the output:

``` python
tf.math.cumulative_logsumexp([a, b, c])  # => [a, log(exp(a) + exp(b)), log(exp(a) + exp(b) + exp(c))]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumulative log-sum-exp is
performed instead:

``` python
tf.cumulative_logsumexp([a, b, c], exclusive=True)  # => [-inf, a, log(exp(a) * exp(b))]
```

Note that the neutral element of the log-sum-exp operation is `-inf`,
however, for performance reasons, the minimal value representable by the
floating point type is used instead.

By setting the `reverse` kwarg to `True`, the cumulative log-sum-exp is performed in the
opposite direction.

#### Parameters

  - x: - x: A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
  - axis: - axis: A `Tensor` of type `int32` (default: 0). Must be in the range `[-rank(x), rank(x))`.

### `dataFormatDimMap(_:srcFormat:dstFormat:)`

Returns the dimension index in the destination data format given the one in

``` swift
@inlinable @inline(__always) public static func dataFormatDimMap<T: TensorFlowIndex>(_ x: Tensor<T>, srcFormat: String = "NHWC", dstFormat: String = "NCHW") -> Tensor<T>
```

the source data format.

#### Parameters

  - x: - x: A Tensor with each element as a dimension index in source data format. Must be in the range \[-4, 4).

### `dataFormatVecPermute(_:srcFormat:dstFormat:)`

Returns the permuted vector/tensor in the destination data format given the

``` swift
@inlinable @inline(__always) public static func dataFormatVecPermute<T: TensorFlowIndex>(_ x: Tensor<T>, srcFormat: String = "NHWC", dstFormat: String = "NCHW") -> Tensor<T>
```

one in the source data format.

#### Parameters

  - x: - x: Vector of size 4 or Tensor of shape (4, 2) in source data format.

### `datasetCardinality(inputDataset:)`

Returns the cardinality of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func datasetCardinality(inputDataset: VariantHandle) -> Tensor<Int64>
```

Returns the cardinality of `input_dataset`.

### `datasetFromGraph(graphDef:)`

Creates a dataset from the given `graph_def`.

``` swift
@inlinable @inline(__always) public static func datasetFromGraph(graphDef: StringTensor) -> VariantHandle
```

Creates a dataset from the provided `graph_def`.

### `datasetToGraph(inputDataset:statefulWhitelist:allowStateful:stripDeviceAssignment:)`

Returns a serialized GraphDef representing `input_dataset`.

``` swift
@inlinable @inline(__always) public static func datasetToGraph(inputDataset: VariantHandle, statefulWhitelist: [String], allowStateful: Bool = false, stripDeviceAssignment: Bool = false) -> StringTensor
```

Returns a graph representation for `input_dataset`.

### `datasetToGraphV2(inputDataset:externalStatePolicy:stripDeviceAssignment:)`

Returns a serialized GraphDef representing `input_dataset`.

``` swift
@inlinable @inline(__always) public static func datasetToGraphV2(inputDataset: VariantHandle, externalStatePolicy: Int64 = 0, stripDeviceAssignment: Bool = false) -> StringTensor
```

Returns a graph representation for `input_dataset`.

### `datasetToSingleElement(dataset:outputShapes:)`

Outputs the single element from the given dataset.

``` swift
@inlinable @inline(__always) public static func datasetToSingleElement<OutputTypes: TensorGroup>(dataset: VariantHandle, outputShapes: [TensorShape?]) -> OutputTypes
```

#### Parameters

  - dataset: - dataset: A handle to a dataset that contains a single element.

### `datasetToTFRecord(inputDataset:filename:compressionType:)`

Writes the given dataset to the given file using the TFRecord format.

``` swift
@inlinable @inline(__always) public static func datasetToTFRecord(inputDataset: VariantHandle, filename: StringTensor, compressionType: StringTensor)
```

#### Parameters

  - filename: - filename: A scalar string tensor representing the filename to use.

### `debugGradientIdentity(_:)`

Identity op for gradient debugging.

``` swift
@inlinable @inline(__always) public static func debugGradientIdentity<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

This op is hidden from public in Python. It is used by TensorFlow Debugger to
register gradient tensors for gradient debugging.
This op operates on non-reference-type tensors.

### `debugIdentity(_:deviceName:tensorName:debugUrls:gatedGrpc:)`

Provides an identity mapping of the non-Ref type input tensor for debugging.

``` swift
@inlinable @inline(__always) public static func debugIdentity<T: TensorFlowScalar>(_ input: Tensor<T>, deviceName: String, tensorName: String, debugUrls: [String], gatedGrpc: Bool = false) -> Tensor<T>
```

Provides an identity mapping of the non-Ref type input tensor for debugging.

#### Parameters

  - input: - input: Input tensor, non-Reference type

### `debugIdentityV2(_:tfdbgContextId:opName:outputSlot:tensorDebugMode:debugUrls:)`

Debug Identity V2 Op.

``` swift
@inlinable @inline(__always) public static func debugIdentityV2<T: TensorFlowScalar>(_ input: Tensor<T>, tfdbgContextId: String, opName: String, outputSlot: Int64 = -1, tensorDebugMode: Int64 = -1, debugUrls: [String]) -> Tensor<T>
```

Provides an identity mapping from input to output, while writing the content of
the input tensor by calling DebugEventsWriter.

The semantics of the input tensor depends on tensor\_debug\_mode. In typical
usage, the input tensor comes directly from the user computation only when
graph\_debug\_mode is FULL\_TENSOR (see protobuf/debug\_event.proto for a
list of all the possible values of graph\_debug\_mode). For the other debug modes,
the input tensor should be produced by an additional op or subgraph that
computes summary information about one or more tensors.

#### Parameters

  - input: - input: Input tensor, non-Reference type

### `debugNanCount(_:deviceName:tensorName:debugUrls:gatedGrpc:)`

Debug NaN Value Counter Op.

``` swift
@inlinable @inline(__always) public static func debugNanCount<T: TensorFlowScalar>(_ input: Tensor<T>, deviceName: String, tensorName: String, debugUrls: [String], gatedGrpc: Bool = false) -> Tensor<Int64>
```

Counts number of NaNs in the input tensor, for debugging.

#### Parameters

  - input: - input: Input tensor, non-Reference type.

### `debugNumericSummary(_:deviceName:tensorName:debugUrls:lowerBound:upperBound:muteIfHealthy:gatedGrpc:)`

Debug Numeric Summary Op.

``` swift
@inlinable @inline(__always) public static func debugNumericSummary<T: TensorFlowScalar>(_ input: Tensor<T>, deviceName: String, tensorName: String, debugUrls: [String], lowerBound: Double = -Double.infinity, upperBound: Double = Double.infinity, muteIfHealthy: Bool = false, gatedGrpc: Bool = false) -> Tensor<Double>
```

Provide a basic summary of numeric value types, range and distribution.

output: A double tensor of shape \[14 + nDimensions\], where nDimensions is the
the number of dimensions of the tensor's shape. The elements of output are:
\[0\]: is initialized (1.0) or not (0.0).
\[1\]: total number of elements
\[2\]: NaN element count
\[3\]: generalized -inf count: elements \<= lower\_bound. lower\_bound is -inf by
default.
\[4\]: negative element count (excluding -inf), if lower\_bound is the default
\-inf. Otherwise, this is the count of elements \> lower\_bound and \< 0.
\[5\]: zero element count
\[6\]: positive element count (excluding +inf), if upper\_bound is the default
\-inf. Otherwise, this is the count of elements \< upper\_bound and \> 0.
\[7\]: generalized +inf count, elements \>= upper\_bound. upper\_bound is +inf by
default.
Output elements \[1:8\] are all zero, if the tensor is uninitialized.
\[8\]: minimum of all non-inf and non-NaN elements.
If uninitialized or no such element exists: +inf.
\[9\]: maximum of all non-inf and non-NaN elements.
If uninitialized or no such element exists: -inf.
\[10\]: mean of all non-inf and non-NaN elements.
If uninitialized or no such element exists: NaN.
\[11\]: variance of all non-inf and non-NaN elements.
If uninitialized or no such element exists: NaN.
\[12\]: Data type of the tensor encoded as an enum integer. See the DataType
proto for more details.
\[13\]: Number of dimensions of the tensor (ndims).
\[14+\]: Sizes of the dimensions.

#### Parameters

  - input: - input: Input tensor, non-Reference type.

### `debugNumericSummaryV2(_:tensorDebugMode:tensorId:)`

``` swift
@inlinable @inline(__always) public static func debugNumericSummaryV2<T: TensorFlowScalar>(_ input: Tensor<T>, tensorDebugMode: Int64 = -1, tensorId: Int64 = -1) -> Tensor<Float>
```

#### Parameters

  - input: - input: Input tensor, to be summarized by the op.

### `decodeAndCropJpeg(contents:cropWindow:channels:ratio:fancyUpscaling:tryRecoverTruncated:acceptableFraction:dctMethod:)`

Decode and Crop a JPEG-encoded image to a uint8 tensor.

``` swift
@inlinable @inline(__always) public static func decodeAndCropJpeg(contents: StringTensor, cropWindow: Tensor<Int32>, channels: Int64 = 0, ratio: Int64 = 1, fancyUpscaling: Bool = true, tryRecoverTruncated: Bool = false, acceptableFraction: Double = 1, dctMethod: String) -> Tensor<UInt8>
```

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.

The attr `ratio` allows downscaling the image by an integer factor during
decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
downscaling the image later.

It is equivalent to a combination of decode and crop, but much faster by only
decoding partial jpeg image.

#### Parameters

  - contents: - contents: 0-D.  The JPEG-encoded image.

### `decodeBase64(_:)`

Decode web-safe base64-encoded strings.

``` swift
@inlinable @inline(__always) public static func decodeBase64(_ input: StringTensor) -> StringTensor
```

Input may or may not have padding at the end. See EncodeBase64 for padding.
Web-safe means that input must use - and \_ instead of + and /.

#### Parameters

  - input: - input: Base64 strings to decode.

### `decodeBmp(contents:channels:)`

Decode the first frame of a BMP-encoded image to a uint8 tensor.

``` swift
@inlinable @inline(__always) public static func decodeBmp(contents: StringTensor, channels: Int64 = 0) -> Tensor<UInt8>
```

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

#### Parameters

  - contents: - contents: 0-D.  The BMP-encoded image.

### `decodeCSV(records:recordDefaults:fieldDelim:useQuoteDelim:naValue:selectCols:)`

Convert CSV records to tensors. Each column maps to one tensor.

``` swift
@inlinable @inline(__always) public static func decodeCSV<OutType: TensorArrayProtocol>(records: StringTensor, recordDefaults: OutType, fieldDelim: String = ",", useQuoteDelim: Bool = true, naValue: String, selectCols: [Int32]) -> OutType
```

RFC 4180 format is expected for the CSV records.
(https://tools.ietf.org/html/rfc4180)
Note that we allow leading and trailing spaces with int or float field.

#### Parameters

  - records: - records: Each string is a record/row in the csv and all records should have the same format.

### `decodeCompressed(bytes:compressionType:)`

Decompress strings.

``` swift
@inlinable @inline(__always) public static func decodeCompressed(bytes: StringTensor, compressionType: String) -> StringTensor
```

This op decompresses each element of the `bytes` input `Tensor`, which
is assumed to be compressed using the given `compression_type`.

The `output` is a string `Tensor` of the same shape as `bytes`,
each element containing the decompressed data from the corresponding
element in `bytes`.

#### Parameters

  - bytes: - bytes: A Tensor of string which is compressed.

### `decodeGif(contents:)`

Decode the frame(s) of a GIF-encoded image to a uint8 tensor.

``` swift
@inlinable @inline(__always) public static func decodeGif(contents: StringTensor) -> Tensor<UInt8>
```

GIF images with frame or transparency compression are not supported.
On Linux and MacOS systems, convert animated GIFs from compressed to
uncompressed by running:

``` 
convert $src.gif -coalesce $dst.gif
```

This op also supports decoding JPEGs and PNGs, though it is cleaner to use
`tf.image.decode_image`.

#### Parameters

  - contents: - contents: 0-D.  The GIF-encoded image.

### `decodeJSONExample(jsonExamples:)`

Convert JSON-encoded Example records to binary protocol buffer strings.

``` swift
@inlinable @inline(__always) public static func decodeJSONExample(jsonExamples: StringTensor) -> StringTensor
```

This op translates a tensor containing Example records, encoded using
the [standard JSON
mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
into a tensor containing the same records encoded as binary protocol
buffers. The resulting tensor can then be fed to any of the other
Example-parsing ops.

### `decodeJpeg(contents:channels:ratio:fancyUpscaling:tryRecoverTruncated:acceptableFraction:dctMethod:)`

Decode a JPEG-encoded image to a uint8 tensor.

``` swift
@inlinable @inline(__always) public static func decodeJpeg(contents: StringTensor, channels: Int64 = 0, ratio: Int64 = 1, fancyUpscaling: Bool = true, tryRecoverTruncated: Bool = false, acceptableFraction: Double = 1, dctMethod: String) -> Tensor<UInt8>
```

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.

The attr `ratio` allows downscaling the image by an integer factor during
decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
downscaling the image later.

This op also supports decoding PNGs and non-animated GIFs since the interface is
the same, though it is cleaner to use `tf.image.decode_image`.

#### Parameters

  - contents: - contents: 0-D.  The JPEG-encoded image.

### `decodePaddedRaw(inputBytes:fixedLength:littleEndian:)`

Reinterpret the bytes of a string as a vector of numbers.

``` swift
@inlinable @inline(__always) public static func decodePaddedRaw<OutType: TensorFlowNumeric>(inputBytes: StringTensor, fixedLength: Tensor<Int32>, littleEndian: Bool = true) -> Tensor<OutType>
```

### `decodePng(contents:channels:)`

Decode a PNG-encoded image to a uint8 or uint16 tensor.

``` swift
@inlinable @inline(__always) public static func decodePng<Dtype: UnsignedInteger & TensorFlowScalar>(contents: StringTensor, channels: Int64 = 0) -> Tensor<Dtype>
```

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

If needed, the PNG-encoded image is transformed to match the requested number
of color channels.

This op also supports decoding JPEGs and non-animated GIFs since the interface
is the same, though it is cleaner to use `tf.image.decode_image`.

#### Parameters

  - contents: - contents: 0-D.  The PNG-encoded image.

### `decodeProtoV2(bytes:messageType:fieldNames:descriptorSource:messageFormat:sanitize:)`

The op extracts fields from a serialized protocol buffers message into tensors.

``` swift
@inlinable @inline(__always) public static func decodeProtoV2<OutputTypes: TensorGroup>(bytes: StringTensor, messageType: String, fieldNames: [String], descriptorSource: String = "local://", messageFormat: String = "binary", sanitize: Bool = false) -> (sizes: Tensor<Int32>, values: OutputTypes)
```

The `decode_proto` op extracts fields from a serialized protocol buffers
message into tensors.  The fields in `field_names` are decoded and converted
to the corresponding `output_types` if possible.

A `message_type` name must be provided to give context for the field names.
The actual message descriptor can be looked up either in the linked-in
descriptor pool or a filename provided by the caller using the
`descriptor_source` attribute.

Each output tensor is a dense tensor. This means that it is padded to hold
the largest number of repeated elements seen in the input minibatch. (The
shape is also padded by one to prevent zero-sized dimensions). The actual
repeat counts for each example in the minibatch can be found in the `sizes`
output. In many cases the output of `decode_proto` is fed immediately into
tf.squeeze if missing values are not a concern. When using tf.squeeze, always
pass the squeeze dimension explicitly to avoid surprises.

For the most part, the mapping between Proto field types and TensorFlow dtypes
is straightforward. However, there are a few special cases:

Both binary and text proto serializations are supported, and can be
chosen using the `format` attribute.

The `descriptor_source` attribute selects the source of protocol
descriptors to consult when looking up `message_type`. This may be:

#### Parameters

  - bytes: - bytes: Tensor of serialized protos with shape `batch_shape`.

### `decodeRaw(bytes:littleEndian:)`

Reinterpret the bytes of a string as a vector of numbers.

``` swift
@inlinable @inline(__always) public static func decodeRaw<OutType: TensorFlowScalar>(bytes: StringTensor, littleEndian: Bool = true) -> Tensor<OutType>
```

#### Parameters

  - bytes: - bytes: All the elements must have the same length.

### `decodeWav(contents:desiredChannels:desiredSamples:)`

Decode a 16-bit PCM WAV file to a float tensor.

``` swift
@inlinable @inline(__always) public static func decodeWav(contents: StringTensor, desiredChannels: Int64 = -1, desiredSamples: Int64 = -1) -> (audio: Tensor<Float>, sampleRate: Tensor<Int32>)
```

The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.

When desired\_channels is set, if the input contains fewer channels than this
then the last channel will be duplicated to give the requested number, else if
the input has more channels than requested then the additional channels will be
ignored.

If desired\_samples is set, then the audio will be cropped or padded with zeroes
to the requested length.

The first output contains a Tensor with the content of the audio samples. The
lowest dimension will be the number of channels, and the second will be the
number of samples. For example, a ten-sample-long stereo WAV file should give an
output shape of \[10, 2\].

#### Parameters

  - contents: - contents: The WAV-encoded audio, usually from a file.

### `deepCopy(_:)`

Makes a copy of `x`.

``` swift
@inlinable @inline(__always) public static func deepCopy<T: TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

#### Parameters

  - x: - x: The source tensor of type `T`.

### `deleteIterator(handle:deleter:)`

A container for an iterator resource.

``` swift
@inlinable @inline(__always) public static func deleteIterator(handle: ResourceHandle, deleter: VariantHandle)
```

#### Parameters

  - handle: - handle: A handle to the iterator to delete.
  - deleter: - deleter: A variant deleter.

### `deleteMemoryCache(handle:deleter:)`

``` swift
@inlinable @inline(__always) public static func deleteMemoryCache(handle: ResourceHandle, deleter: VariantHandle)
```

### `deleteMultiDeviceIterator(multiDeviceIterator:iterators:deleter:)`

A container for an iterator resource.

``` swift
@inlinable @inline(__always) public static func deleteMultiDeviceIterator(multiDeviceIterator: ResourceHandle, iterators: [ResourceHandle], deleter: VariantHandle)
```

#### Parameters

  - iterators: - iterators: A list of iterator handles (unused). This is added so that automatic control dependencies get added during function tracing that ensure this op runs after all the dependent iterators are deleted.
  - deleter: - deleter: A variant deleter.

### `deleteRandomSeedGenerator(handle:deleter:)`

``` swift
@inlinable @inline(__always) public static func deleteRandomSeedGenerator(handle: ResourceHandle, deleter: VariantHandle)
```

### `deleteSessionTensor(handle:)`

Delete the tensor specified by its handle in the session.

``` swift
@inlinable @inline(__always) public static func deleteSessionTensor(handle: StringTensor)
```

#### Parameters

  - handle: - handle: The handle for a tensor stored in the session state.

### `denseToCSRSparseMatrix(denseInput:indices:)`

Converts a dense tensor to a (possibly batched) CSRSparseMatrix.

``` swift
@inlinable @inline(__always) public static func denseToCSRSparseMatrix<T: FloatingPoint & TensorFlowScalar>(denseInput: Tensor<T>, indices: Tensor<Int64>) -> VariantHandle
```

#### Parameters

  - indices: - indices: Indices of nonzero elements.

### `denseToDenseSetOperation(set1:set2:setOperation:validateIndices:)`

Applies set operation along last dimension of 2 `Tensor` inputs.

``` swift
@inlinable @inline(__always) public static func denseToDenseSetOperation<T: TensorFlowInteger>(set1: Tensor<T>, set2: Tensor<T>, setOperation: String, validateIndices: Bool = true) -> (resultIndices: Tensor<Int64>, resultValues: Tensor<T>, resultShape: Tensor<Int64>)
```

See SetOperationOp::SetOperationFromContext for values of `set_operation`.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

#### Parameters

  - set1: - set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`. Dimension `n` contains values in a set, duplicates are allowed but ignored.
  - set2: - set2: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set1`. Dimension `n` contains values in a set, duplicates are allowed but ignored.

### `denseToDenseSetOperation(set1:set2:setOperation:validateIndices:)`

Applies set operation along last dimension of 2 `Tensor` inputs.

``` swift
@inlinable @inline(__always) public static func denseToDenseSetOperation(set1: StringTensor, set2: StringTensor, setOperation: String, validateIndices: Bool = true) -> (resultIndices: Tensor<Int64>, resultValues: StringTensor, resultShape: Tensor<Int64>)
```

See SetOperationOp::SetOperationFromContext for values of `set_operation`.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

#### Parameters

  - set1: - set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`. Dimension `n` contains values in a set, duplicates are allowed but ignored.
  - set2: - set2: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set1`. Dimension `n` contains values in a set, duplicates are allowed but ignored.

### `denseToSparseBatchDataset(inputDataset:batchSize:rowShape:outputTypes:outputShapes:)`

Creates a dataset that batches input elements into a SparseTensor.

``` swift
@inlinable @inline(__always) public static func denseToSparseBatchDataset(inputDataset: VariantHandle, batchSize: Tensor<Int64>, rowShape: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `denseToSparseSetOperation(set1:set2Indices:set2Values:set2Shape:setOperation:validateIndices:)`

Applies set operation along last dimension of `Tensor` and `SparseTensor`.

``` swift
@inlinable @inline(__always) public static func denseToSparseSetOperation<T: TensorFlowInteger>(set1: Tensor<T>, set2Indices: Tensor<Int64>, set2Values: Tensor<T>, set2Shape: Tensor<Int64>, setOperation: String, validateIndices: Bool = true) -> (resultIndices: Tensor<Int64>, resultValues: Tensor<T>, resultShape: Tensor<Int64>)
```

See SetOperationOp::SetOperationFromContext for values of `set_operation`.

Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

If `validate_indices` is `True`, this op validates the order and range of `set2`
indices.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

#### Parameters

  - set1: - set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`. Dimension `n` contains values in a set, duplicates are allowed but ignored.

### `denseToSparseSetOperation(set1:set2Indices:set2Values:set2Shape:setOperation:validateIndices:)`

Applies set operation along last dimension of `Tensor` and `SparseTensor`.

``` swift
@inlinable @inline(__always) public static func denseToSparseSetOperation(set1: StringTensor, set2Indices: Tensor<Int64>, set2Values: StringTensor, set2Shape: Tensor<Int64>, setOperation: String, validateIndices: Bool = true) -> (resultIndices: Tensor<Int64>, resultValues: StringTensor, resultShape: Tensor<Int64>)
```

See SetOperationOp::SetOperationFromContext for values of `set_operation`.

Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

If `validate_indices` is `True`, this op validates the order and range of `set2`
indices.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

#### Parameters

  - set1: - set1: `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`. Dimension `n` contains values in a set, duplicates are allowed but ignored.

### `depthToSpace(_:blockSize:dataFormat:)`

DepthToSpace for tensors of type T.

``` swift
@inlinable @inline(__always) public static func depthToSpace<T: TensorFlowScalar>(_ input: Tensor<T>, blockSize: Int64, dataFormat: DataFormat5 = .nhwc) -> Tensor<T>
```

Rearranges data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically,
this op outputs a copy of the input tensor where values from the `depth`
dimension are moved in spatial blocks to the `height` and `width` dimensions.
The attr `block_size` indicates the input block size and how the data is moved.

The `data_format` attr specifies the layout of the input and output tensors
with the following options:
"NHWC": `[ batch, height, width, channels ]`
"NCHW": `[ batch, channels, height, width ]`
"NCHW\_VECT\_C":
`qint8 [ batch, channels / 4, height, width, 4 ]`

It is useful to consider the operation as transforming a 6-D Tensor.
e.g. for data\_format = NHWC,
Each element in the input tensor can be specified via 6 coordinates,
ordered by decreasing memory layout significance as:
n,iY,iX,bY,bX,oC  (where n=batch index, iX, iY means X or Y coordinates
within the input image, bX, bY means coordinates
within the output block, oC means output channels).
The output would be the input transposed to the following layout:
n,iY,bY,iX,bX,oC

This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.

For example, given an input of shape `[1, 1, 1, 4]`, data\_format = "NHWC" and
block\_size = 2:

``` 
x = [[[[1, 2, 3, 4]]]]
 
```

This operation will output a tensor of shape `[1, 2, 2, 1]`:

``` 
   [[[[1], [2]],
     [[3], [4]]]]
```

Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
the corresponding output will have 2x2 elements and will have a depth of
1 channel (1 = `4 / (block_size * block_size)`).
The output element shape is `[2, 2, 1]`.

For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.

``` 
x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```

This operation, for block size of 2, will return the following tensor of shape
`[1, 2, 2, 3]`

``` 
   [[[[1, 2, 3], [4, 5, 6]],
     [[7, 8, 9], [10, 11, 12]]]]
 
```

Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:

``` 
x =  [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```

the operator will return the following tensor of shape `[1 4 4 1]`:

``` 
x = [[[ [1],   [2],  [5],  [6]],
      [ [3],   [4],  [7],  [8]],
      [ [9],  [10], [13],  [14]],
      [ [11], [12], [15],  [16]]]]
 
```

### `depthwiseConv2dNative(_:filter:strides:padding:dataFormat:dilations:)`

Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

``` swift
@inlinable @inline(__always) public static func depthwiseConv2dNative<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, filter: Tensor<T>, strides: [Int32], padding: Padding, dataFormat: DataFormat = .nhwc, dilations: [Int32] = [1, 1, 1, 1]) -> Tensor<T>
```

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, channel_multiplier]`, containing
`in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
a different filter to each input channel (expanding from 1 channel to
`channel_multiplier` channels for each), then concatenates the results
together. Thus, the output has `in_channels * channel_multiplier` channels.

``` 
for k in 0..in_channels-1
  for q in 0..channel_multiplier-1
    output[b, i, j, k * channel_multiplier + q] =
      sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                        filter[di, dj, k, q]
```

Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

### `depthwiseConv2dNativeBackpropFilter(_:filterSizes:outBackprop:strides:padding:dataFormat:dilations:)`

Computes the gradients of depthwise convolution with respect to the filter.

``` swift
@inlinable @inline(__always) public static func depthwiseConv2dNativeBackpropFilter<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, filterSizes: Tensor<Int32>, outBackprop: Tensor<T>, strides: [Int32], padding: Padding, dataFormat: DataFormat = .nhwc, dilations: [Int32] = [1, 1, 1, 1]) -> Tensor<T>
```

#### Parameters

  - input: - input: 4-D with shape based on `data_format`.  For example, if `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height, in_width, in_channels]` tensor.

### `depthwiseConv2dNativeBackpropInput(inputSizes:filter:outBackprop:strides:padding:dataFormat:dilations:)`

Computes the gradients of depthwise convolution with respect to the input.

``` swift
@inlinable @inline(__always) public static func depthwiseConv2dNativeBackpropInput<T: FloatingPoint & TensorFlowScalar>(inputSizes: Tensor<Int32>, filter: Tensor<T>, outBackprop: Tensor<T>, strides: [Int32], padding: Padding, dataFormat: DataFormat = .nhwc, dilations: [Int32] = [1, 1, 1, 1]) -> Tensor<T>
```

#### Parameters

  - filter: - filter: 4-D with shape `[filter_height, filter_width, in_channels, depthwise_multiplier]`.

### `dequantize(_:minRange:maxRange:mode:narrowRange:axis:)`

``` swift
@inlinable @inline(__always) public static func dequantize<T: TensorFlowScalar>(_ input: Tensor<T>, minRange: Tensor<Float>, maxRange: Tensor<Float>, mode: Mode = .minCombined, narrowRange: Bool = false, axis: Int64 = -1) -> Tensor<Float>
```

### `deserializeIterator(resourceHandle:serialized:)`

Converts the given variant tensor to an iterator and stores it in the given resource.

``` swift
@inlinable @inline(__always) public static func deserializeIterator(resourceHandle: ResourceHandle, serialized: VariantHandle)
```

#### Parameters

  - serialized: - serialized: A variant tensor storing the state of the iterator contained in the resource.

### `deserializeManySparse(serializedSparse:)`

Deserialize and concatenate `SparseTensors` from a serialized minibatch.

``` swift
@inlinable @inline(__always) public static func deserializeManySparse<Dtype: TensorFlowScalar>(serializedSparse: StringTensor) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>)
```

The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
`N` is the minibatch size and the rows correspond to packed outputs of
`SerializeSparse`.  The ranks of the original `SparseTensor` objects
must all match.  When the final `SparseTensor` is created, it has rank one
higher than the ranks of the incoming `SparseTensor` objects
(they have been concatenated along a new row dimension).

The output `SparseTensor` object's shape values for all dimensions but the
first are the max across the input `SparseTensor` objects' shape values
for the corresponding dimensions.  Its first shape value is `N`, the minibatch
size.

The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.

For example, if the serialized input is a `[2 x 3]` matrix representing two
original `SparseTensor` objects:

``` 
index = [ 0]
        [10]
        [20]
values = [1, 2, 3]
shape = [50]
```

and

``` 
index = [ 2]
        [10]
values = [4, 5]
shape = [30]
```

then the final deserialized `SparseTensor` will be:

``` 
index = [0  0]
        [0 10]
        [0 20]
        [1  2]
        [1 10]
values = [1, 2, 3, 4, 5]
shape = [2 50]
```

### `deserializeSparse(serializedSparse:)`

Deserialize `SparseTensor` objects.

``` swift
@inlinable @inline(__always) public static func deserializeSparse<Dtype: TensorFlowScalar, Tserialized: TensorFlowScalar>(serializedSparse: Tensor<Tserialized>) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>)
```

The input `serialized_sparse` must have the shape `[?, ?, ..., ?, 3]` where
the last dimension stores serialized `SparseTensor` objects and the other N
dimensions (N \>= 0) correspond to a batch. The ranks of the original
`SparseTensor` objects must all match. When the final `SparseTensor` is
created, its rank is the rank of the incoming `SparseTensor` objects plus N;
the sparse tensors have been concatenated along new dimensions, one for each
batch.

The output `SparseTensor` object's shape values for the original dimensions
are the max across the input `SparseTensor` objects' shape values for the
corresponding dimensions. The new dimensions match the size of the batch.

The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.

For example, if the serialized input is a `[2 x 3]` matrix representing two
original `SparseTensor` objects:

``` 
index = [ 0]
        [10]
        [20]
values = [1, 2, 3]
shape = [50]
```

and

``` 
index = [ 2]
        [10]
values = [4, 5]
shape = [30]
```

then the final deserialized `SparseTensor` will be:

``` 
index = [0  0]
        [0 10]
        [0 20]
        [1  2]
        [1 10]
values = [1, 2, 3, 4, 5]
shape = [2 50]
```

### `deserializeSparse(serializedSparse:)`

Deserialize `SparseTensor` objects.

``` swift
@inlinable @inline(__always) public static func deserializeSparse<Dtype: TensorFlowScalar>(serializedSparse: StringTensor) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>)
```

The input `serialized_sparse` must have the shape `[?, ?, ..., ?, 3]` where
the last dimension stores serialized `SparseTensor` objects and the other N
dimensions (N \>= 0) correspond to a batch. The ranks of the original
`SparseTensor` objects must all match. When the final `SparseTensor` is
created, its rank is the rank of the incoming `SparseTensor` objects plus N;
the sparse tensors have been concatenated along new dimensions, one for each
batch.

The output `SparseTensor` object's shape values for the original dimensions
are the max across the input `SparseTensor` objects' shape values for the
corresponding dimensions. The new dimensions match the size of the batch.

The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.

For example, if the serialized input is a `[2 x 3]` matrix representing two
original `SparseTensor` objects:

``` 
index = [ 0]
        [10]
        [20]
values = [1, 2, 3]
shape = [50]
```

and

``` 
index = [ 2]
        [10]
values = [4, 5]
shape = [30]
```

then the final deserialized `SparseTensor` will be:

``` 
index = [0  0]
        [0 10]
        [0 20]
        [1  2]
        [1 10]
values = [1, 2, 3, 4, 5]
shape = [2 50]
```

### `destroyResourceOp(resource:ignoreLookupError:)`

Deletes the resource specified by the handle.

``` swift
@inlinable @inline(__always) public static func destroyResourceOp(resource: ResourceHandle, ignoreLookupError: Bool = true)
```

All subsequent operations using the resource will result in a NotFound
error status.

#### Parameters

  - resource: - resource: handle to the resource to delete.

### `devicePlacementOp()`

``` swift
@inlinable @inline(__always) public static func devicePlacementOp() -> StringTensor
```

### `diag(diagonal:)`

Returns a diagonal tensor with a given diagonal values.

``` swift
@inlinable @inline(__always) public static func diag<T: TensorFlowNumeric>(diagonal: Tensor<T>) -> Tensor<T>
```

Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:

Assume `diagonal` has dimensions \[D1,..., Dk\], then the output is a tensor of
rank 2k with dimensions \[D1,..., Dk, D1,..., Dk\] where:

`output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.

For example:

``` 
# 'diagonal' is [1, 2, 3, 4]
tf.diag(diagonal) ==> [[1, 0, 0, 0]
                       [0, 2, 0, 0]
                       [0, 0, 3, 0]
                       [0, 0, 0, 4]]
```

#### Parameters

  - diagonal: - diagonal: Rank k tensor where k is at most 1.

### `diagPart(_:)`

Returns the diagonal part of the tensor.

``` swift
@inlinable @inline(__always) public static func diagPart<T: TensorFlowNumeric>(_ input: Tensor<T>) -> Tensor<T>
```

This operation returns a tensor with the `diagonal` part
of the `input`. The `diagonal` part is computed as follows:

Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
tensor of rank `k` with dimensions `[D1,..., Dk]` where:

`diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

For example:

``` 
# 'input' is [[1, 0, 0, 0]
              [0, 2, 0, 0]
              [0, 0, 3, 0]
              [0, 0, 0, 4]]
 
tf.diag_part(input) ==> [1, 2, 3, 4]
```

#### Parameters

  - input: - input: Rank k tensor where k is even and not zero.

### `digamma(_:)`

Computes Psi, the derivative of Lgamma (the log of the absolute value of

``` swift
@inlinable @inline(__always) public static func digamma<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

`Gamma(x)`), element-wise.

### `dilation2D(_:filter:strides:rates:padding:)`

Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

``` swift
@inlinable @inline(__always) public static func dilation2D<T: TensorFlowNumeric>(_ input: Tensor<T>, filter: Tensor<T>, strides: [Int32], rates: [Int32], padding: Padding) -> Tensor<T>
```

The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
`filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
input channel is processed independently of the others with its own structuring
function. The `output` tensor has shape
`[batch, out_height, out_width, depth]`. The spatial dimensions of the output
tensor depend on the `padding` algorithm. We currently only support the default
"NHWC" `data_format`.

In detail, the grayscale morphological 2-D dilation is the max-sum correlation
(for consistency with `conv2d`, we use unmirrored filters):

``` 
output[b, y, x, c] =
   max_{dy, dx} input[b,
                      strides[1] * y + rates[1] * dy,
                      strides[2] * x + rates[2] * dx,
                      c] +
                filter[dy, dx, c]
```

Max-pooling is a special case when the filter has size equal to the pooling
kernel size and contains all zeros.

Note on duality: The dilation of `input` by the `filter` is equal to the
negation of the erosion of `-input` by the reflected `filter`.

#### Parameters

  - input: - input: 4-D with shape `[batch, in_height, in_width, depth]`.
  - filter: - filter: 3-D with shape `[filter_height, filter_width, depth]`.

### `dilation2DBackpropFilter(_:filter:outBackprop:strides:rates:padding:)`

Computes the gradient of morphological 2-D dilation with respect to the filter.

``` swift
@inlinable @inline(__always) public static func dilation2DBackpropFilter<T: TensorFlowNumeric>(_ input: Tensor<T>, filter: Tensor<T>, outBackprop: Tensor<T>, strides: [Int32], rates: [Int32], padding: Padding) -> Tensor<T>
```

#### Parameters

  - input: - input: 4-D with shape `[batch, in_height, in_width, depth]`.
  - filter: - filter: 3-D with shape `[filter_height, filter_width, depth]`.

### `dilation2DBackpropInput(_:filter:outBackprop:strides:rates:padding:)`

Computes the gradient of morphological 2-D dilation with respect to the input.

``` swift
@inlinable @inline(__always) public static func dilation2DBackpropInput<T: TensorFlowNumeric>(_ input: Tensor<T>, filter: Tensor<T>, outBackprop: Tensor<T>, strides: [Int32], rates: [Int32], padding: Padding) -> Tensor<T>
```

#### Parameters

  - input: - input: 4-D with shape `[batch, in_height, in_width, depth]`.
  - filter: - filter: 3-D with shape `[filter_height, filter_width, depth]`.

### `directedInterleaveDataset(selectorInputDataset:dataInputDatasets:outputTypes:outputShapes:)`

A substitute for `InterleaveDataset` on a fixed list of `N` datasets.

``` swift
@inlinable @inline(__always) public static func directedInterleaveDataset(selectorInputDataset: VariantHandle, dataInputDatasets: [VariantHandle], outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `div(_:_:)`

Returns x / y element-wise.

``` swift
@inlinable @inline(__always) public static func div<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `divNoNan(_:_:)`

Returns 0 if the denominator is zero.

``` swift
@inlinable @inline(__always) public static func divNoNan<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `drawBoundingBoxes(images:boxes:)`

Draw bounding boxes on a batch of images.

``` swift
@inlinable @inline(__always) public static func drawBoundingBoxes<T: FloatingPoint & TensorFlowScalar>(images: Tensor<T>, boxes: Tensor<Float>) -> Tensor<T>
```

Outputs a copy of `images` but draws on top of the pixels zero or more bounding
boxes specified by the locations in `boxes`. The coordinates of the each
bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example, if an image is 100 x 200 pixels (height x width) and the bounding
box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).

Parts of the bounding box may fall outside the image.

#### Parameters

  - images: - images: 4-D with shape `[batch, height, width, depth]`. A batch of images.
  - boxes: - boxes: 3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding boxes.

### `drawBoundingBoxesV2(images:boxes:colors:)`

Draw bounding boxes on a batch of images.

``` swift
@inlinable @inline(__always) public static func drawBoundingBoxesV2<T: FloatingPoint & TensorFlowScalar>(images: Tensor<T>, boxes: Tensor<Float>, colors: Tensor<Float>) -> Tensor<T>
```

Outputs a copy of `images` but draws on top of the pixels zero or more bounding
boxes specified by the locations in `boxes`. The coordinates of the each
bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example, if an image is 100 x 200 pixels (height x width) and the bounding
box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
the bounding box will be `(40, 10)` to `(100, 50)` (in (x,y) coordinates).

Parts of the bounding box may fall outside the image.

#### Parameters

  - images: - images: 4-D with shape `[batch, height, width, depth]`. A batch of images.
  - boxes: - boxes: 3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding boxes.
  - colors: - colors: 2-D. A list of RGBA colors to cycle through for the boxes.

### `dynamicPartition(data:partitions:numPartitions:)`

Partitions `data` into `num_partitions` tensors using indices from `partitions`.

``` swift
@inlinable @inline(__always) public static func dynamicPartition<T: TensorFlowScalar>(data: Tensor<T>, partitions: Tensor<Int32>, numPartitions: Int64) -> [Tensor<T>]
```

For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
are placed in `outputs[i]` in lexicographic order of `js`, and the first
dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
In detail,

``` python
    outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]
 
    outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
```

`data.shape` must start with `partitions.shape`.

For example:

``` python
    # Scalar partitions.
    partitions = 1
    num_partitions = 2
    data = [10, 20]
    outputs[0] = []  # Empty with shape [0, 2]
    outputs[1] = [[10, 20]]
 
    # Vector partitions.
    partitions = [0, 0, 1, 1, 0]
    num_partitions = 2
    data = [10, 20, 30, 40, 50]
    outputs[0] = [10, 20, 50]
    outputs[1] = [30, 40]
```

See `dynamic_stitch` for an example on how to merge partitions back.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/DynamicPartition.png" alt>
</div>

#### Parameters

  - partitions: - partitions: Any shape.  Indices in the range `[0, num_partitions)`.

### `dynamicStitch(indices:data:)`

Interleave the values from the `data` tensors into a single tensor.

``` swift
@inlinable @inline(__always) public static func dynamicStitch<T: TensorFlowScalar>(indices: [Tensor<Int32>], data: [Tensor<T>]) -> Tensor<T>
```

Builds a merged tensor such that

``` python
    merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
```

For example, if each `indices[m]` is scalar or vector, we have

``` python
    # Scalar indices:
    merged[indices[m], ...] = data[m][...]
 
    # Vector indices:
    merged[indices[m][i], ...] = data[m][i, ...]
```

Each `data[i].shape` must start with the corresponding `indices[i].shape`,
and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
must have `data[i].shape = indices[i].shape + constant`.  In terms of this
`constant`, the output shape is

``` 
merged.shape = [max(indices)] + constant
```

Values are merged in order, so if an index appears in both `indices[m][i]` and
`indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
merged result. If you do not need this guarantee, ParallelDynamicStitch might
perform better on some devices.

For example:

``` python
    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]
```

This method can be used to merge partitions created by `dynamic_partition`
as illustrated on the following example:

``` python
    # Apply function (increments x_i) on elements for which a certain condition
    # apply (x_i != -1 in this example).
    x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
    condition_mask=tf.not_equal(x,tf.constant(-1.))
    partitioned_data = tf.dynamic_partition(
        x, tf.cast(condition_mask, tf.int32) , 2)
    partitioned_data[1] = partitioned_data[1] + 1.0
    condition_indices = tf.dynamic_partition(
        tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
    x = tf.dynamic_stitch(condition_indices, partitioned_data)
    # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
    # unchanged.
```

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
</div>

### `eagerPyFunc(_:token:isAsync:)`

Eagerly executes a python function to compute func(input)-\>output. The

``` swift
@inlinable @inline(__always) public static func eagerPyFunc<Tin: TensorArrayProtocol, Tout: TensorGroup>(_ input: Tin, token: String, isAsync: Bool = false) -> Tout
```

semantics of the input, output, and attributes are the same as those for
PyFunc.

### `editDistance(hypothesisIndices:hypothesisValues:hypothesisShape:truthIndices:truthValues:truthShape:normalize:)`

Computes the (possibly normalized) Levenshtein Edit Distance.

``` swift
@inlinable @inline(__always) public static func editDistance<T: TensorFlowScalar>(hypothesisIndices: Tensor<Int64>, hypothesisValues: Tensor<T>, hypothesisShape: Tensor<Int64>, truthIndices: Tensor<Int64>, truthValues: Tensor<T>, truthShape: Tensor<Int64>, normalize: Bool = true) -> Tensor<Float>
```

The inputs are variable-length sequences provided by SparseTensors
(hypothesis\_indices, hypothesis\_values, hypothesis\_shape)
and
(truth\_indices, truth\_values, truth\_shape).

The inputs are:

### `eig(_:computeV:)`

Computes the eigen decomposition of one or more square matrices.

``` swift
@inlinable @inline(__always) public static func eig<T: FloatingPoint & TensorFlowScalar, Tout: TensorFlowScalar>(_ input: Tensor<T>, computeV: Bool = true) -> (e: Tensor<Tout>, v: Tensor<Tout>)
```

Computes the eigenvalues and (optionally) right eigenvectors of each inner matrix in
`input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`. The eigenvalues
are sorted in non-decreasing order.

``` python
# a is a tensor.
# e is a tensor of eigenvalues.
# v is a tensor of eigenvectors.
e, v = eig(a)
e = eig(a, compute_v=False)
```

#### Parameters

  - input: - input: `Tensor` input of shape `[N, N]`.

### `einsum(inputs:equation:)`

Tensor contraction according to Einstein summation convention.

``` swift
@inlinable @inline(__always) public static func einsum<T: TensorFlowScalar>(inputs: [Tensor<T>], equation: String) -> Tensor<T>
```

Implements generalized Tensor contraction and reduction. Each input Tensor must
have a corresponding input subscript appearing in the comma-separated left-hand
side of the equation. The right-hand side of the equation consists of the
output subscript. The input subscripts and the output subscript should consist
of zero or more named axis labels and at most one ellipsis (`...`).

The named axis labels may be any single character other than those having
special meaning, namely `,.->`. The behavior of this Op is undefined if it
receives an ill-formatted equation; since the validation is done at
graph-building time, we omit format validation checks at runtime.

Note: This Op is *not* intended to be called by the user; instead users should
call `tf.einsum` directly. It is a hidden Op used by `tf.einsum`.

Operations are applied to the input(s) according to the following rules:

(a) Generalized Diagonals: For input dimensions corresponding to axis labels
appearing more than once in the same input subscript, we take the
generalized (`k`-dimensional) diagonal.
For example, in the equation `iii->i` with input shape `[3, 3, 3]`, the
generalized diagonal would consist of `3` elements at indices `(0, 0, 0)`,
`(1, 1, 1)` and `(2, 2, 2)` to create a Tensor of shape `[3]`.

(b) Reduction: Axes corresponding to labels appearing only in one input
subscript but not in the output subscript are summed over prior to Tensor
contraction.
For example, in the equation `ab,bc->b`, the axis labels `a` and `c` are
the reduction axis labels.

(c) Batch Dimensions: Axes corresponding to labels appearing in each of the
input subscripts and also in the output subscript make up the batch
dimensions in Tensor contraction. Unnamed axis labels corresponding to
ellipsis (`...`) also correspond to batch dimensions.
For example, for the equation denoting batch matrix multiplication,
`bij,bjk->bik`, the axis label `b` corresponds to a batch dimension.

(d) Contraction: In case of binary einsum, axes corresponding to labels
appearing in two different inputs (and not in the output) are contracted
against each other.
Considering the batch matrix multiplication equation again
(`bij,bjk->bik`), the contracted axis label is `j`.

(e) Expand Diagonal: If the output subscripts contain repeated (explicit) axis
labels, the opposite operation of (a) is applied. For example, in the
equation `i->iii`, and input shape `[3]`, the output of shape `[3, 3, 3]`
are all zeros, except for the (generalized) diagonal which is populated
with values from the input.
Note: This operation is not supported by `np.einsum` or `tf.einsum`; it is
provided to enable computing the symbolic gradient of `tf.einsum`.

The output subscripts must contain only labels appearing in at least one of the
input subscripts. Furthermore, all dimensions mapping to the same axis label
must be equal.

Any of the input and output subscripts may contain at most a single ellipsis
(`...`). These ellipsis are mapped against dimensions not corresponding to any
named axis label. If two inputs contain ellipsis, then they are broadcasted
according to standard NumPy broadcasting
[rules](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

The broadcasted dimensions are placed in the corresponding location of the
ellipsis in the output subscript. If the broadcasted dimensions are non-empty
and the output subscripts do not contain ellipsis, then an InvalidArgument error
is raised.

@compatibility(numpy)
Similar to [`numpy.einsum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html).

Comparison with `numpy.einsum`:

#### Parameters

  - inputs: - inputs: List of 1 or 2 Tensors.

### `elu(features:)`

Computes exponential linear: `exp(features) - 1` if \< 0, `features` otherwise.

``` swift
@inlinable @inline(__always) public static func elu<T: FloatingPoint & TensorFlowScalar>(features: Tensor<T>) -> Tensor<T>
```

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)

### `eluGrad(gradients:outputs:)`

Computes gradients for the exponential linear (Elu) operation.

``` swift
@inlinable @inline(__always) public static func eluGrad<T: FloatingPoint & TensorFlowScalar>(gradients: Tensor<T>, outputs: Tensor<T>) -> Tensor<T>
```

#### Parameters

  - gradients: - gradients: The backpropagated gradients to the corresponding Elu operation.
  - outputs: - outputs: The outputs of the corresponding Elu operation.

### `empty(shape:init_:)`

Creates a tensor with the given shape.

``` swift
@inlinable @inline(__always) public static func empty<Dtype: TensorFlowScalar>(shape: Tensor<Int32>, init_: Bool = false) -> Tensor<Dtype>
```

This operation creates a tensor of `shape` and `dtype`.

#### Parameters

  - shape: - shape: 1-D. Represents the shape of the output tensor.

### `emptyTensorList(elementShape:maxNumElements:elementDtype:)`

Creates and returns an empty tensor list.

``` swift
@inlinable @inline(__always) public static func emptyTensorList<ShapeType: TensorFlowIndex>(elementShape: Tensor<ShapeType>, maxNumElements: Tensor<Int32>, elementDtype: TensorDataType) -> VariantHandle
```

All list elements must be tensors of dtype element\_dtype and shape compatible
with element\_shape.

handle: an empty tensor list.
element\_dtype: the type of elements in the list.
element\_shape: a shape compatible with that of elements in the list.

### `encodeBase64(_:pad:)`

Encode strings into web-safe base64 format.

``` swift
@inlinable @inline(__always) public static func encodeBase64(_ input: StringTensor, pad: Bool = false) -> StringTensor
```

Refer to the following article for more information on base64 format:
en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
end so that the encoded has length multiple of 4. See Padding section of the
link above.

Web-safe means that the encoder uses - and \_ instead of + and /.

#### Parameters

  - input: - input: Strings to be encoded.

### `encodeJpeg(image:format:quality:progressive:optimizeSize:chromaDownsampling:densityUnit:xDensity:yDensity:xmpMetadata:)`

JPEG-encode an image.

``` swift
@inlinable @inline(__always) public static func encodeJpeg(image: Tensor<UInt8>, format: Format, quality: Int64 = 95, progressive: Bool = false, optimizeSize: Bool = false, chromaDownsampling: Bool = true, densityUnit: DensityUnit = .in_, xDensity: Int64 = 300, yDensity: Int64 = 300, xmpMetadata: String) -> StringTensor
```

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.

The attr `format` can be used to override the color format of the encoded
output.  Values can be:

If `format` is not specified or is the empty string, a default format is picked
in function of the number of channels in `image`:

#### Parameters

  - image: - image: 3-D with shape `[height, width, channels]`.

### `encodeJpegVariableQuality(images:quality:)`

JPEG encode input image with provided compression quality.

``` swift
@inlinable @inline(__always) public static func encodeJpegVariableQuality(images: Tensor<UInt8>, quality: Tensor<Int32>) -> StringTensor
```

`image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
`quality` is an int32 jpeg compression quality value between 0 and 100.

#### Parameters

  - images: - images: Images to adjust.  At least 3-D.
  - quality: - quality: An int quality to encode to.

### `encodePng(image:compression:)`

PNG-encode an image.

``` swift
@inlinable @inline(__always) public static func encodePng<T: UnsignedInteger & TensorFlowScalar>(image: Tensor<T>, compression: Int64 = -1) -> StringTensor
```

`image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
where `channels` is:

The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
default or a value from 0 to 9.  9 is the highest compression level, generating
the smallest output, but is slower.

#### Parameters

  - image: - image: 3-D with shape `[height, width, channels]`.

### `encodeProto(sizes:_:fieldNames:messageType:descriptorSource:)`

The op serializes protobuf messages provided in the input tensors.

``` swift
@inlinable @inline(__always) public static func encodeProto<TinputTypes: TensorArrayProtocol>(sizes: Tensor<Int32>, _ values: TinputTypes, fieldNames: [String], messageType: String, descriptorSource: String = "local://") -> StringTensor
```

The types of the tensors in `values` must match the schema for the fields
specified in `field_names`. All the tensors in `values` must have a common
shape prefix, *batch\_shape*.

The `sizes` tensor specifies repeat counts for each field.  The repeat count
(last dimension) of a each tensor in `values` must be greater than or equal
to corresponding repeat count in `sizes`.

A `message_type` name must be provided to give context for the field names.
The actual message descriptor can be looked up either in the linked-in
descriptor pool or a filename provided by the caller using the
`descriptor_source` attribute.

For the most part, the mapping between Proto field types and TensorFlow dtypes
is straightforward. However, there are a few special cases:

The `descriptor_source` attribute selects the source of protocol
descriptors to consult when looking up `message_type`. This may be:

#### Parameters

  - sizes: - sizes: Tensor of int32 with shape `[batch_shape, len(field_names)]`.
  - values: - values: List of tensors containing values for the corresponding field.

### `encodeWav(audio:sampleRate:)`

Encode audio data using the WAV file format.

``` swift
@inlinable @inline(__always) public static func encodeWav(audio: Tensor<Float>, sampleRate: Tensor<Int32>) -> StringTensor
```

This operation will generate a string suitable to be saved out to create a .wav
audio file. It will be encoded in the 16-bit PCM format. It takes in float
values in the range -1.0f to 1.0f, and any outside that value will be clamped to
that range.

`audio` is a 2-D float Tensor of shape `[length, channels]`.
`sample_rate` is a scalar Tensor holding the rate to use (e.g. 44100).

#### Parameters

  - audio: - audio: 2-D with shape `[length, channels]`.

### `enqueueTPUEmbeddingIntegerBatch(batch:modeOverride:deviceOrdinal:)`

An op that enqueues a list of input batch tensors to TPUEmbedding.

``` swift
@inlinable @inline(__always) public static func enqueueTPUEmbeddingIntegerBatch(batch: [Tensor<Int32>], modeOverride: StringTensor, deviceOrdinal: Int64 = -1)
```

#### Parameters

  - batch: - batch: A list of 1D tensors, one for each embedding table, containing the indices into the tables.

### `enqueueTPUEmbeddingSparseBatch(sampleIndices:embeddingIndices:aggregationWeights:modeOverride:deviceOrdinal:combiners:)`

An op that enqueues TPUEmbedding input indices from a SparseTensor.

``` swift
@inlinable @inline(__always) public static func enqueueTPUEmbeddingSparseBatch<T1: TensorFlowIndex, T2: TensorFlowIndex, T3: FloatingPoint & TensorFlowScalar>(sampleIndices: [Tensor<T1>], embeddingIndices: [Tensor<T2>], aggregationWeights: [Tensor<T3>], modeOverride: StringTensor, deviceOrdinal: Int64 = -1, combiners: [String])
```

This Op eases the porting of code that uses embedding\_lookup\_sparse(),
although some Python preprocessing of the SparseTensor arguments to
embedding\_lookup\_sparse() is required to produce the arguments to this Op,
since only a single EnqueueTPUEmbeddingSparseBatch Op is allowed per training
step.

The tensors at corresponding positions in the three input lists
must have the same shape, i.e. rank 1 with dim\_size() equal to the total
number of lookups into the table described by the corresponding table\_id.

### `enqueueTPUEmbeddingSparseTensorBatch(sampleIndices:embeddingIndices:aggregationWeights:modeOverride:deviceOrdinal:combiners:tableIds:maxSequenceLengths:)`

Eases the porting of code that uses tf.nn.embedding\_lookup\_sparse().

``` swift
@inlinable @inline(__always) public static func enqueueTPUEmbeddingSparseTensorBatch<T1: TensorFlowIndex, T2: TensorFlowIndex, T3: FloatingPoint & TensorFlowScalar>(sampleIndices: [Tensor<T1>], embeddingIndices: [Tensor<T2>], aggregationWeights: [Tensor<T3>], modeOverride: StringTensor, deviceOrdinal: Int64 = -1, combiners: [String], tableIds: [Int32], maxSequenceLengths: [Int32])
```

sample\_indices\[i\], embedding\_indices\[i\] and aggregation\_weights\[i\] correspond
to the ith feature. table\_ids\[i\] indicates which embedding table to look up ith
feature.

The tensors at corresponding positions in the three input lists (sample\_indices,
embedding\_indices and aggregation\_weights) must have the same shape, i.e. rank 1
with dim\_size() equal to the total number of lookups into the table described by
the corresponding feature.

### `ensureShape(_:shape:)`

Ensures that the tensor's shape matches the expected shape.

``` swift
@inlinable @inline(__always) public static func ensureShape<T: TensorFlowScalar>(_ input: Tensor<T>, shape: TensorShape?) -> Tensor<T>
```

Raises an error if the input tensor's shape does not match the specified shape.
Returns the input tensor otherwise.

#### Parameters

  - input: - input: A tensor, whose shape is to be validated.

### `enter(data:frameName:isConstant:parallelIterations:)`

Creates or finds a child frame, and makes `data` available to the child frame.

``` swift
@inlinable @inline(__always) public static func enter<T: TensorFlowScalar>(data: Tensor<T>, frameName: String, isConstant: Bool = false, parallelIterations: Int64 = 10) -> Tensor<T>
```

This op is used together with `Exit` to create loops in the graph.
The unique `frame_name` is used by the `Executor` to identify frames. If
`is_constant` is true, `output` is a constant in the child frame; otherwise
it may be changed in the child frame. At most `parallel_iterations` iterations
are run in parallel in the child frame.

#### Parameters

  - data: - data: The tensor to be made available to the child frame.

### `equal(_:_:incompatibleShapeError:)`

Returns the truth value of (x == y) element-wise.

``` swift
@inlinable @inline(__always) public static func equal<T: TensorFlowScalar>(_ x: Tensor<T>, _ y: Tensor<T>, incompatibleShapeError: Bool = true) -> Tensor<Bool>
```

*NOTE*: `Equal` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

``` python
x = tf.constant([2, 4])
y = tf.constant(2)
tf.math.equal(x, y) ==> array([True, False])
 
x = tf.constant([2, 4])
y = tf.constant([2, 4])
tf.math.equal(x, y) ==> array([True,  True])
```

### `equal(_:_:incompatibleShapeError:)`

Returns the truth value of (x == y) element-wise.

``` swift
@inlinable @inline(__always) public static func equal(_ x: StringTensor, _ y: StringTensor, incompatibleShapeError: Bool = true) -> Tensor<Bool>
```

*NOTE*: `Equal` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

``` python
x = tf.constant([2, 4])
y = tf.constant(2)
tf.math.equal(x, y) ==> array([True, False])
 
x = tf.constant([2, 4])
y = tf.constant([2, 4])
tf.math.equal(x, y) ==> array([True,  True])
```

### `erf(_:)`

Computes the Gauss error function of `x` element-wise.

``` swift
@inlinable @inline(__always) public static func erf<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

### `erfc(_:)`

Computes the complementary error function of `x` element-wise.

``` swift
@inlinable @inline(__always) public static func erfc<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

### `erfinv(_:)`

``` swift
@inlinable @inline(__always) public static func erfinv<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

### `euclideanNorm(_:reductionIndices:keepDims:)`

Computes the euclidean norm of elements across dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func euclideanNorm<T: TensorFlowNumeric, Tidx: TensorFlowIndex>(_ input: Tensor<T>, reductionIndices: Tensor<Tidx>, keepDims: Bool = false) -> Tensor<T>
```

Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

#### Parameters

  - input: - input: The tensor to reduce.

### `exit(data:)`

Exits the current frame to its parent frame.

``` swift
@inlinable @inline(__always) public static func exit<T: TensorFlowScalar>(data: Tensor<T>) -> Tensor<T>
```

Exit makes its input `data` available to the parent frame.

#### Parameters

  - data: - data: The tensor to be made available to the parent frame.

### `exp(_:)`

Computes exponential of x element-wise.  \\(y = e^x\\).

``` swift
@inlinable @inline(__always) public static func exp<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

This function computes the exponential of every element in the input tensor.
i.e. `exp(x)` or `e^(x)`, where `x` is the input tensor.
`e` denotes Euler's number and is approximately equal to 2.718281.
Output is positive for any real input.

``` python
x = tf.constant(2.0)
tf.math.exp(x) ==> 7.389056

x = tf.constant([2.0, 8.0])
tf.math.exp(x) ==> array([7.389056, 2980.958], dtype=float32)
```

For complex numbers, the exponential value is calculated as follows:

``` 
e^(x+iy) = e^x * e^iy = e^x * (cos y + i sin y)
```

Let's consider complex number 1+1j as an example.
e^1 \* (cos 1 + i sin 1) = 2.7182818284590 \* (0.54030230586+0.8414709848j)

``` python
x = tf.constant(1 + 1j)
tf.math.exp(x) ==> 1.4686939399158851+2.2873552871788423j
```

### `expandDims(_:dim:)`

Inserts a dimension of 1 into a tensor's shape.

``` swift
@inlinable @inline(__always) public static func expandDims<T: TensorFlowScalar, Tdim: TensorFlowIndex>(_ input: Tensor<T>, dim: Tensor<Tdim>) -> Tensor<T>
```

Given a tensor `input`, this operation inserts a dimension of 1 at the
dimension index `axis` of `input`'s shape. The dimension index `axis` starts at
zero; if you specify a negative number for `axis` it is counted backward from
the end.

This operation is useful if you want to add a batch dimension to a single
element. For example, if you have a single image of shape `[height, width, channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
which will make the shape `[1, height, width, channels]`.

Other examples:

``` 
# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]
 
# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
```

This operation requires that:

`-1-input.dims() <= dim <= input.dims()`

This operation is related to `squeeze()`, which removes dimensions of
size 1.

#### Parameters

  - dim: - dim: 0-D (scalar). Specifies the dimension index at which to expand the shape of `input`. Must be in the range `[-rank(input) - 1, rank(input)]`.

### `experimentalAssertNextDataset(inputDataset:transformations:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func experimentalAssertNextDataset(inputDataset: VariantHandle, transformations: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalAutoShardDataset(inputDataset:numWorkers:index:autoShardPolicy:outputTypes:outputShapes:)`

Creates a dataset that shards the input dataset.

``` swift
@inlinable @inline(__always) public static func experimentalAutoShardDataset(inputDataset: VariantHandle, numWorkers: Tensor<Int64>, index: Tensor<Int64>, autoShardPolicy: Int64 = 0, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

Creates a dataset that shards the input dataset by num\_workers, returning a
sharded dataset for the index-th worker. This attempts to automatically shard
a dataset by examining the Dataset graph and inserting a shard op before the
inputs to a reader Dataset (e.g. CSVDataset, TFRecordDataset).

This dataset will throw a NotFound error if we cannot shard the dataset
automatically.

#### Parameters

  - index: - index: A scalar representing the index of the current worker out of num\_workers.

### `experimentalBytesProducedStatsDataset(inputDataset:tag:outputTypes:outputShapes:)`

Records the bytes size of each element of `input_dataset` in a StatsAggregator.

``` swift
@inlinable @inline(__always) public static func experimentalBytesProducedStatsDataset(inputDataset: VariantHandle, tag: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalCSVDataset(filenames:compressionType:bufferSize:header:fieldDelim:useQuoteDelim:naValue:selectCols:recordDefaults:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func experimentalCSVDataset<OutputTypes: TensorArrayProtocol>(filenames: StringTensor, compressionType: StringTensor, bufferSize: Tensor<Int64>, header: Tensor<Bool>, fieldDelim: StringTensor, useQuoteDelim: Tensor<Bool>, naValue: StringTensor, selectCols: Tensor<Int64>, recordDefaults: OutputTypes, outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalChooseFastestDataset(inputDatasets:numExperiments:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func experimentalChooseFastestDataset(inputDatasets: [VariantHandle], numExperiments: Int64, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalDatasetCardinality(inputDataset:)`

Returns the cardinality of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalDatasetCardinality(inputDataset: VariantHandle) -> Tensor<Int64>
```

Returns the cardinality of `input_dataset`.

### `experimentalDatasetToTFRecord(inputDataset:filename:compressionType:)`

Writes the given dataset to the given file using the TFRecord format.

``` swift
@inlinable @inline(__always) public static func experimentalDatasetToTFRecord(inputDataset: VariantHandle, filename: StringTensor, compressionType: StringTensor)
```

#### Parameters

  - filename: - filename: A scalar string tensor representing the filename to use.

### `experimentalDenseToSparseBatchDataset(inputDataset:batchSize:rowShape:outputTypes:outputShapes:)`

Creates a dataset that batches input elements into a SparseTensor.

``` swift
@inlinable @inline(__always) public static func experimentalDenseToSparseBatchDataset(inputDataset: VariantHandle, batchSize: Tensor<Int64>, rowShape: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalDirectedInterleaveDataset(selectorInputDataset:dataInputDatasets:outputTypes:outputShapes:)`

A substitute for `InterleaveDataset` on a fixed list of `N` datasets.

``` swift
@inlinable @inline(__always) public static func experimentalDirectedInterleaveDataset(selectorInputDataset: VariantHandle, dataInputDatasets: [VariantHandle], outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalGroupByReducerDataset(inputDataset:keyFuncOtherArguments:initFuncOtherArguments:reduceFuncOtherArguments:finalizeFuncOtherArguments:keyFunc:initFunc:reduceFunc:finalizeFunc:outputTypes:outputShapes:)`

Creates a dataset that computes a group-by on `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalGroupByReducerDataset<KeyfuncIn: TensorGroup, KeyfuncOut: TensorGroup, InitfuncIn: TensorGroup, InitfuncOut: TensorGroup, ReducefuncIn: TensorGroup, ReducefuncOut: TensorGroup, FinalizefuncIn: TensorGroup, FinalizefuncOut: TensorGroup, TkeyFuncOtherArguments: TensorArrayProtocol, TinitFuncOtherArguments: TensorArrayProtocol, TreduceFuncOtherArguments: TensorArrayProtocol, TfinalizeFuncOtherArguments: TensorArrayProtocol>(inputDataset: VariantHandle, keyFuncOtherArguments: TkeyFuncOtherArguments, initFuncOtherArguments: TinitFuncOtherArguments, reduceFuncOtherArguments: TreduceFuncOtherArguments, finalizeFuncOtherArguments: TfinalizeFuncOtherArguments, keyFunc: (KeyfuncIn) -> KeyfuncOut, initFunc: (InitfuncIn) -> InitfuncOut, reduceFunc: (ReducefuncIn) -> ReducefuncOut, finalizeFunc: (FinalizefuncIn) -> FinalizefuncOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

Creates a dataset that computes a group-by on `input_dataset`.

### `experimentalGroupByWindowDataset(inputDataset:keyFuncOtherArguments:reduceFuncOtherArguments:windowSizeFuncOtherArguments:keyFunc:reduceFunc:windowSizeFunc:outputTypes:outputShapes:)`

Creates a dataset that computes a windowed group-by on `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalGroupByWindowDataset<KeyfuncIn: TensorGroup, KeyfuncOut: TensorGroup, ReducefuncIn: TensorGroup, ReducefuncOut: TensorGroup, WindowsizefuncIn: TensorGroup, WindowsizefuncOut: TensorGroup, TkeyFuncOtherArguments: TensorArrayProtocol, TreduceFuncOtherArguments: TensorArrayProtocol, TwindowSizeFuncOtherArguments: TensorArrayProtocol>(inputDataset: VariantHandle, keyFuncOtherArguments: TkeyFuncOtherArguments, reduceFuncOtherArguments: TreduceFuncOtherArguments, windowSizeFuncOtherArguments: TwindowSizeFuncOtherArguments, keyFunc: (KeyfuncIn) -> KeyfuncOut, reduceFunc: (ReducefuncIn) -> ReducefuncOut, windowSizeFunc: (WindowsizefuncIn) -> WindowsizefuncOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

// TODO(mrry): Support non-int64 keys.

### `experimentalIgnoreErrorsDataset(inputDataset:outputTypes:outputShapes:)`

Creates a dataset that contains the elements of `input_dataset` ignoring errors.

``` swift
@inlinable @inline(__always) public static func experimentalIgnoreErrorsDataset(inputDataset: VariantHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalIteratorGetDevice(resource:)`

Returns the name of the device on which `resource` has been placed.

``` swift
@inlinable @inline(__always) public static func experimentalIteratorGetDevice(resource: ResourceHandle) -> StringTensor
```

### `experimentalLMDBDataset(filenames:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func experimentalLMDBDataset(filenames: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalLatencyStatsDataset(inputDataset:tag:outputTypes:outputShapes:)`

Records the latency of producing `input_dataset` elements in a StatsAggregator.

``` swift
@inlinable @inline(__always) public static func experimentalLatencyStatsDataset(inputDataset: VariantHandle, tag: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalMapAndBatchDataset(inputDataset:otherArguments:batchSize:numParallelCalls:dropRemainder:f:outputTypes:outputShapes:preserveCardinality:)`

Creates a dataset that fuses mapping with batching.

``` swift
@inlinable @inline(__always) public static func experimentalMapAndBatchDataset<FIn: TensorGroup, FOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, batchSize: Tensor<Int64>, numParallelCalls: Tensor<Int64>, dropRemainder: Tensor<Bool>, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?], preserveCardinality: Bool = false) -> VariantHandle
```

Creates a dataset that applies `f` to the outputs of `input_dataset` and then
batches `batch_size` of them.

Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
to `batch_size * num_parallel_batches` copies of `f` in parallel.

### `experimentalMapDataset(inputDataset:otherArguments:f:outputTypes:outputShapes:useInterOpParallelism:preserveCardinality:)`

Creates a dataset that applies `f` to the outputs of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalMapDataset<FIn: TensorGroup, FOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?], useInterOpParallelism: Bool = true, preserveCardinality: Bool = false) -> VariantHandle
```

### `experimentalMatchingFilesDataset(patterns:)`

``` swift
@inlinable @inline(__always) public static func experimentalMatchingFilesDataset(patterns: StringTensor) -> VariantHandle
```

### `experimentalMaxIntraOpParallelismDataset(inputDataset:maxIntraOpParallelism:outputTypes:outputShapes:)`

Creates a dataset that overrides the maximum intra-op parallelism.

``` swift
@inlinable @inline(__always) public static func experimentalMaxIntraOpParallelismDataset(inputDataset: VariantHandle, maxIntraOpParallelism: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalNonSerializableDataset(inputDataset:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func experimentalNonSerializableDataset(inputDataset: VariantHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalParallelInterleaveDataset(inputDataset:otherArguments:cycleLength:blockLength:sloppy:bufferOutputElements:prefetchInputElements:f:outputTypes:outputShapes:)`

Creates a dataset that applies `f` to the outputs of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalParallelInterleaveDataset<FIn: TensorGroup, FOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, cycleLength: Tensor<Int64>, blockLength: Tensor<Int64>, sloppy: Tensor<Bool>, bufferOutputElements: Tensor<Int64>, prefetchInputElements: Tensor<Int64>, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

The resulting dataset is similar to the `InterleaveDataset`, with the exception
that if retrieving the next value from a dataset would cause the requester to
block, it will skip that input dataset. This dataset is especially useful
when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
allows the training step to proceed so long as some data is available.

\!\! WARNING \!\! This dataset is not deterministic\!

### `experimentalParseExampleDataset(inputDataset:numParallelCalls:denseDefaults:sparseKeys:denseKeys:sparseTypes:denseShapes:outputTypes:outputShapes:sloppy:)`

Transforms `input_dataset` containing `Example` protos as vectors of DT\_STRING into a dataset of `Tensor` or `SparseTensor` objects representing the parsed features.

``` swift
@inlinable @inline(__always) public static func experimentalParseExampleDataset<Tdense: TensorArrayProtocol>(inputDataset: VariantHandle, numParallelCalls: Tensor<Int64>, denseDefaults: Tdense, sparseKeys: [String], denseKeys: [String], sparseTypes: [TensorDataType], denseShapes: [TensorShape?], outputTypes: [TensorDataType], outputShapes: [TensorShape?], sloppy: Bool = false) -> VariantHandle
```

### `experimentalPrivateThreadPoolDataset(inputDataset:numThreads:outputTypes:outputShapes:)`

Creates a dataset that uses a custom thread pool to compute `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalPrivateThreadPoolDataset(inputDataset: VariantHandle, numThreads: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalRandomDataset(seed:seed2:outputTypes:outputShapes:)`

Creates a Dataset that returns pseudorandom numbers.

``` swift
@inlinable @inline(__always) public static func experimentalRandomDataset(seed: Tensor<Int64>, seed2: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

#### Parameters

  - seed: - seed: A scalar seed for the random number generator. If either seed or seed2 is set to be non-zero, the random number generator is seeded by the given seed.  Otherwise, a random seed is used.
  - seed2: - seed2: A second scalar seed to avoid seed collision.

### `experimentalRebatchDataset(inputDataset:numReplicas:outputTypes:outputShapes:useFallback:)`

Creates a dataset that changes the batch size.

``` swift
@inlinable @inline(__always) public static func experimentalRebatchDataset(inputDataset: VariantHandle, numReplicas: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?], useFallback: Bool = true) -> VariantHandle
```

Creates a dataset that changes the batch size of the dataset to current batch
size // num\_replicas.

### `experimentalScanDataset(inputDataset:initialState:otherArguments:f:outputTypes:outputShapes:preserveCardinality:)`

Creates a dataset successively reduces `f` over the elements of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalScanDataset<FIn: TensorGroup, FOut: TensorGroup, Tstate: TensorArrayProtocol, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, initialState: Tstate, otherArguments: Targuments, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?], preserveCardinality: Bool = false) -> VariantHandle
```

### `experimentalSetStatsAggregatorDataset(inputDataset:statsAggregator:tag:counterPrefix:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func experimentalSetStatsAggregatorDataset(inputDataset: VariantHandle, statsAggregator: ResourceHandle, tag: StringTensor, counterPrefix: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalSleepDataset(inputDataset:sleepMicroseconds:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func experimentalSleepDataset(inputDataset: VariantHandle, sleepMicroseconds: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalSlidingWindowDataset(inputDataset:windowSize:windowShift:windowStride:outputTypes:outputShapes:)`

Creates a dataset that passes a sliding window over `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalSlidingWindowDataset(inputDataset: VariantHandle, windowSize: Tensor<Int64>, windowShift: Tensor<Int64>, windowStride: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalSqlDataset(driverName:dataSourceName:query:outputTypes:outputShapes:)`

Creates a dataset that executes a SQL query and emits rows of the result set.

``` swift
@inlinable @inline(__always) public static func experimentalSqlDataset(driverName: StringTensor, dataSourceName: StringTensor, query: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

#### Parameters

  - query: - query: A SQL query to execute.

### `experimentalStatsAggregatorHandle(container:sharedName:)`

Creates a statistics manager resource.

``` swift
@inlinable @inline(__always) public static func experimentalStatsAggregatorHandle(container: String, sharedName: String) -> ResourceHandle
```

### `experimentalStatsAggregatorSummary(iterator:)`

Produces a summary of any statistics recorded by the given statistics manager.

``` swift
@inlinable @inline(__always) public static func experimentalStatsAggregatorSummary(iterator: ResourceHandle) -> StringTensor
```

### `experimentalTakeWhileDataset(inputDataset:otherArguments:predicate:outputTypes:outputShapes:)`

Creates a dataset that stops iteration when predicate\` is false.

``` swift
@inlinable @inline(__always) public static func experimentalTakeWhileDataset<PredicateIn: TensorGroup, PredicateOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, predicate: (PredicateIn) -> PredicateOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

The `predicate` function must return a scalar boolean and accept the
following arguments:

### `experimentalThreadPoolDataset(inputDataset:threadPool:outputTypes:outputShapes:)`

Creates a dataset that uses a custom thread pool to compute `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalThreadPoolDataset(inputDataset: VariantHandle, threadPool: ResourceHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalThreadPoolHandle(numThreads:maxIntraOpParallelism:displayName:container:sharedName:)`

Creates a dataset that uses a custom thread pool to compute `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalThreadPoolHandle(numThreads: Int64, maxIntraOpParallelism: Int64 = 1, displayName: String, container: String, sharedName: String) -> ResourceHandle
```

### `experimentalUnbatchDataset(inputDataset:outputTypes:outputShapes:)`

A dataset that splits the elements of its input into multiple elements.

``` swift
@inlinable @inline(__always) public static func experimentalUnbatchDataset(inputDataset: VariantHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `experimentalUniqueDataset(inputDataset:outputTypes:outputShapes:)`

Creates a dataset that contains the unique elements of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func experimentalUniqueDataset(inputDataset: VariantHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `expm1(_:)`

Computes `exp(x) - 1` element-wise.

``` swift
@inlinable @inline(__always) public static func expm1<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

i.e. `exp(x) - 1` or `e^(x) - 1`, where `x` is the input tensor.
`e` denotes Euler's number and is approximately equal to 2.718281.

``` python
x = tf.constant(2.0)
tf.math.expm1(x) ==> 6.389056

x = tf.constant([2.0, 8.0])
tf.math.expm1(x) ==> array([6.389056, 2979.958], dtype=float32)

x = tf.constant(1 + 1j)
tf.math.expm1(x) ==> (0.46869393991588515+2.2873552871788423j)
```

### `extractGlimpse(_:size:offsets:centered:normalized:uniformNoise:noise:)`

Extracts a glimpse from the input tensor.

``` swift
@inlinable @inline(__always) public static func extractGlimpse(_ input: Tensor<Float>, size: Tensor<Int32>, offsets: Tensor<Float>, centered: Bool = true, normalized: Bool = true, uniformNoise: Bool = true, noise: String = "uniform") -> Tensor<Float>
```

Returns a set of windows called glimpses extracted at location
`offsets` from the input tensor. If the windows only partially
overlaps the inputs, the non overlapping areas will be filled with
random noise.

The result is a 4-D tensor of shape `[batch_size, glimpse_height, glimpse_width, channels]`. The channels and batch dimensions are the
same as that of the input tensor. The height and width of the output
windows are specified in the `size` parameter.

The argument `normalized` and `centered` controls how the windows are built:

#### Parameters

  - input: - input: A 4-D float tensor of shape `[batch_size, height, width, channels]`.
  - size: - size: A 1-D tensor of 2 elements containing the size of the glimpses to extract.  The glimpse height must be specified first, following by the glimpse width.
  - offsets: - offsets: A 2-D integer tensor of shape `[batch_size, 2]` containing the y, x locations of the center of each window.

### `extractImagePatches(images:ksizes:strides:rates:padding:)`

Extract `patches` from `images` and put them in the "depth" output dimension.

``` swift
@inlinable @inline(__always) public static func extractImagePatches<T: TensorFlowNumeric>(images: Tensor<T>, ksizes: [Int32], strides: [Int32], rates: [Int32], padding: Padding) -> Tensor<T>
```

#### Parameters

  - images: - images: 4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.

### `extractJpegShape(contents:)`

Extract the shape information of a JPEG-encoded image.

``` swift
@inlinable @inline(__always) public static func extractJpegShape<OutputType: TensorFlowIndex>(contents: StringTensor) -> Tensor<OutputType>
```

This op only parses the image header, so it is much faster than DecodeJpeg.

#### Parameters

  - contents: - contents: 0-D. The JPEG-encoded image.

### `extractVolumePatches(_:ksizes:strides:padding:)`

Extract `patches` from `input` and put them in the "depth" output dimension. 3D extension of `extract_image_patches`.

``` swift
@inlinable @inline(__always) public static func extractVolumePatches<T: TensorFlowNumeric>(_ input: Tensor<T>, ksizes: [Int32], strides: [Int32], padding: Padding) -> Tensor<T>
```

#### Parameters

  - input: - input: 5-D Tensor with shape `[batch, in_planes, in_rows, in_cols, depth]`.

### `fFT(_:)`

Fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func fFT<Tcomplex: TensorFlowScalar>(_ input: Tensor<Tcomplex>) -> Tensor<Tcomplex>
```

Computes the 1-dimensional discrete Fourier transform over the inner-most
dimension of `input`.

#### Parameters

  - input: - input: A complex tensor.

### `fFT2D(_:)`

2D fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func fFT2D<Tcomplex: TensorFlowScalar>(_ input: Tensor<Tcomplex>) -> Tensor<Tcomplex>
```

Computes the 2-dimensional discrete Fourier transform over the inner-most
2 dimensions of `input`.

#### Parameters

  - input: - input: A complex tensor.

### `fFT3D(_:)`

3D fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func fFT3D<Tcomplex: TensorFlowScalar>(_ input: Tensor<Tcomplex>) -> Tensor<Tcomplex>
```

Computes the 3-dimensional discrete Fourier transform over the inner-most 3
dimensions of `input`.

#### Parameters

  - input: - input: A complex tensor.

### `fIFOQueueV2(componentTypes:shapes:capacity:container:sharedName:)`

A queue that produces elements in first-in first-out order.

``` swift
@inlinable @inline(__always) public static func fIFOQueueV2(componentTypes: [TensorDataType], shapes: [TensorShape?], capacity: Int64 = -1, container: String, sharedName: String) -> ResourceHandle
```

### `fact()`

Output a fact about factorials.

``` swift
@inlinable @inline(__always) public static func fact() -> StringTensor
```

### `fakeParam(shape:)`

This op is used as a placeholder in If branch functions. It doesn't provide a
valid output when run, so must either be removed (e.g. replaced with a
function input) or guaranteed not to be used (e.g. if mirroring an
intermediate output needed for the gradient computation of the other branch).

``` swift
@inlinable @inline(__always) public static func fakeParam<Dtype: TensorFlowScalar>(shape: TensorShape?) -> Tensor<Dtype>
```

### `fakeQuantWithMinMaxArgs(inputs:min:max:numBits:narrowRange:)`

Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.

``` swift
@inlinable @inline(__always) public static func fakeQuantWithMinMaxArgs(inputs: Tensor<Float>, min: Double = -6, max: Double = 6, numBits: Int64 = 8, narrowRange: Bool = false) -> Tensor<Float>
```

Attributes `[min; max]` define the clamping range for the `inputs` data.
`inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
then de-quantized and output as floats in `[min; max]` interval.
`num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

Before quantization, `min` and `max` values are adjusted with the following
logic.
It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
the behavior can be unexpected:
If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
`min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

Quantization is called fake since the output is still in floating point.

### `fakeQuantWithMinMaxArgsGradient(gradients:inputs:min:max:numBits:narrowRange:)`

Compute gradients for a FakeQuantWithMinMaxArgs operation.

``` swift
@inlinable @inline(__always) public static func fakeQuantWithMinMaxArgsGradient(gradients: Tensor<Float>, inputs: Tensor<Float>, min: Double = -6, max: Double = 6, numBits: Int64 = 8, narrowRange: Bool = false) -> Tensor<Float>
```

#### Parameters

  - gradients: - gradients: Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
  - inputs: - inputs: Values passed as inputs to the FakeQuantWithMinMaxArgs operation.

### `fakeQuantWithMinMaxVars(inputs:min:max:numBits:narrowRange:)`

Fake-quantize the 'inputs' tensor of type float via global float scalars `min`

``` swift
@inlinable @inline(__always) public static func fakeQuantWithMinMaxVars(inputs: Tensor<Float>, min: Tensor<Float>, max: Tensor<Float>, numBits: Int64 = 8, narrowRange: Bool = false) -> Tensor<Float>
```

and `max` to 'outputs' tensor of same shape as `inputs`.

`[min; max]` define the clamping range for the `inputs` data.
`inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
then de-quantized and output as floats in `[min; max]` interval.
`num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

Before quantization, `min` and `max` values are adjusted with the following
logic.
It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
the behavior can be unexpected:
If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
`min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

This operation has a gradient and thus allows for training `min` and `max`
values.

### `fakeQuantWithMinMaxVarsGradient(gradients:inputs:min:max:numBits:narrowRange:)`

Compute gradients for a FakeQuantWithMinMaxVars operation.

``` swift
@inlinable @inline(__always) public static func fakeQuantWithMinMaxVarsGradient(gradients: Tensor<Float>, inputs: Tensor<Float>, min: Tensor<Float>, max: Tensor<Float>, numBits: Int64 = 8, narrowRange: Bool = false) -> (
    backpropsWrtInput: Tensor<Float>, backpropWrtMin: Tensor<Float>,
    backpropWrtMax: Tensor<Float>
  )
```

#### Parameters

  - gradients: - gradients: Backpropagated gradients above the FakeQuantWithMinMaxVars operation.
  - inputs: - inputs: Values passed as inputs to the FakeQuantWithMinMaxVars operation. min, max: Quantization interval, scalar floats.

### `fakeQuantWithMinMaxVarsPerChannel(inputs:min:max:numBits:narrowRange:)`

Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`,

``` swift
@inlinable @inline(__always) public static func fakeQuantWithMinMaxVarsPerChannel(inputs: Tensor<Float>, min: Tensor<Float>, max: Tensor<Float>, numBits: Int64 = 8, narrowRange: Bool = false) -> Tensor<Float>
```

`[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
to 'outputs' tensor of same shape as `inputs`.

`[min; max]` define the clamping range for the `inputs` data.
`inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
then de-quantized and output as floats in `[min; max]` interval.
`num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

Before quantization, `min` and `max` values are adjusted with the following
logic.
It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
the behavior can be unexpected:
If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
`min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

This operation has a gradient and thus allows for training `min` and `max`
values.

### `fakeQuantWithMinMaxVarsPerChannelGradient(gradients:inputs:min:max:numBits:narrowRange:)`

Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.

``` swift
@inlinable @inline(__always) public static func fakeQuantWithMinMaxVarsPerChannelGradient(gradients: Tensor<Float>, inputs: Tensor<Float>, min: Tensor<Float>, max: Tensor<Float>, numBits: Int64 = 8, narrowRange: Bool = false) -> (
    backpropsWrtInput: Tensor<Float>, backpropWrtMin: Tensor<Float>,
    backpropWrtMax: Tensor<Float>
  )
```

#### Parameters

  - gradients: - gradients: Backpropagated gradients above the FakeQuantWithMinMaxVars operation, shape one of: `[d]`, `[b, d]`,  `[b, h, w, d]`.
  - inputs: - inputs: Values passed as inputs to the FakeQuantWithMinMaxVars operation, shape same as `gradients`. min, max: Quantization interval, floats of shape `[d]`.

### `fill(dims:value:)`

Creates a tensor filled with a scalar value.

``` swift
@inlinable @inline(__always) public static func fill<T: TensorFlowScalar, IndexType: TensorFlowIndex>(dims: Tensor<IndexType>, value: Tensor<T>) -> Tensor<T>
```

This operation creates a tensor of shape `dims` and fills it with `value`.

For example:

``` 
# Output tensor has shape [2, 3].
fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
```

`tf.fill` differs from `tf.constant` in a few ways:

#### Parameters

  - dims: - dims: 1-D. Represents the shape of the output tensor.
  - value: - value: @compatibility(numpy)
    Equivalent to np.full
    @end\_compatibility

### `filterByLastComponentDataset(inputDataset:outputTypes:outputShapes:)`

Creates a dataset containing elements of first component of `input_dataset` having true in the last component.

``` swift
@inlinable @inline(__always) public static func filterByLastComponentDataset(inputDataset: VariantHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `filterDataset(inputDataset:otherArguments:predicate:outputTypes:outputShapes:)`

Creates a dataset containing elements of `input_dataset` matching `predicate`.

``` swift
@inlinable @inline(__always) public static func filterDataset<PredicateIn: TensorGroup, PredicateOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, predicate: (PredicateIn) -> PredicateOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

The `predicate` function must return a scalar boolean and accept the
following arguments:

### `fingerprint(data:method:)`

Generates fingerprint values.

``` swift
@inlinable @inline(__always) public static func fingerprint<T: TensorFlowScalar>(data: Tensor<T>, method: StringTensor) -> Tensor<UInt8>
```

Generates fingerprint values of `data`.

Fingerprint op considers the first dimension of `data` as the batch dimension,
and `output[i]` contains the fingerprint value generated from contents in
`data[i, ...]` for all `i`.

Fingerprint op writes fingerprint values as byte arrays. For example, the
default method `farmhash64` generates a 64-bit fingerprint value at a time.
This 8-byte value is written out as an `uint8` array of size 8, in little-endian
order.

For example, suppose that `data` has data type `DT_INT32` and shape (2, 3, 4),
and that the fingerprint method is `farmhash64`. In this case, the output shape
is (2, 8), where 2 is the batch dimension size of `data`, and 8 is the size of
each fingerprint value in bytes. `output[0, :]` is generated from 12 integers in
`data[0, :, :]` and similarly `output[1, :]` is generated from other 12 integers
in `data[1, :, :]`.

Note that this op fingerprints the raw underlying buffer, and it does not
fingerprint Tensor's metadata such as data type and/or shape. For example, the
fingerprint values are invariant under reshapes and bitcasts as long as the
batch dimension remain the same:

``` 
Fingerprint(data) == Fingerprint(Reshape(data, ...))
Fingerprint(data) == Fingerprint(Bitcast(data, ...))
```

For string data, one should expect `Fingerprint(data) != Fingerprint(ReduceJoin(data))` in general.

#### Parameters

  - data: - data: Must have rank 1 or higher.
  - method: - method: Fingerprint method used by this op. Currently available method is `farmhash::fingerprint64`.

### `fiveFloatOutputs()`

``` swift
@inlinable @inline(__always) public static func fiveFloatOutputs() -> (
    a: Tensor<Float>, b: Tensor<Float>, c: Tensor<Float>, d: Tensor<Float>, e: Tensor<Float>
  )
```

### `fixedLengthRecordDataset(filenames:headerBytes:recordBytes:footerBytes:bufferSize:)`

Creates a dataset that emits the records from one or more binary files.

``` swift
@inlinable @inline(__always) public static func fixedLengthRecordDataset(filenames: StringTensor, headerBytes: Tensor<Int64>, recordBytes: Tensor<Int64>, footerBytes: Tensor<Int64>, bufferSize: Tensor<Int64>) -> VariantHandle
```

#### Parameters

  - filenames: - filenames: A scalar or a vector containing the name(s) of the file(s) to be read.

### `fixedLengthRecordDatasetV2(filenames:headerBytes:recordBytes:footerBytes:bufferSize:compressionType:)`

``` swift
@inlinable @inline(__always) public static func fixedLengthRecordDatasetV2(filenames: StringTensor, headerBytes: Tensor<Int64>, recordBytes: Tensor<Int64>, footerBytes: Tensor<Int64>, bufferSize: Tensor<Int64>, compressionType: StringTensor) -> VariantHandle
```

### `fixedLengthRecordReaderV2(headerBytes:recordBytes:footerBytes:hopBytes:container:sharedName:encoding:)`

A Reader that outputs fixed-length records from a file.

``` swift
@inlinable @inline(__always) public static func fixedLengthRecordReaderV2(headerBytes: Int64 = 0, recordBytes: Int64, footerBytes: Int64 = 0, hopBytes: Int64 = 0, container: String, sharedName: String, encoding: String) -> ResourceHandle
```

### `fixedUnigramCandidateSampler(trueClasses:numTrue:numSampled:unique:rangeMax:vocabFile:distortion:numReservedIds:numShards:shard:unigrams:seed:seed2:)`

Generates labels for candidate sampling with a learned unigram distribution.

``` swift
@inlinable @inline(__always) public static func fixedUnigramCandidateSampler(trueClasses: Tensor<Int64>, numTrue: Int64, numSampled: Int64, unique: Bool, rangeMax: Int64, vocabFile: String, distortion: Double = 1, numReservedIds: Int64 = 0, numShards: Int64 = 1, shard: Int64 = 0, unigrams: [Double], seed: Int64 = 0, seed2: Int64 = 0) -> (
    sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>,
    sampledExpectedCount: Tensor<Float>
  )
```

A unigram sampler could use a fixed unigram distribution read from a
file or passed in as an in-memory array instead of building up the distribution
from data on the fly. There is also an option to skew the distribution by
applying a distortion power to the weights.

The vocabulary file should be in CSV-like format, with the last field
being the weight associated with the word.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

### `flatMapDataset(inputDataset:otherArguments:f:outputTypes:outputShapes:)`

Creates a dataset that applies `f` to the outputs of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func flatMapDataset<FIn: TensorGroup, FOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
Dataset variant, and FlatMapDataset will flatten successive results
into a single Dataset.

### `floatInput(_:)`

``` swift
@inlinable @inline(__always) public static func floatInput(_ a: Tensor<Float>)
```

### `floatOutput()`

``` swift
@inlinable @inline(__always) public static func floatOutput() -> Tensor<Float>
```

### `floatOutputStringOutput()`

``` swift
@inlinable @inline(__always) public static func floatOutputStringOutput() -> (a: Tensor<Float>, b: StringTensor)
```

### `floor(_:)`

Returns element-wise largest integer not greater than x.

``` swift
@inlinable @inline(__always) public static func floor<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

### `floorDiv(_:_:)`

Returns x // y element-wise.

``` swift
@inlinable @inline(__always) public static func floorDiv<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `floorMod(_:_:)`

Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

``` swift
@inlinable @inline(__always) public static func floorMod<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

*NOTE*: `FloorMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `flushSummaryWriter(writer:)`

``` swift
@inlinable @inline(__always) public static func flushSummaryWriter(writer: ResourceHandle)
```

### `foo1(_:_:c:)`

``` swift
@inlinable @inline(__always) public static func foo1(_ a: Tensor<Float>, _ b: Tensor<Int32>, c: Tensor<Int32>) -> (d: Tensor<Float>, e: Tensor<Int32>)
```

### `foo2(_:_:c:)`

``` swift
@inlinable @inline(__always) public static func foo2(_ a: Tensor<Float>, _ b: StringTensor, c: StringTensor) -> (d: Tensor<Float>, e: Tensor<Int32>)
```

### `foo3(_:_:c:)`

``` swift
@inlinable @inline(__always) public static func foo3(_ a: Tensor<Float>, _ b: StringTensor, c: Tensor<Float>) -> (d: Tensor<Float>, e: Tensor<Int32>)
```

### `for_(start:limit:delta:_:body:)`

``` swift
@inlinable @inline(__always) public static func for_<T: TensorArrayProtocol, BodyIn: TensorGroup, BodyOut: TensorGroup>(start: Tensor<Int32>, limit: Tensor<Int32>, delta: Tensor<Int32>, _ input: T, body: (BodyIn) -> BodyOut) -> T
```

``` python
 output = input;
 for i in range(start, limit, delta)
   output = body(i, output);
```

#### Parameters

  - start: - start: The lower bound. An int32
  - limit: - limit: The upper bound. An int32
  - delta: - delta: The increment. An int32
  - input: - input: A list of input tensors whose types are T.

### `fractionalAvgPool(value:poolingRatio:pseudoRandom:overlapping:deterministic:seed:seed2:)`

Performs fractional average pooling on the input.

``` swift
@inlinable @inline(__always) public static func fractionalAvgPool<T: TensorFlowNumeric>(value: Tensor<T>, poolingRatio: [Double], pseudoRandom: Bool = false, overlapping: Bool = false, deterministic: Bool = false, seed: Int64 = 0, seed2: Int64 = 0) -> (output: Tensor<T>, rowPoolingSequence: Tensor<Int64>, colPoolingSequence: Tensor<Int64>)
```

Fractional average pooling is similar to Fractional max pooling in the pooling
region generation step. The only difference is that after pooling regions are
generated, a mean operation is performed instead of a max operation in each
pooling region.

#### Parameters

  - value: - value: 4-D with shape `[batch, height, width, channels]`.

### `fractionalAvgPoolGrad(origInputTensorShape:outBackprop:rowPoolingSequence:colPoolingSequence:overlapping:)`

Computes gradient of the FractionalAvgPool function.

``` swift
@inlinable @inline(__always) public static func fractionalAvgPoolGrad<T: TensorFlowNumeric>(origInputTensorShape: Tensor<Int64>, outBackprop: Tensor<T>, rowPoolingSequence: Tensor<Int64>, colPoolingSequence: Tensor<Int64>, overlapping: Bool = false) -> Tensor<T>
```

Unlike FractionalMaxPoolGrad, we don't need to find arg\_max for
FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
out\_backprop to those indices that form the same pooling cell. Therefore, we
just need to know the shape of original input tensor, instead of the whole
tensor.

### `fractionalMaxPool(value:poolingRatio:pseudoRandom:overlapping:deterministic:seed:seed2:)`

Performs fractional max pooling on the input.

``` swift
@inlinable @inline(__always) public static func fractionalMaxPool<T: TensorFlowNumeric>(value: Tensor<T>, poolingRatio: [Double], pseudoRandom: Bool = false, overlapping: Bool = false, deterministic: Bool = false, seed: Int64 = 0, seed2: Int64 = 0) -> (output: Tensor<T>, rowPoolingSequence: Tensor<Int64>, colPoolingSequence: Tensor<Int64>)
```

Fractional max pooling is slightly different than regular max pooling.  In
regular max pooling, you downsize an input set by taking the maximum value of
smaller N x N subsections of the set (often 2x2), and try to reduce the set by
a factor of N, where N is an integer.  Fractional max pooling, as you might
expect from the word "fractional", means that the overall reduction ratio N
does not have to be an integer.

The sizes of the pooling regions are generated randomly but are fairly uniform.
For example, let's look at the height dimension, and the constraints on the
list of rows that will be pool boundaries.

First we define the following:

1.  input\_row\_length : the number of rows from the input set
2.  output\_row\_length : which will be smaller than the input
3.  alpha = input\_row\_length / output\_row\_length : our reduction ratio
4.  K = floor(alpha)
5.  row\_pooling\_sequence : this is the result list of pool boundary rows

Then, row\_pooling\_sequence should satisfy:

1.  a\[0\] = 0 : the first value of the sequence is 0
2.  a\[end\] = input\_row\_length : the last value of the sequence is the size
3.  K \<= (a\[i+1\] - a\[i\]) \<= K+1 : all intervals are K or K+1 size
4.  length(row\_pooling\_sequence) = output\_row\_length+1

For more details on fractional max pooling, see this paper:
[Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)

#### Parameters

  - value: - value: 4-D with shape `[batch, height, width, channels]`.

### `fractionalMaxPoolGrad(origInput:origOutput:outBackprop:rowPoolingSequence:colPoolingSequence:overlapping:)`

Computes gradient of the FractionalMaxPool function.

``` swift
@inlinable @inline(__always) public static func fractionalMaxPoolGrad<T: TensorFlowNumeric>(origInput: Tensor<T>, origOutput: Tensor<T>, outBackprop: Tensor<T>, rowPoolingSequence: Tensor<Int64>, colPoolingSequence: Tensor<Int64>, overlapping: Bool = false) -> Tensor<T>
```

### `funcAttr(f:)`

``` swift
@inlinable @inline(__always) public static func funcAttr<FIn: TensorGroup, FOut: TensorGroup>(f: (FIn) -> FOut)
```

### `fusedBatchNorm(_:scale:offset:mean:variance:epsilon:dataFormat:isTraining:)`

Batch normalization.

``` swift
@inlinable @inline(__always) public static func fusedBatchNorm<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, scale: Tensor<T>, offset: Tensor<T>, mean: Tensor<T>, variance: Tensor<T>, epsilon: Double = 0.0001, dataFormat: DataFormat = .nhwc, isTraining: Bool = true) -> (
    y: Tensor<T>, batchMean: Tensor<T>, batchVariance: Tensor<T>, reserveSpace1: Tensor<T>,
    reserveSpace2: Tensor<T>
  )
```

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

#### Parameters

  - x: - x: A 4D Tensor for input data.
  - scale: - scale: A 1D Tensor for scaling factor, to scale the normalized x.
  - offset: - offset: A 1D Tensor for offset, to shift to the normalized x.
  - mean: - mean: A 1D Tensor for population mean. Used for inference only; must be empty for training.
  - variance: - variance: A 1D Tensor for population variance. Used for inference only; must be empty for training.

### `fusedBatchNormGrad(yBackprop:_:scale:reserveSpace1:reserveSpace2:epsilon:dataFormat:isTraining:)`

Gradient for batch normalization.

``` swift
@inlinable @inline(__always) public static func fusedBatchNormGrad<T: FloatingPoint & TensorFlowScalar>(yBackprop: Tensor<T>, _ x: Tensor<T>, scale: Tensor<T>, reserveSpace1: Tensor<T>, reserveSpace2: Tensor<T>, epsilon: Double = 0.0001, dataFormat: DataFormat = .nhwc, isTraining: Bool = true) -> (
    xBackprop: Tensor<T>, scaleBackprop: Tensor<T>, offsetBackprop: Tensor<T>,
    reserveSpace3: Tensor<T>, reserveSpace4: Tensor<T>
  )
```

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

#### Parameters

  - x: - x: A 4D Tensor for input data.
  - scale: - scale: A 1D Tensor for scaling factor, to scale the normalized x.

### `fusedBatchNormGradV2(yBackprop:_:scale:reserveSpace1:reserveSpace2:epsilon:dataFormat:isTraining:)`

Gradient for batch normalization.

``` swift
@inlinable @inline(__always) public static func fusedBatchNormGradV2<T: FloatingPoint & TensorFlowScalar, U: FloatingPoint & TensorFlowScalar>(yBackprop: Tensor<T>, _ x: Tensor<T>, scale: Tensor<Float>, reserveSpace1: Tensor<U>, reserveSpace2: Tensor<U>, epsilon: Double = 0.0001, dataFormat: DataFormat = .nhwc, isTraining: Bool = true) -> (
    xBackprop: Tensor<T>, scaleBackprop: Tensor<U>, offsetBackprop: Tensor<U>,
    reserveSpace3: Tensor<U>, reserveSpace4: Tensor<U>
  )
```

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

#### Parameters

  - x: - x: A 4D Tensor for input data.
  - scale: - scale: A 1D Tensor for scaling factor, to scale the normalized x.

### `fusedBatchNormGradV3(yBackprop:_:scale:reserveSpace1:reserveSpace2:reserveSpace3:epsilon:dataFormat:isTraining:)`

Gradient for batch normalization.

``` swift
@inlinable @inline(__always) public static func fusedBatchNormGradV3<T: FloatingPoint & TensorFlowScalar, U: FloatingPoint & TensorFlowScalar>(yBackprop: Tensor<T>, _ x: Tensor<T>, scale: Tensor<Float>, reserveSpace1: Tensor<U>, reserveSpace2: Tensor<U>, reserveSpace3: Tensor<U>, epsilon: Double = 0.0001, dataFormat: DataFormat = .nhwc, isTraining: Bool = true) -> (
    xBackprop: Tensor<T>, scaleBackprop: Tensor<U>, offsetBackprop: Tensor<U>,
    reserveSpace4: Tensor<U>, reserveSpace5: Tensor<U>
  )
```

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

#### Parameters

  - x: - x: A 4D Tensor for input data.
  - scale: - scale: A 1D Tensor for scaling factor, to scale the normalized x.

### `fusedBatchNormV2(_:scale:offset:mean:variance:epsilon:dataFormat:isTraining:)`

Batch normalization.

``` swift
@inlinable @inline(__always) public static func fusedBatchNormV2<T: FloatingPoint & TensorFlowScalar, U: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, scale: Tensor<U>, offset: Tensor<U>, mean: Tensor<U>, variance: Tensor<U>, epsilon: Double = 0.0001, dataFormat: DataFormat = .nhwc, isTraining: Bool = true) -> (
    y: Tensor<T>, batchMean: Tensor<U>, batchVariance: Tensor<U>, reserveSpace1: Tensor<U>,
    reserveSpace2: Tensor<U>
  )
```

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

#### Parameters

  - x: - x: A 4D Tensor for input data.
  - scale: - scale: A 1D Tensor for scaling factor, to scale the normalized x.
  - offset: - offset: A 1D Tensor for offset, to shift to the normalized x.
  - mean: - mean: A 1D Tensor for population mean. Used for inference only; must be empty for training.
  - variance: - variance: A 1D Tensor for population variance. Used for inference only; must be empty for training.

### `fusedBatchNormV3(_:scale:offset:mean:variance:epsilon:dataFormat:isTraining:)`

Batch normalization.

``` swift
@inlinable @inline(__always) public static func fusedBatchNormV3<T: FloatingPoint & TensorFlowScalar, U: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, scale: Tensor<U>, offset: Tensor<U>, mean: Tensor<U>, variance: Tensor<U>, epsilon: Double = 0.0001, dataFormat: DataFormat = .nhwc, isTraining: Bool = true) -> (
    y: Tensor<T>, batchMean: Tensor<U>, batchVariance: Tensor<U>, reserveSpace1: Tensor<U>,
    reserveSpace2: Tensor<U>, reserveSpace3: Tensor<U>
  )
```

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

#### Parameters

  - x: - x: A 4D Tensor for input data.
  - scale: - scale: A 1D Tensor for scaling factor, to scale the normalized x.
  - offset: - offset: A 1D Tensor for offset, to shift to the normalized x.
  - mean: - mean: A 1D Tensor for population mean. Used for inference only; must be empty for training.
  - variance: - variance: A 1D Tensor for population variance. Used for inference only; must be empty for training.

### `fusedPadConv2D(_:paddings:filter:mode:strides:padding:)`

Performs a padding as a preprocess during a convolution.

``` swift
@inlinable @inline(__always) public static func fusedPadConv2D<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, paddings: Tensor<Int32>, filter: Tensor<T>, mode: Mode6, strides: [Int32], padding: Padding) -> Tensor<T>
```

Similar to FusedResizeAndPadConv2d, this op allows for an optimized
implementation where the spatial padding transformation stage is fused with the
im2col lookup, but in this case without the bilinear filtering required for
resizing. Fusing the padding prevents the need to write out the intermediate
results as whole tensors, reducing memory pressure, and we can get some latency
gains by merging the transformation calculations.
The data\_format attribute for Conv2D isn't supported by this op, and 'NHWC'
order is used instead.
Internally this op uses a single per-graph scratch buffer, which means that it
will block if multiple versions are being run in parallel. This is because this
operator is primarily an optimization to minimize memory usage.

#### Parameters

  - input: - input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
  - paddings: - paddings: A two-column matrix specifying the padding sizes. The number of rows must be the same as the rank of `input`.
  - filter: - filter: 4-D with shape `[filter_height, filter_width, in_channels, out_channels]`.

### `fusedResizeAndPadConv2D(_:size:paddings:filter:resizeAlignCorners:mode:strides:padding:)`

Performs a resize and padding as a preprocess during a convolution.

``` swift
@inlinable @inline(__always) public static func fusedResizeAndPadConv2D<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, size: Tensor<Int32>, paddings: Tensor<Int32>, filter: Tensor<T>, resizeAlignCorners: Bool = false, mode: Mode6, strides: [Int32], padding: Padding) -> Tensor<T>
```

It's often possible to do spatial transformations more efficiently as part of
the packing stage of a convolution, so this op allows for an optimized
implementation where these stages are fused together. This prevents the need to
write out the intermediate results as whole tensors, reducing memory pressure,
and we can get some latency gains by merging the transformation calculations.
The data\_format attribute for Conv2D isn't supported by this op, and defaults to
'NHWC' order.
Internally this op uses a single per-graph scratch buffer, which means that it
will block if multiple versions are being run in parallel. This is because this
operator is primarily an optimization to minimize memory usage.

#### Parameters

  - input: - input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
  - size: - size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new size for the images.
  - paddings: - paddings: A two-column matrix specifying the padding sizes. The number of rows must be the same as the rank of `input`.
  - filter: - filter: 4-D with shape `[filter_height, filter_width, in_channels, out_channels]`.

### `gRUBlockCell(_:hPrev:wRu:wC:bRu:bC:)`

Computes the GRU cell forward propagation for 1 time step.

``` swift
@inlinable @inline(__always) public static func gRUBlockCell<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, hPrev: Tensor<T>, wRu: Tensor<T>, wC: Tensor<T>, bRu: Tensor<T>, bC: Tensor<T>) -> (r: Tensor<T>, u: Tensor<T>, c: Tensor<T>, h: Tensor<T>)
```

Args
x: Input to the GRU cell.
h\_prev: State input from the previous GRU cell.
w\_ru: Weight matrix for the reset and update gate.
w\_c: Weight matrix for the cell connection gate.
b\_ru: Bias vector for the reset and update gate.
b\_c: Bias vector for the cell connection gate.

Returns
r: Output of the reset gate.
u: Output of the update gate.
c: Output of the cell connection gate.
h: Current state of the GRU cell.

Note on notation of the variables:

Concatenation of a and b is represented by a\_b
Element-wise dot product of a and b is represented by ab
Element-wise dot product is represented by \\circ
Matrix multiplication is represented by \*

Biases are initialized with :
`b_ru` - constant\_initializer(1.0)
`b_c` - constant\_initializer(0.0)

This kernel op implements the following mathematical equations:

``` 
x_h_prev = [x, h_prev]
 
[r_bar u_bar] = x_h_prev * w_ru + b_ru
 
r = sigmoid(r_bar)
u = sigmoid(u_bar)
 
h_prevr = h_prev \circ r
 
x_h_prevr = [x h_prevr]
 
c_bar = x_h_prevr * w_c + b_c
c = tanh(c_bar)
 
h = (1-u) \circ c + u \circ h_prev
```

### `gRUBlockCellGrad(_:hPrev:wRu:wC:bRu:bC:r:u:c:dH:)`

Computes the GRU cell back-propagation for 1 time step.

``` swift
@inlinable @inline(__always) public static func gRUBlockCellGrad<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, hPrev: Tensor<T>, wRu: Tensor<T>, wC: Tensor<T>, bRu: Tensor<T>, bC: Tensor<T>, r: Tensor<T>, u: Tensor<T>, c: Tensor<T>, dH: Tensor<T>) -> (dX: Tensor<T>, dHPrev: Tensor<T>, dCBar: Tensor<T>, dRBarUBar: Tensor<T>)
```

Args
x: Input to the GRU cell.
h\_prev: State input from the previous GRU cell.
w\_ru: Weight matrix for the reset and update gate.
w\_c: Weight matrix for the cell connection gate.
b\_ru: Bias vector for the reset and update gate.
b\_c: Bias vector for the cell connection gate.
r: Output of the reset gate.
u: Output of the update gate.
c: Output of the cell connection gate.
d\_h: Gradients of the h\_new wrt to objective function.

Returns
d\_x: Gradients of the x wrt to objective function.
d\_h\_prev: Gradients of the h wrt to objective function.
d\_c\_bar Gradients of the c\_bar wrt to objective function.
d\_r\_bar\_u\_bar Gradients of the r\_bar & u\_bar wrt to objective function.

This kernel op implements the following mathematical equations:

Note on notation of the variables:

Concatenation of a and b is represented by a\_b
Element-wise dot product of a and b is represented by ab
Element-wise dot product is represented by \\circ
Matrix multiplication is represented by \*

Additional notes for clarity:

`w_ru` can be segmented into 4 different matrices.

``` 
w_ru = [w_r_x w_u_x
        w_r_h_prev w_u_h_prev]
```

Similarly, `w_c` can be segmented into 2 different matrices.

``` 
w_c = [w_c_x w_c_h_prevr]
```

Same goes for biases.

``` 
b_ru = [b_ru_x b_ru_h]
b_c = [b_c_x b_c_h]
```

Another note on notation:

``` 
d_x = d_x_component_1 + d_x_component_2
 
where d_x_component_1 = d_r_bar * w_r_x^T + d_u_bar * w_r_x^T
and d_x_component_2 = d_c_bar * w_c_x^T
 
d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + d_h \circ u
where d_h_prev_componenet_1 = d_r_bar * w_r_h_prev^T + d_u_bar * w_r_h_prev^T
```

Mathematics behind the Gradients below:

``` 
d_c_bar = d_h \circ (1-u) \circ (1-c \circ c)
d_u_bar = d_h \circ (h-c) \circ u \circ (1-u)
 
d_r_bar_u_bar = [d_r_bar d_u_bar]
 
[d_x_component_1 d_h_prev_component_1] = d_r_bar_u_bar * w_ru^T
 
[d_x_component_2 d_h_prevr] = d_c_bar * w_c^T
 
d_x = d_x_component_1 + d_x_component_2
 
d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + u
```

Below calculation is performed in the python wrapper for the Gradients
(not in the gradient kernel.)

``` 
d_w_ru = x_h_prevr^T * d_c_bar
 
d_w_c = x_h_prev^T * d_r_bar_u_bar
 
d_b_ru = sum of d_r_bar_u_bar along axis = 0
 
d_b_c = sum of d_c_bar along axis = 0
```

### `gather(params:indices:validateIndices:)`

Gather slices from `params` according to `indices`.

``` swift
@inlinable @inline(__always) public static func gather<Tparams: TensorFlowScalar, Tindices: TensorFlowIndex>(params: Tensor<Tparams>, indices: Tensor<Tindices>, validateIndices: Bool = true) -> Tensor<Tparams>
```

`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

``` python
    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]
 
    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]
 
    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
```

If `indices` is a permutation and `len(indices) == params.shape[0]` then
this operation will permute `params` accordingly.

`validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
`indices` are always validated to be within range. If assigned to GPU,
out-of-bound indices result in safe but unspecified behavior, which may include
raising an error.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
</div>

### `gatherNd(params:indices:)`

Gather slices from `params` into a Tensor with shape specified by `indices`.

``` swift
@inlinable @inline(__always) public static func gatherNd<Tparams: TensorFlowScalar, Tindices: TensorFlowIndex>(params: Tensor<Tparams>, indices: Tensor<Tindices>) -> Tensor<Tparams>
```

`indices` is a K-dimensional integer tensor, best thought of as a
(K-1)-dimensional tensor of indices into `params`, where each element defines a
slice of `params`:

``` 
output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]
```

Whereas in `tf.gather` `indices` defines slices into the `axis`
dimension of `params`, in `tf.gather_nd`, `indices` defines slices into the
first `N` dimensions of `params`, where `N = indices.shape[-1]`.

The last dimension of `indices` can be at most the rank of
`params`:

``` 
indices.shape[-1] <= params.rank
```

The last dimension of `indices` corresponds to elements
(if `indices.shape[-1] == params.rank`) or slices
(if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
of `params`.  The output tensor has shape

``` 
indices.shape[:-1] + params.shape[indices.shape[-1]:]
```

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, a 0 is stored in the
corresponding output value.

Some examples below.

Simple indexing into a matrix:

``` python
    indices = [[0, 0], [1, 1]]
    params = [['a', 'b'], ['c', 'd']]
    output = ['a', 'd']
```

Slice indexing into a matrix:

``` python
    indices = [[1], [0]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['c', 'd'], ['a', 'b']]
```

Indexing into a 3-tensor:

``` python
    indices = [[1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['a1', 'b1'], ['c1', 'd1']]]
 
 
    indices = [[0, 1], [1, 0]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['c0', 'd0'], ['a1', 'b1']]
 
 
    indices = [[0, 0, 1], [1, 0, 1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = ['b0', 'b1']
```

Batched indexing into a matrix:

``` python
    indices = [[[0, 0]], [[0, 1]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['a'], ['b']]
```

Batched slice indexing into a matrix:

``` python
    indices = [[[1]], [[0]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [[['c', 'd']], [['a', 'b']]]
```

Batched indexing into a 3-tensor:

``` python
    indices = [[[1]], [[0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[[['a1', 'b1'], ['c1', 'd1']]],
              [[['a0', 'b0'], ['c0', 'd0']]]]
 
    indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['c0', 'd0'], ['a1', 'b1']],
              [['a0', 'b0'], ['c1', 'd1']]]
 
 
    indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['b0', 'b1'], ['d0', 'c1']]
```

See also `tf.gather` and `tf.batch_gather`.

#### Parameters

  - params: - params: The tensor from which to gather values.
  - indices: - indices: Index tensor.

### `gatherV2(params:indices:axis:batchDims:)`

Gather slices from `params` axis `axis` according to `indices`.

``` swift
@inlinable @inline(__always) public static func gatherV2<Tparams: TensorFlowScalar, Tindices: TensorFlowIndex, Taxis: TensorFlowIndex>(params: Tensor<Tparams>, indices: Tensor<Tindices>, axis: Tensor<Taxis>, batchDims: Int64 = 0) -> Tensor<Tparams>
```

`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `params.shape[:axis] + indices.shape + params.shape[axis + 1:]` where:

``` python
    # Scalar indices (output is rank(params) - 1).
    output[a_0, ..., a_n, b_0, ..., b_n] =
      params[a_0, ..., a_n, indices, b_0, ..., b_n]
 
    # Vector indices (output is rank(params)).
    output[a_0, ..., a_n, i, b_0, ..., b_n] =
      params[a_0, ..., a_n, indices[i], b_0, ..., b_n]
 
    # Higher rank indices (output is rank(params) + rank(indices) - 1).
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      params[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
```

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
</div>

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, a 0 is stored in the
corresponding output value.

See also `tf.batch_gather` and `tf.gather_nd`.

#### Parameters

  - params: - params: The tensor from which to gather values. Must be at least rank `axis + 1`.
  - indices: - indices: Index tensor. Must be in range `[0, params.shape[axis])`.
  - axis: - axis: The axis in `params` to gather `indices` from. Defaults to the first dimension. Supports negative indexes.

### `generateBoundingBoxProposals(scores:bboxDeltas:imageInfo:anchors:nmsThreshold:preNmsTopn:minSize:postNmsTopn:)`

This op produces Region of Interests from given bounding boxes(bbox\_deltas) encoded wrt anchors according to eq.2 in arXiv:1506.01497

``` swift
@inlinable @inline(__always) public static func generateBoundingBoxProposals(scores: Tensor<Float>, bboxDeltas: Tensor<Float>, imageInfo: Tensor<Float>, anchors: Tensor<Float>, nmsThreshold: Tensor<Float>, preNmsTopn: Tensor<Int32>, minSize: Tensor<Float>, postNmsTopn: Int64 = 300) -> (rois: Tensor<Float>, roiProbabilities: Tensor<Float>)
```

``` 
  The op selects top `pre_nms_topn` scoring boxes, decodes them with respect to anchors,
  applies non-maximal suppression on overlapping boxes with higher than
  `nms_threshold` intersection-over-union (iou) value, discarding boxes where shorter
  side is less than `min_size`.
  Inputs:
  `scores`: A 4D tensor of shape [Batch, Height, Width, Num Anchors] containing the scores per anchor at given postion
  `bbox_deltas`: is a tensor of shape [Batch, Height, Width, 4 x Num Anchors] boxes encoded to each anchor
  `anchors`: A 1D tensor of shape [4 x Num Anchors], representing the anchors.
  Outputs:
  `rois`: output RoIs, a 3D tensor of shape [Batch, post_nms_topn, 4], padded by 0 if less than post_nms_topn candidates found.
  `roi_probabilities`: probability scores of each roi in 'rois', a 2D tensor of shape [Batch,post_nms_topn], padded with 0 if needed, sorted by scores.
```

#### Parameters

  - scores: - scores: A 4-D float tensor of shape `[num_images, height, width, num_achors]` containing scores of the boxes for given anchors, can be unsorted.
  - anchors: - anchors: A 2-D float tensor of shape `[num_anchors, 4]` describing the anchor boxes. Boxes are formatted in the form \[y1, x1, y2, x2\].

### `generateVocabRemapping(newVocabFile:oldVocabFile:newVocabOffset:numNewVocab:oldVocabSize:)`

Given a path to new and old vocabulary files, returns a remapping Tensor of

``` swift
@inlinable @inline(__always) public static func generateVocabRemapping(newVocabFile: StringTensor, oldVocabFile: StringTensor, newVocabOffset: Int64, numNewVocab: Int64, oldVocabSize: Int64 = -1) -> (remapping: Tensor<Int64>, numPresent: Tensor<Int32>)
```

length `num_new_vocab`, where `remapping[i]` contains the row number in the old
vocabulary that corresponds to row `i` in the new vocabulary (starting at line
`new_vocab_offset` and up to `num_new_vocab` entities), or `-1` if entry `i`
in the new vocabulary is not in the old vocabulary.  The old vocabulary is
constrained to the first `old_vocab_size` entries if `old_vocab_size` is not the
default value of -1.

`num_vocab_offset` enables
use in the partitioned variable case, and should generally be set through
examining partitioning info.  The format of the files should be a text file,
with each line containing a single entity within the vocabulary.

For example, with `new_vocab_file` a text file containing each of the following
elements on a single line: `[f0, f1, f2, f3]`, old\_vocab\_file = \[f1, f0, f3\],
`num_new_vocab = 3, new_vocab_offset = 1`, the returned remapping would be
`[0, -1, 2]`.

The op also returns a count of how many entries in the new vocabulary
were present in the old vocabulary, which is used to calculate the number of
values to initialize in a weight matrix remapping

This functionality can be used to remap both row vocabularies (typically,
features) and column vocabularies (typically, classes) from TensorFlow
checkpoints.  Note that the partitioning logic relies on contiguous vocabularies
corresponding to div-partitioned variables.  Moreover, the underlying remapping
uses an IndexTable (as opposed to an inexact CuckooTable), so client code should
use the corresponding index\_table\_from\_file() as the FeatureColumn framework
does (as opposed to tf.feature\_to\_id(), which uses a CuckooTable).

### `generatorDataset(initFuncOtherArgs:nextFuncOtherArgs:finalizeFuncOtherArgs:initFunc:nextFunc:finalizeFunc:outputTypes:outputShapes:)`

Creates a dataset that invokes a function to generate elements.

``` swift
@inlinable @inline(__always) public static func generatorDataset<InitfuncIn: TensorGroup, InitfuncOut: TensorGroup, NextfuncIn: TensorGroup, NextfuncOut: TensorGroup, FinalizefuncIn: TensorGroup, FinalizefuncOut: TensorGroup, TinitFuncArgs: TensorArrayProtocol, TnextFuncArgs: TensorArrayProtocol, TfinalizeFuncArgs: TensorArrayProtocol>(initFuncOtherArgs: TinitFuncArgs, nextFuncOtherArgs: TnextFuncArgs, finalizeFuncOtherArgs: TfinalizeFuncArgs, initFunc: (InitfuncIn) -> InitfuncOut, nextFunc: (NextfuncIn) -> NextfuncOut, finalizeFunc: (FinalizefuncIn) -> FinalizefuncOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `getSessionHandle(value:)`

Store the input tensor in the state of the current session.

``` swift
@inlinable @inline(__always) public static func getSessionHandle<T: TensorFlowScalar>(value: Tensor<T>) -> StringTensor
```

#### Parameters

  - value: - value: The tensor to be stored.

### `getSessionHandleV2(value:)`

Store the input tensor in the state of the current session.

``` swift
@inlinable @inline(__always) public static func getSessionHandleV2<T: TensorFlowScalar>(value: Tensor<T>) -> ResourceHandle
```

#### Parameters

  - value: - value: The tensor to be stored.

### `getSessionTensor(handle:)`

Get the value of the tensor specified by its handle.

``` swift
@inlinable @inline(__always) public static func getSessionTensor<Dtype: TensorFlowScalar>(handle: StringTensor) -> Tensor<Dtype>
```

#### Parameters

  - handle: - handle: The handle for a tensor stored in the session state.

### `graphDefVersion()`

``` swift
@inlinable @inline(__always) public static func graphDefVersion() -> Tensor<Int32>
```

### `greater(_:_:)`

Returns the truth value of (x \> y) element-wise.

``` swift
@inlinable @inline(__always) public static func greater<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<Bool>
```

*NOTE*: `Greater` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Example:

``` python
x = tf.constant([5, 4, 6])
y = tf.constant([5, 2, 5])
tf.math.greater(x, y) ==> [False, True, True]
 
x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.greater(x, y) ==> [False, False, True]
```

### `greaterEqual(_:_:)`

Returns the truth value of (x \>= y) element-wise.

``` swift
@inlinable @inline(__always) public static func greaterEqual<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<Bool>
```

*NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Example:

``` python
x = tf.constant([5, 4, 6, 7])
y = tf.constant([5, 2, 5, 10])
tf.math.greater_equal(x, y) ==> [True, True, True, False]
 
x = tf.constant([5, 4, 6, 7])
y = tf.constant([5])
tf.math.greater_equal(x, y) ==> [True, False, True, True]
```

### `groupByReducerDataset(inputDataset:keyFuncOtherArguments:initFuncOtherArguments:reduceFuncOtherArguments:finalizeFuncOtherArguments:keyFunc:initFunc:reduceFunc:finalizeFunc:outputTypes:outputShapes:)`

Creates a dataset that computes a group-by on `input_dataset`.

``` swift
@inlinable @inline(__always) public static func groupByReducerDataset<KeyfuncIn: TensorGroup, KeyfuncOut: TensorGroup, InitfuncIn: TensorGroup, InitfuncOut: TensorGroup, ReducefuncIn: TensorGroup, ReducefuncOut: TensorGroup, FinalizefuncIn: TensorGroup, FinalizefuncOut: TensorGroup, TkeyFuncOtherArguments: TensorArrayProtocol, TinitFuncOtherArguments: TensorArrayProtocol, TreduceFuncOtherArguments: TensorArrayProtocol, TfinalizeFuncOtherArguments: TensorArrayProtocol>(inputDataset: VariantHandle, keyFuncOtherArguments: TkeyFuncOtherArguments, initFuncOtherArguments: TinitFuncOtherArguments, reduceFuncOtherArguments: TreduceFuncOtherArguments, finalizeFuncOtherArguments: TfinalizeFuncOtherArguments, keyFunc: (KeyfuncIn) -> KeyfuncOut, initFunc: (InitfuncIn) -> InitfuncOut, reduceFunc: (ReducefuncIn) -> ReducefuncOut, finalizeFunc: (FinalizefuncIn) -> FinalizefuncOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

Creates a dataset that computes a group-by on `input_dataset`.

### `groupByWindowDataset(inputDataset:keyFuncOtherArguments:reduceFuncOtherArguments:windowSizeFuncOtherArguments:keyFunc:reduceFunc:windowSizeFunc:outputTypes:outputShapes:)`

Creates a dataset that computes a windowed group-by on `input_dataset`.

``` swift
@inlinable @inline(__always) public static func groupByWindowDataset<KeyfuncIn: TensorGroup, KeyfuncOut: TensorGroup, ReducefuncIn: TensorGroup, ReducefuncOut: TensorGroup, WindowsizefuncIn: TensorGroup, WindowsizefuncOut: TensorGroup, TkeyFuncOtherArguments: TensorArrayProtocol, TreduceFuncOtherArguments: TensorArrayProtocol, TwindowSizeFuncOtherArguments: TensorArrayProtocol>(inputDataset: VariantHandle, keyFuncOtherArguments: TkeyFuncOtherArguments, reduceFuncOtherArguments: TreduceFuncOtherArguments, windowSizeFuncOtherArguments: TwindowSizeFuncOtherArguments, keyFunc: (KeyfuncIn) -> KeyfuncOut, reduceFunc: (ReducefuncIn) -> ReducefuncOut, windowSizeFunc: (WindowsizefuncIn) -> WindowsizefuncOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

// TODO(mrry): Support non-int64 keys.

### `guaranteeConst(_:)`

Gives a guarantee to the TF runtime that the input tensor is a constant.

``` swift
@inlinable @inline(__always) public static func guaranteeConst<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

The runtime is then free to make optimizations based on this.

Only accepts value typed tensors as inputs and rejects resource variable handles
as input.

Returns the input tensor without modification.

### `hSVToRGB(images:)`

Convert one or more images from HSV to RGB.

``` swift
@inlinable @inline(__always) public static func hSVToRGB<T: FloatingPoint & TensorFlowScalar>(images: Tensor<T>) -> Tensor<T>
```

Outputs a tensor of the same shape as the `images` tensor, containing the RGB
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

See `rgb_to_hsv` for a description of the HSV encoding.

#### Parameters

  - images: - images: 1-D or higher rank. HSV data to convert. Last dimension must be size 3.

### `hashTableV2(container:sharedName:useNodeNameSharing:keyDtype:valueDtype:)`

Creates a non-initialized hash table.

``` swift
@inlinable @inline(__always) public static func hashTableV2(container: String, sharedName: String, useNodeNameSharing: Bool = false, keyDtype: TensorDataType, valueDtype: TensorDataType) -> ResourceHandle
```

This op creates a hash table, specifying the type of its keys and values.
Before using the table you will have to initialize it.  After initialization the
table will be immutable.

### `histogramFixedWidth(_:valueRange:nbins:)`

Return histogram of values.

``` swift
@inlinable @inline(__always) public static func histogramFixedWidth<T: TensorFlowNumeric, Dtype: TensorFlowIndex>(_ values: Tensor<T>, valueRange: Tensor<T>, nbins: Tensor<Int32>) -> Tensor<Dtype>
```

Given the tensor `values`, this operation returns a rank 1 histogram counting
the number of entries in `values` that fall into every bin.  The bins are
equal width and determined by the arguments `value_range` and `nbins`.

``` python
# Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
nbins = 5
value_range = [0.0, 5.0]
new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
 
with tf.get_default_session() as sess:
  hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
  variables.global_variables_initializer().run()
  sess.run(hist) => [2, 1, 1, 0, 2]
```

#### Parameters

  - values: - values: Numeric `Tensor`.
  - nbins: - nbins: Scalar `int32 Tensor`.  Number of histogram bins.

### `histogramSummary(tag:_:)`

Outputs a `Summary` protocol buffer with a histogram.

``` swift
@inlinable @inline(__always) public static func histogramSummary<T: TensorFlowNumeric>(tag: StringTensor, _ values: Tensor<T>) -> StringTensor
```

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite.

#### Parameters

  - tag: - tag: Scalar.  Tag to use for the `Summary.Value`.
  - values: - values: Any shape. Values to use to build the histogram.

### `iFFT(_:)`

Inverse fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func iFFT<Tcomplex: TensorFlowScalar>(_ input: Tensor<Tcomplex>) -> Tensor<Tcomplex>
```

Computes the inverse 1-dimensional discrete Fourier transform over the
inner-most dimension of `input`.

#### Parameters

  - input: - input: A complex tensor.

### `iFFT2D(_:)`

Inverse 2D fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func iFFT2D<Tcomplex: TensorFlowScalar>(_ input: Tensor<Tcomplex>) -> Tensor<Tcomplex>
```

Computes the inverse 2-dimensional discrete Fourier transform over the
inner-most 2 dimensions of `input`.

#### Parameters

  - input: - input: A complex tensor.

### `iFFT3D(_:)`

Inverse 3D fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func iFFT3D<Tcomplex: TensorFlowScalar>(_ input: Tensor<Tcomplex>) -> Tensor<Tcomplex>
```

Computes the inverse 3-dimensional discrete Fourier transform over the
inner-most 3 dimensions of `input`.

#### Parameters

  - input: - input: A complex tensor.

### `iRFFT(_:fftLength:)`

Inverse real-valued fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func iRFFT<Treal: FloatingPoint & TensorFlowScalar, Tcomplex: TensorFlowScalar>(_ input: Tensor<Tcomplex>, fftLength: Tensor<Int32>) -> Tensor<Treal>
```

Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
signal over the inner-most dimension of `input`.

The inner-most dimension of `input` is assumed to be the result of `RFFT`: the
`fft_length / 2 + 1` unique components of the DFT of a real-valued signal. If
`fft_length` is not provided, it is computed from the size of the inner-most
dimension of `input` (`fft_length = 2 * (inner - 1)`). If the FFT length used to
compute `input` is odd, it should be provided since it cannot be inferred
properly.

Along the axis `IRFFT` is computed on, if `fft_length / 2 + 1` is smaller
than the corresponding dimension of `input`, the dimension is cropped. If it is
larger, the dimension is padded with zeros.

#### Parameters

  - input: - input: A complex tensor.

### `iRFFT2D(_:fftLength:)`

Inverse 2D real-valued fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func iRFFT2D<Treal: FloatingPoint & TensorFlowScalar, Tcomplex: TensorFlowScalar>(_ input: Tensor<Tcomplex>, fftLength: Tensor<Int32>) -> Tensor<Treal>
```

Computes the inverse 2-dimensional discrete Fourier transform of a real-valued
signal over the inner-most 2 dimensions of `input`.

The inner-most 2 dimensions of `input` are assumed to be the result of `RFFT2D`:
The inner-most dimension contains the `fft_length / 2 + 1` unique components of
the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
from the size of the inner-most 2 dimensions of `input`. If the FFT length used
to compute `input` is odd, it should be provided since it cannot be inferred
properly.

Along each axis `IRFFT2D` is computed on, if `fft_length` (or
`fft_length / 2 + 1` for the inner-most dimension) is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.

#### Parameters

  - input: - input: A complex tensor.

### `iRFFT3D(_:fftLength:)`

Inverse 3D real-valued fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func iRFFT3D<Treal: FloatingPoint & TensorFlowScalar, Tcomplex: TensorFlowScalar>(_ input: Tensor<Tcomplex>, fftLength: Tensor<Int32>) -> Tensor<Treal>
```

Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
signal over the inner-most 3 dimensions of `input`.

The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
The inner-most dimension contains the `fft_length / 2 + 1` unique components of
the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
from the size of the inner-most 3 dimensions of `input`. If the FFT length used
to compute `input` is odd, it should be provided since it cannot be inferred
properly.

Along each axis `IRFFT3D` is computed on, if `fft_length` (or
`fft_length / 2 + 1` for the inner-most dimension) is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.

#### Parameters

  - input: - input: A complex tensor.

### `identity(_:)`

Return a tensor with the same shape and contents as the input tensor or value.

``` swift
@inlinable @inline(__always) public static func identity<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

### `identityN(_:)`

Returns a list of tensors with the same shapes and contents as the input

``` swift
@inlinable @inline(__always) public static func identityN<T: TensorArrayProtocol>(_ input: T) -> T
```

tensors.

This op can be used to override the gradient for complicated functions. For
example, suppose y = f(x) and we wish to apply a custom function g for backprop
such that dx = g(dy). In Python,

``` python
with tf.get_default_graph().gradient_override_map(
    {'IdentityN': 'OverrideGradientWithG'}):
  y, _ = identity_n([f(x), x])
 
@tf.RegisterGradient('OverrideGradientWithG')
def ApplyG(op, dy, _):
  return [None, g(dy)]  # Do not backprop to f(x).
```

### `identityReaderV2(container:sharedName:)`

A Reader that outputs the queued work as both the key and value.

``` swift
@inlinable @inline(__always) public static func identityReaderV2(container: String, sharedName: String) -> ResourceHandle
```

To use, enqueue strings in a Queue.  ReaderRead will take the front
work string and output (work, work).

### `if_(cond:_:thenBranch:elseBranch:outputShapes:)`

output = cond ? then\_branch(input) : else\_branch(input)

``` swift
@inlinable @inline(__always) public static func if_<Tcond: TensorFlowScalar, Tin: TensorArrayProtocol, Tout: TensorGroup, ThenbranchIn: TensorGroup, ThenbranchOut: TensorGroup, ElsebranchIn: TensorGroup, ElsebranchOut: TensorGroup>(cond: Tensor<Tcond>, _ input: Tin, thenBranch: (ThenbranchIn) -> ThenbranchOut, elseBranch: (ElsebranchIn) -> ElsebranchOut, outputShapes: [TensorShape?]) -> Tout
```

#### Parameters

  - cond: - cond: A Tensor. If the tensor is a scalar of non-boolean type, the scalar is converted to a boolean according to the following rule: if the scalar is a numerical value, non-zero means `True` and zero means False; if the scalar is a string, non-empty means `True` and empty means `False`. If the tensor is not a scalar, being empty means False and being non-empty means True.
  - input: - input: A list of input tensors.

### `igamma(_:_:)`

Compute the lower regularized incomplete Gamma function `P(a, x)`.

``` swift
@inlinable @inline(__always) public static func igamma<T: FloatingPoint & TensorFlowScalar>(_ a: Tensor<T>, _ x: Tensor<T>) -> Tensor<T>
```

The lower regularized incomplete Gamma function is defined as:

\\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)

where

\\(gamma(a, x) = \\int\_{0}^{x} t^{a-1} exp(-t) dt\\)

is the lower incomplete Gamma function.

Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
Gamma function.

### `igammaGradA(_:_:)`

Computes the gradient of `igamma(a, x)` wrt `a`.

``` swift
@inlinable @inline(__always) public static func igammaGradA<T: FloatingPoint & TensorFlowScalar>(_ a: Tensor<T>, _ x: Tensor<T>) -> Tensor<T>
```

### `igammac(_:_:)`

Compute the upper regularized incomplete Gamma function `Q(a, x)`.

``` swift
@inlinable @inline(__always) public static func igammac<T: FloatingPoint & TensorFlowScalar>(_ a: Tensor<T>, _ x: Tensor<T>) -> Tensor<T>
```

The upper regularized incomplete Gamma function is defined as:

\\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)

where

\\(Gamma(a, x) = int\_{x}^{\\infty} t^{a-1} exp(-t) dt\\)

is the upper incomplete Gama function.

Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
Gamma function.

### `ignoreErrorsDataset(inputDataset:outputTypes:outputShapes:)`

Creates a dataset that contains the elements of `input_dataset` ignoring errors.

``` swift
@inlinable @inline(__always) public static func ignoreErrorsDataset(inputDataset: VariantHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `imag(_:)`

Returns the imaginary part of a complex number.

``` swift
@inlinable @inline(__always) public static func imag<T: TensorFlowScalar, Tout: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<Tout>
```

Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the imaginary part of each element in `input`. All
elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
is the real part and *b* is the imaginary part returned by this operation.

For example:

``` 
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.imag(input) ==> [4.75, 5.75]
```

### `immutableConst(shape:memoryRegionName:)`

Returns immutable tensor from memory region.

``` swift
@inlinable @inline(__always) public static func immutableConst<Dtype: TensorFlowScalar>(shape: TensorShape?, memoryRegionName: String) -> Tensor<Dtype>
```

The current implementation memmaps the tensor from a file.

### `importEvent(writer:event:)`

``` swift
@inlinable @inline(__always) public static func importEvent(writer: ResourceHandle, event: StringTensor)
```

### `inPolymorphicTwice(_:_:)`

``` swift
@inlinable @inline(__always) public static func inPolymorphicTwice<T: TensorFlowScalar>(_ a: [Tensor<T>], _ b: [Tensor<T>])
```

### `inTopK(predictions:targets:k:)`

Says whether the targets are in the top `K` predictions.

``` swift
@inlinable @inline(__always) public static func inTopK<T: TensorFlowIndex>(predictions: Tensor<Float>, targets: Tensor<T>, k: Int64) -> Tensor<Bool>
```

This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
prediction for the target class is among the top `k` predictions among
all predictions for example `i`. Note that the behavior of `InTopK` differs
from the `TopK` op in its handling of ties; if multiple classes have the
same prediction value and straddle the top-`k` boundary, all of those
classes are considered to be in the top `k`.

More formally, let

\\(predictions\_i\\) be the predictions for all classes for example `i`,
\\(targets\_i\\) be the target class for example `i`,
\\(out\_i\\) be the output for example `i`,

$$out\_i = predictions\_{i, targets\_i} \\in TopKIncludingTies(predictions\_i)$$

#### Parameters

  - predictions: - predictions: A `batch_size` x `classes` tensor.
  - targets: - targets: A `batch_size` vector of class ids.

### `inTopKV2(predictions:targets:k:)`

Says whether the targets are in the top `K` predictions.

``` swift
@inlinable @inline(__always) public static func inTopKV2<T: TensorFlowIndex>(predictions: Tensor<Float>, targets: Tensor<T>, k: Tensor<T>) -> Tensor<Bool>
```

This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
prediction for the target class is among the top `k` predictions among
all predictions for example `i`. Note that the behavior of `InTopK` differs
from the `TopK` op in its handling of ties; if multiple classes have the
same prediction value and straddle the top-`k` boundary, all of those
classes are considered to be in the top `k`.

More formally, let

\\(predictions\_i\\) be the predictions for all classes for example `i`,
\\(targets\_i\\) be the target class for example `i`,
\\(out\_i\\) be the output for example `i`,

$$out\_i = predictions\_{i, targets\_i} \\in TopKIncludingTies(predictions\_i)$$

#### Parameters

  - predictions: - predictions: A `batch_size` x `classes` tensor.
  - targets: - targets: A `batch_size` vector of class ids.
  - k: - k: Number of top elements to look at for computing precision.

### `infeedDequeue(shape:)`

A placeholder op for a value that will be fed into the computation.

``` swift
@inlinable @inline(__always) public static func infeedDequeue<Dtype: TensorFlowScalar>(shape: TensorShape?) -> Tensor<Dtype>
```

### `infeedDequeueTuple(shapes:)`

Fetches multiple values from infeed as an XLA tuple.

``` swift
@inlinable @inline(__always) public static func infeedDequeueTuple<Dtypes: TensorGroup>(shapes: [TensorShape?]) -> Dtypes
```

### `infeedEnqueue(_:shape:layout:deviceOrdinal:)`

An op which feeds a single Tensor value into the computation.

``` swift
@inlinable @inline(__always) public static func infeedEnqueue<Dtype: TensorFlowScalar>(_ input: Tensor<Dtype>, shape: TensorShape?, layout: [Int32], deviceOrdinal: Int64 = -1)
```

#### Parameters

  - input: - input: A tensor that will be provided using the infeed mechanism.

### `infeedEnqueuePrelinearizedBuffer(_:deviceOrdinal:)`

An op which enqueues prelinearized buffer into TPU infeed.

``` swift
@inlinable @inline(__always) public static func infeedEnqueuePrelinearizedBuffer(_ input: VariantHandle, deviceOrdinal: Int64 = -1)
```

#### Parameters

  - input: - input: A variant tensor representing linearized output.

### `infeedEnqueueTuple(inputs:shapes:layouts:deviceOrdinal:)`

Feeds multiple Tensor values into the computation as an XLA tuple.

``` swift
@inlinable @inline(__always) public static func infeedEnqueueTuple<Dtypes: TensorArrayProtocol>(inputs: Dtypes, shapes: [TensorShape?], layouts: [Int32], deviceOrdinal: Int64 = -1)
```

#### Parameters

  - inputs: - inputs: A list of tensors that will be provided using the infeed mechanism.

### `initializeTableFromTextFileV2(tableHandle:filename:keyIndex:valueIndex:vocabSize:delimiter:)`

Initializes a table from a text file.

``` swift
@inlinable @inline(__always) public static func initializeTableFromTextFileV2(tableHandle: ResourceHandle, filename: StringTensor, keyIndex: Int64, valueIndex: Int64, vocabSize: Int64 = -1, delimiter: String = "\t")
```

It inserts one key-value pair into the table for each line of the file.
The key and value is extracted from the whole line content, elements from the
split line based on `delimiter` or the line number (starting from zero).
Where to extract the key and value from a line is specified by `key_index` and
`value_index`.

#### Parameters

  - filename: - filename: Filename of a vocabulary text file.

### `initializeTableV2(tableHandle:keys:_:)`

Table initializer that takes two tensors for keys and values respectively.

``` swift
@inlinable @inline(__always) public static func initializeTableV2<Tkey: TensorFlowScalar, Tval: TensorFlowScalar>(tableHandle: ResourceHandle, keys: Tensor<Tkey>, _ values: Tensor<Tval>)
```

#### Parameters

  - keys: - keys: Keys of type Tkey.
  - values: - values: Values of type Tval.

### `inplaceAdd(_:i:v:)`

``` swift
@inlinable @inline(__always) public static func inplaceAdd<T: TensorFlowScalar>(_ x: Tensor<T>, i: Tensor<Int32>, v: Tensor<T>) -> Tensor<T>
```

``` 
Adds v into specified rows of x.

Computes y = x; y[i, :] += v; return y.
```

#### Parameters

  - x: - x: A `Tensor` of type T.
  - i: - i: A vector. Indices into the left-most dimension of `x`.
  - v: - v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.

### `inplaceSub(_:i:v:)`

``` swift
@inlinable @inline(__always) public static func inplaceSub<T: TensorFlowScalar>(_ x: Tensor<T>, i: Tensor<Int32>, v: Tensor<T>) -> Tensor<T>
```

``` 
Subtracts `v` into specified rows of `x`.

Computes y = x; y[i, :] -= v; return y.
```

#### Parameters

  - x: - x: A `Tensor` of type T.
  - i: - i: A vector. Indices into the left-most dimension of `x`.
  - v: - v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.

### `inplaceUpdate(_:i:v:)`

``` swift
@inlinable @inline(__always) public static func inplaceUpdate<T: TensorFlowScalar>(_ x: Tensor<T>, i: Tensor<Int32>, v: Tensor<T>) -> Tensor<T>
```

``` 
Updates specified rows with values in `v`.

Computes `x[i, :] = v; return x`.
```

#### Parameters

  - x: - x: A tensor of type `T`.
  - i: - i: A vector. Indices into the left-most dimension of `x`.
  - v: - v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.

### `int64Output()`

``` swift
@inlinable @inline(__always) public static func int64Output() -> Tensor<Int64>
```

### `intAttr(foo:)`

``` swift
@inlinable @inline(__always) public static func intAttr(foo: Int64 = 1) -> Tensor<Int64>
```

### `intInput(_:)`

``` swift
@inlinable @inline(__always) public static func intInput(_ a: Tensor<Int32>)
```

### `intInputFloatInput(_:_:)`

``` swift
@inlinable @inline(__always) public static func intInputFloatInput(_ a: Tensor<Int32>, _ b: Tensor<Float>)
```

### `intInputIntOutput(_:)`

``` swift
@inlinable @inline(__always) public static func intInputIntOutput(_ a: Tensor<Int32>) -> Tensor<Int32>
```

### `intOutput()`

``` swift
@inlinable @inline(__always) public static func intOutput() -> Tensor<Int32>
```

### `intOutputFloatOutput()`

``` swift
@inlinable @inline(__always) public static func intOutputFloatOutput() -> (a: Tensor<Int32>, b: Tensor<Float>)
```

### `interleaveDataset(inputDataset:otherArguments:cycleLength:blockLength:f:outputTypes:outputShapes:)`

Creates a dataset that applies `f` to the outputs of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func interleaveDataset<FIn: TensorGroup, FOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, cycleLength: Tensor<Int64>, blockLength: Tensor<Int64>, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

Unlike MapDataset, the `f` in InterleaveDataset is expected to return
a Dataset variant, and InterleaveDataset will flatten successive
results into a single Dataset. Unlike FlatMapDataset,
InterleaveDataset will interleave sequences of up to `block_length`
consecutive elements from `cycle_length` input elements.

### `inv(_:)`

Computes the reciprocal of x element-wise.

``` swift
@inlinable @inline(__always) public static func inv<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

I.e., \\(y = 1 / x\\).

### `invGrad(_:dy:)`

Computes the gradient for the inverse of `x` wrt its input.

``` swift
@inlinable @inline(__always) public static func invGrad<T: FloatingPoint & TensorFlowScalar>(_ y: Tensor<T>, dy: Tensor<T>) -> Tensor<T>
```

Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
is the corresponding input gradient.

### `invert(_:)`

Invert (flip) each bit of supported types; for example, type `uint8` value 01010101 becomes 10101010.

``` swift
@inlinable @inline(__always) public static func invert<T: TensorFlowInteger>(_ x: Tensor<T>) -> Tensor<T>
```

Flip each bit of supported types.  For example, type `int8` (decimal 2) binary 00000010 becomes (decimal -3) binary 11111101.
This operation is performed on each element of the tensor argument `x`.

Example:

``` python
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
 
# flip 2 (00000010) to -3 (11111101)
tf.assert_equal(-3, bitwise_ops.invert(2))
 
dtype_list = [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
              dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64]
 
inputs = [0, 5, 3, 14]
for dtype in dtype_list:
  # Because of issues with negative numbers, let's test this indirectly.
  # 1. invert(a) and a = 0
  # 2. invert(a) or a = invert(0)
  input_tensor = tf.constant([0, 5, 3, 14], dtype=dtype)
  not_a_and_a, not_a_or_a, not_0 = [bitwise_ops.bitwise_and(
                                      input_tensor, bitwise_ops.invert(input_tensor)),
                                    bitwise_ops.bitwise_or(
                                      input_tensor, bitwise_ops.invert(input_tensor)),
                                    bitwise_ops.invert(
                                      tf.constant(0, dtype=dtype))]
 
  expected = tf.constant([0, 0, 0, 0], dtype=tf.float32)
  tf.assert_equal(tf.cast(not_a_and_a, tf.float32), expected)
 
  expected = tf.cast([not_0] * 4, tf.float32)
  tf.assert_equal(tf.cast(not_a_or_a, tf.float32), expected)
 
  # For unsigned dtypes let's also check the result directly.
  if dtype.is_unsigned:
    inverted = bitwise_ops.invert(input_tensor)
    expected = tf.constant([dtype.max - x for x in inputs], dtype=tf.float32)
    tf.assert_equal(tf.cast(inverted, tf.float32), tf.cast(expected, tf.float32))
```

### `invertPermutation(_:)`

Computes the inverse permutation of a tensor.

``` swift
@inlinable @inline(__always) public static func invertPermutation<T: TensorFlowIndex>(_ x: Tensor<T>) -> Tensor<T>
```

This operation computes the inverse of an index permutation. It takes a 1-D
integer tensor `x`, which represents the indices of a zero-based array, and
swaps each value with its index position. In other words, for an output tensor
`y` and an input tensor `x`, this operation computes the following:

`y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

The values must include 0. There can be no duplicate values or negative values.

For example:

``` 
# tensor `x` is [3, 4, 0, 2, 1]
invert_permutation(x) ==> [2, 4, 3, 0, 1]
```

#### Parameters

  - x: - x: 1-D.

### `isBoostedTreesEnsembleInitialized(treeEnsembleHandle:)`

Checks whether a tree ensemble has been initialized.

``` swift
@inlinable @inline(__always) public static func isBoostedTreesEnsembleInitialized(treeEnsembleHandle: ResourceHandle) -> Tensor<Bool>
```

### `isBoostedTreesQuantileStreamResourceInitialized(quantileStreamResourceHandle:)`

Checks whether a quantile stream has been initialized.

``` swift
@inlinable @inline(__always) public static func isBoostedTreesQuantileStreamResourceInitialized(quantileStreamResourceHandle: ResourceHandle) -> Tensor<Bool>
```

An Op that checks if quantile stream resource is initialized.

### `isFinite(_:)`

Returns which elements of x are finite.

``` swift
@inlinable @inline(__always) public static func isFinite<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<Bool>
```

@compatibility(numpy)
Equivalent to np.isfinite
@end\_compatibility

Example:

``` python
x = tf.constant([5.0, 4.8, 6.8, np.inf, np.nan])
tf.math.is_finite(x) ==> [True, True, True, False, False]
```

### `isInf(_:)`

Returns which elements of x are Inf.

``` swift
@inlinable @inline(__always) public static func isInf<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<Bool>
```

@compatibility(numpy)
Equivalent to np.isinf
@end\_compatibility

Example:

``` python
x = tf.constant([5.0, np.inf, 6.8, np.inf])
tf.math.is_inf(x) ==> [False, True, False, True]
```

### `isNan(_:)`

Returns which elements of x are NaN.

``` swift
@inlinable @inline(__always) public static func isNan<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<Bool>
```

@compatibility(numpy)
Equivalent to np.isnan
@end\_compatibility

Example:

``` python
x = tf.constant([5.0, np.nan, 6.8, np.nan, np.inf])
tf.math.is_nan(x) ==> [False, True, False, True, False]
```

### `iterator(sharedName:container:outputTypes:outputShapes:)`

A container for an iterator resource.

``` swift
@inlinable @inline(__always) public static func iterator(sharedName: String, container: String, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> ResourceHandle
```

### `iteratorFromStringHandle(stringHandle:outputTypes:outputShapes:)`

Converts the given string representing a handle to an iterator to a resource.

``` swift
@inlinable @inline(__always) public static func iteratorFromStringHandle(stringHandle: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> ResourceHandle
```

### `iteratorFromStringHandleV2(stringHandle:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func iteratorFromStringHandleV2(stringHandle: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> ResourceHandle
```

### `iteratorGetDevice(resource:)`

Returns the name of the device on which `resource` has been placed.

``` swift
@inlinable @inline(__always) public static func iteratorGetDevice(resource: ResourceHandle) -> StringTensor
```

### `iteratorGetNext(iterator:outputShapes:)`

Gets the next output from the given iterator .

``` swift
@inlinable @inline(__always) public static func iteratorGetNext<OutputTypes: TensorGroup>(iterator: ResourceHandle, outputShapes: [TensorShape?]) -> OutputTypes
```

### `iteratorGetNextAsOptional(iterator:outputTypes:outputShapes:)`

Gets the next output from the given iterator as an Optional variant.

``` swift
@inlinable @inline(__always) public static func iteratorGetNextAsOptional(iterator: ResourceHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `iteratorGetNextSync(iterator:outputShapes:)`

Gets the next output from the given iterator.

``` swift
@inlinable @inline(__always) public static func iteratorGetNextSync<OutputTypes: TensorGroup>(iterator: ResourceHandle, outputShapes: [TensorShape?]) -> OutputTypes
```

This operation is a synchronous version IteratorGetNext. It should only be used
in situations where the iterator does not block the calling thread, or where
the calling thread is not a member of the thread pool used to execute parallel
operations (e.g. in eager mode).

### `iteratorToStringHandle(resourceHandle:)`

Converts the given `resource_handle` representing an iterator to a string.

``` swift
@inlinable @inline(__always) public static func iteratorToStringHandle(resourceHandle: ResourceHandle) -> StringTensor
```

### `iteratorV2(sharedName:container:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func iteratorV2(sharedName: String, container: String, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> ResourceHandle
```

### `kMC2ChainInitialization(distances:seed:)`

Returns the index of a data point that should be added to the seed set.

``` swift
@inlinable @inline(__always) public static func kMC2ChainInitialization(distances: Tensor<Float>, seed: Tensor<Int64>) -> Tensor<Int64>
```

Entries in distances are assumed to be squared distances of candidate points to
the already sampled centers in the seed set. The op constructs one Markov chain
of the k-MC^2 algorithm and returns the index of one candidate point to be added
as an additional cluster center.

#### Parameters

  - distances: - distances: Vector with squared distances to the closest previously sampled cluster center for each candidate point.
  - seed: - seed: Scalar. Seed for initializing the random number generator.

### `kernelLabel()`

``` swift
@inlinable @inline(__always) public static func kernelLabel() -> StringTensor
```

### `kernelLabelRequired(_:)`

``` swift
@inlinable @inline(__always) public static func kernelLabelRequired(_ input: Tensor<Int32>) -> StringTensor
```

### `kmeansPlusPlusInitialization(points:numToSample:seed:numRetriesPerSample:)`

Selects num\_to\_sample rows of input using the KMeans++ criterion.

``` swift
@inlinable @inline(__always) public static func kmeansPlusPlusInitialization(points: Tensor<Float>, numToSample: Tensor<Int64>, seed: Tensor<Int64>, numRetriesPerSample: Tensor<Int64>) -> Tensor<Float>
```

Rows of points are assumed to be input points. One row is selected at random.
Subsequent rows are sampled with probability proportional to the squared L2
distance from the nearest row selected thus far till num\_to\_sample rows have
been sampled.

#### Parameters

  - points: - points: Matrix of shape (n, d). Rows are assumed to be input points.
  - seed: - seed: Scalar. Seed for initializing the random number generator.

### `l2Loss(t:)`

L2 Loss.

``` swift
@inlinable @inline(__always) public static func l2Loss<T: FloatingPoint & TensorFlowScalar>(t: Tensor<T>) -> Tensor<T>
```

Computes half the L2 norm of a tensor without the `sqrt`:

``` 
output = sum(t ** 2) / 2
```

#### Parameters

  - t: - t: Typically 2-D, but may have any dimensions.

### `lMDBDataset(filenames:outputTypes:outputShapes:)`

Creates a dataset that emits the key-value pairs in one or more LMDB files.

``` swift
@inlinable @inline(__always) public static func lMDBDataset(filenames: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

The Lightning Memory-Mapped Database Manager, or LMDB, is an embedded binary
key-value database. This dataset can read the contents of LMDB database files,
the names of which generally have the `.mdb` suffix.

Each output element consists of a key-value pair represented as a pair of
scalar string `Tensor`s, where the first `Tensor` contains the key and the
second `Tensor` contains the value.

LMDB uses different file formats on big- and little-endian machines.
`LMDBDataset` can only read files in the format of the host machine.

#### Parameters

  - filenames: - filenames: A scalar or a vector containing the name(s) of the binary file(s) to be read.

### `lRN(_:depthRadius:bias:alpha:beta:)`

Local Response Normalization.

``` swift
@inlinable @inline(__always) public static func lRN<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, depthRadius: Int64 = 5, bias: Double = 1, alpha: Double = 1, beta: Double = 0.5) -> Tensor<T>
```

The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
dimension), and each vector is normalized independently.  Within a given vector,
each component is divided by the weighted, squared sum of inputs within
`depth_radius`.  In detail,

``` 
sqr_sum[a, b, c, d] =
    sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
output = input / (bias + alpha * sqr_sum) ** beta
```

For details, see [Krizhevsky et al., ImageNet classification with deep
convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

#### Parameters

  - input: - input: 4-D.

### `lRNGrad(inputGrads:inputImage:outputImage:depthRadius:bias:alpha:beta:)`

Gradients for Local Response Normalization.

``` swift
@inlinable @inline(__always) public static func lRNGrad<T: FloatingPoint & TensorFlowScalar>(inputGrads: Tensor<T>, inputImage: Tensor<T>, outputImage: Tensor<T>, depthRadius: Int64 = 5, bias: Double = 1, alpha: Double = 1, beta: Double = 0.5) -> Tensor<T>
```

### `lSTMBlockCell(_:csPrev:hPrev:w:wci:wcf:wco:_:forgetBias:cellClip:usePeephole:)`

Computes the LSTM cell forward propagation for 1 time step.

``` swift
@inlinable @inline(__always) public static func lSTMBlockCell<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, csPrev: Tensor<T>, hPrev: Tensor<T>, w: Tensor<T>, wci: Tensor<T>, wcf: Tensor<T>, wco: Tensor<T>, _ b: Tensor<T>, forgetBias: Double = 1, cellClip: Double = 3, usePeephole: Bool = false) -> (
    i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>,
    h: Tensor<T>
  )
```

This implementation uses 1 weight matrix and 1 bias vector, and there's an
optional peephole connection.

This kernel op implements the following mathematical equations:

``` python
xh = [x, h_prev]
[i, f, ci, o] = xh * w + b
f = f + forget_bias
 
if not use_peephole:
  wci = wcf = wco = 0
 
i = sigmoid(cs_prev * wci + i)
f = sigmoid(cs_prev * wcf + f)
ci = tanh(ci)
 
cs = ci .* i + cs_prev .* f
cs = clip(cs, cell_clip)
 
o = sigmoid(cs * wco + o)
co = tanh(cs)
h = co .* o
```

#### Parameters

  - x: - x: The input to the LSTM cell, shape (batch\_size, num\_inputs).
  - w: - w: The weight matrix.
  - wci: - wci: The weight matrix for input gate peephole connection.
  - wcf: - wcf: The weight matrix for forget gate peephole connection.
  - wco: - wco: The weight matrix for output gate peephole connection.
  - b: - b: The bias vector.

### `lSTMBlockCellGrad(_:csPrev:hPrev:w:wci:wcf:wco:_:i:cs:f:o:ci:co:csGrad:hGrad:usePeephole:)`

Computes the LSTM cell backward propagation for 1 timestep.

``` swift
@inlinable @inline(__always) public static func lSTMBlockCellGrad<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, csPrev: Tensor<T>, hPrev: Tensor<T>, w: Tensor<T>, wci: Tensor<T>, wcf: Tensor<T>, wco: Tensor<T>, _ b: Tensor<T>, i: Tensor<T>, cs: Tensor<T>, f: Tensor<T>, o: Tensor<T>, ci: Tensor<T>, co: Tensor<T>, csGrad: Tensor<T>, hGrad: Tensor<T>, usePeephole: Bool) -> (
    csPrevGrad: Tensor<T>, dicfo: Tensor<T>, wciGrad: Tensor<T>, wcfGrad: Tensor<T>,
    wcoGrad: Tensor<T>
  )
```

This implementation is to be used in conjunction of LSTMBlockCell.

#### Parameters

  - x: - x: The input to the LSTM cell, shape (batch\_size, num\_inputs).
  - w: - w: The weight matrix.
  - wci: - wci: The weight matrix for input gate peephole connection.
  - wcf: - wcf: The weight matrix for forget gate peephole connection.
  - wco: - wco: The weight matrix for output gate peephole connection.
  - b: - b: The bias vector.
  - i: - i: The input gate.
  - cs: - cs: The cell state before the tanh.
  - f: - f: The forget gate.
  - o: - o: The output gate.
  - ci: - ci: The cell input.
  - co: - co: The cell after the tanh.

### `latencyStatsDataset(inputDataset:tag:outputTypes:outputShapes:)`

Records the latency of producing `input_dataset` elements in a StatsAggregator.

``` swift
@inlinable @inline(__always) public static func latencyStatsDataset(inputDataset: VariantHandle, tag: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `leakyRelu(features:alpha:)`

Computes rectified linear: `max(features, features * alpha)`.

``` swift
@inlinable @inline(__always) public static func leakyRelu<T: FloatingPoint & TensorFlowScalar>(features: Tensor<T>, alpha: Double = 0.2) -> Tensor<T>
```

### `leakyReluGrad(gradients:features:alpha:)`

Computes rectified linear gradients for a LeakyRelu operation.

``` swift
@inlinable @inline(__always) public static func leakyReluGrad<T: FloatingPoint & TensorFlowScalar>(gradients: Tensor<T>, features: Tensor<T>, alpha: Double = 0.2) -> Tensor<T>
```

#### Parameters

  - gradients: - gradients: The backpropagated gradients to the corresponding LeakyRelu operation.
  - features: - features: The features passed as input to the corresponding LeakyRelu operation, OR the outputs of that operation (both work equivalently).

### `learnedUnigramCandidateSampler(trueClasses:numTrue:numSampled:unique:rangeMax:seed:seed2:)`

Generates labels for candidate sampling with a learned unigram distribution.

``` swift
@inlinable @inline(__always) public static func learnedUnigramCandidateSampler(trueClasses: Tensor<Int64>, numTrue: Int64, numSampled: Int64, unique: Bool, rangeMax: Int64, seed: Int64 = 0, seed2: Int64 = 0) -> (
    sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>,
    sampledExpectedCount: Tensor<Float>
  )
```

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

### `leftShift(_:_:)`

Elementwise computes the bitwise left-shift of `x` and `y`.

``` swift
@inlinable @inline(__always) public static func leftShift<T: TensorFlowInteger>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

If `y` is negative, or greater than or equal to the width of `x` in bits the
result is implementation defined.

Example:

``` python
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
import numpy as np
dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64]
 
for dtype in dtype_list:
  lhs = tf.constant([-1, -5, -3, -14], dtype=dtype)
  rhs = tf.constant([5, 0, 7, 11], dtype=dtype)
 
  left_shift_result = bitwise_ops.left_shift(lhs, rhs)
 
  print(left_shift_result)
 
# This will print:
# tf.Tensor([ -32   -5 -128    0], shape=(4,), dtype=int8)
# tf.Tensor([   -32     -5   -384 -28672], shape=(4,), dtype=int16)
# tf.Tensor([   -32     -5   -384 -28672], shape=(4,), dtype=int32)
# tf.Tensor([   -32     -5   -384 -28672], shape=(4,), dtype=int64)
 
lhs = np.array([-2, 64, 101, 32], dtype=np.int8)
rhs = np.array([-1, -5, -3, -14], dtype=np.int8)
bitwise_ops.left_shift(lhs, rhs)
# <tf.Tensor: shape=(4,), dtype=int8, numpy=array([ -2,  64, 101,  32], dtype=int8)>
```

### `less(_:_:)`

Returns the truth value of (x \< y) element-wise.

``` swift
@inlinable @inline(__always) public static func less<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<Bool>
```

*NOTE*: `Less` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Example:

``` python
x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.less(x, y) ==> [False, True, False]
 
x = tf.constant([5, 4, 6])
y = tf.constant([5, 6, 7])
tf.math.less(x, y) ==> [False, True, True]
```

### `lessEqual(_:_:)`

Returns the truth value of (x \<= y) element-wise.

``` swift
@inlinable @inline(__always) public static func lessEqual<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<Bool>
```

*NOTE*: `LessEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Example:

``` python
x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.less_equal(x, y) ==> [True, True, False]
 
x = tf.constant([5, 4, 6])
y = tf.constant([5, 6, 6])
tf.math.less_equal(x, y) ==> [True, True, True]
```

### `lgamma(_:)`

Computes the log of the absolute value of `Gamma(x)` element-wise.

``` swift
@inlinable @inline(__always) public static func lgamma<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

For positive numbers, this function computes log((input - 1)\!) for every element in the tensor.
`lgamma(5) = log((5-1)!) = log(4!) = log(24) = 3.1780539`

Example:

``` python
x = tf.constant([0, 0.5, 1, 4.5, -4, -5.6])
tf.math.lgamma(x) ==> [inf, 0.5723649, 0., 2.4537368, inf, -4.6477685]
```

### `linSpace(start:stop:num:)`

Generates values in an interval.

``` swift
@inlinable @inline(__always) public static func linSpace<T: FloatingPoint & TensorFlowScalar, Tidx: TensorFlowIndex>(start: Tensor<T>, stop: Tensor<T>, num: Tensor<Tidx>) -> Tensor<T>
```

A sequence of `num` evenly-spaced values are generated beginning at `start`.
If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
so that the last one is exactly `stop`.

For example:

``` 
tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
```

#### Parameters

  - start: - start: 0-D tensor. First entry in the range.
  - stop: - stop: 0-D tensor. Last entry in the range.
  - num: - num: 0-D tensor. Number of values to generate.

### `listDiff(_:_:)`

Computes the difference between two lists of numbers or strings.

``` swift
@inlinable @inline(__always) public static func listDiff<T: TensorFlowScalar, OutIdx: TensorFlowIndex>(_ x: Tensor<T>, _ y: Tensor<T>) -> (out: Tensor<T>, idx: Tensor<OutIdx>)
```

Given a list `x` and a list `y`, this operation returns a list `out` that
represents all values that are in `x` but not in `y`. The returned list `out`
is sorted in the same order that the numbers appear in `x` (duplicates are
preserved). This operation also returns a list `idx` that represents the
position of each `out` element in `x`. In other words:

`out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

For example, given this input:

``` 
x = [1, 2, 3, 4, 5, 6]
y = [1, 3, 5]
```

This operation would return:

``` 
out ==> [2, 4, 6]
idx ==> [1, 3, 5]
```

#### Parameters

  - x: - x: 1-D. Values to keep.
  - y: - y: 1-D. Values to remove.

### `listInput(_:)`

``` swift
@inlinable @inline(__always) public static func listInput<T: TensorFlowScalar>(_ a: [Tensor<T>])
```

### `listOutput()`

``` swift
@inlinable @inline(__always) public static func listOutput<T: TensorGroup>() -> T
```

### `loadAndRemapMatrix(ckptPath:oldTensorName:rowRemapping:colRemapping:initializingValues:numRows:numCols:maxRowsInMemory:)`

Loads a 2-D (matrix) `Tensor` with name `old_tensor_name` from the checkpoint

``` swift
@inlinable @inline(__always) public static func loadAndRemapMatrix(ckptPath: StringTensor, oldTensorName: StringTensor, rowRemapping: Tensor<Int64>, colRemapping: Tensor<Int64>, initializingValues: Tensor<Float>, numRows: Int64, numCols: Int64, maxRowsInMemory: Int64 = -1) -> Tensor<Float>
```

at `ckpt_path` and potentially reorders its rows and columns using the
specified remappings.

Most users should use one of the wrapper initializers (such as
`tf.contrib.framework.load_and_remap_matrix_initializer`) instead of this
function directly.

The remappings are 1-D tensors with the following properties:

`(r * num_cols) + (c * num_rows) - (r * c) == len(initializing_values)`

The remapping tensors can be generated using the GenerateVocabRemapping op.

As an example, with row\_remapping = \[1, 0, -1\], col\_remapping = \[0, 2, -1\],
initializing\_values = \[0.5, -0.5, 0.25, -0.25, 42\], and w(i, j) representing
the value from row i, column j of the old tensor in the checkpoint, the output
matrix will look like the following:

\[\[w(1, 0),  w(1, 2),  0.5\],
\[w(0, 0),  w(0, 2), -0.5\],
\[0.25,    -0.25,      42\]\]

### `loadTPUEmbeddingADAMParameters(parameters:momenta:velocities:tableId:tableName:numShards:shardId:config:)`

Load ADAM embedding parameters.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingADAMParameters(parameters: Tensor<Float>, momenta: Tensor<Float>, velocities: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the ADAM optimization algorithm.
  - momenta: - momenta: Value of momenta used in the ADAM optimization algorithm.
  - velocities: - velocities: Value of velocities used in the ADAM optimization algorithm.

### `loadTPUEmbeddingADAMParametersGradAccumDebug(parameters:momenta:velocities:gradientAccumulators:tableId:tableName:numShards:shardId:config:)`

Load ADAM embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingADAMParametersGradAccumDebug(parameters: Tensor<Float>, momenta: Tensor<Float>, velocities: Tensor<Float>, gradientAccumulators: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the ADAM optimization algorithm.
  - momenta: - momenta: Value of momenta used in the ADAM optimization algorithm.
  - velocities: - velocities: Value of velocities used in the ADAM optimization algorithm.

### `loadTPUEmbeddingAdadeltaParameters(parameters:accumulators:updates:tableId:tableName:numShards:shardId:config:)`

Load Adadelta embedding parameters.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingAdadeltaParameters(parameters: Tensor<Float>, accumulators: Tensor<Float>, updates: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the Adadelta optimization algorithm.
  - accumulators: - accumulators: Value of accumulators used in the Adadelta optimization algorithm.
  - updates: - updates: Value of updates used in the Adadelta optimization algorithm.

### `loadTPUEmbeddingAdadeltaParametersGradAccumDebug(parameters:accumulators:updates:gradientAccumulators:tableId:tableName:numShards:shardId:config:)`

Load Adadelta parameters with debug support.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingAdadeltaParametersGradAccumDebug(parameters: Tensor<Float>, accumulators: Tensor<Float>, updates: Tensor<Float>, gradientAccumulators: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the Adadelta optimization algorithm.
  - accumulators: - accumulators: Value of accumulators used in the Adadelta optimization algorithm.
  - updates: - updates: Value of updates used in the Adadelta optimization algorithm.

### `loadTPUEmbeddingAdagradParameters(parameters:accumulators:tableId:tableName:numShards:shardId:config:)`

Load Adagrad embedding parameters.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingAdagradParameters(parameters: Tensor<Float>, accumulators: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the Adagrad optimization algorithm.
  - accumulators: - accumulators: Value of accumulators used in the Adagrad optimization algorithm.

### `loadTPUEmbeddingAdagradParametersGradAccumDebug(parameters:accumulators:gradientAccumulators:tableId:tableName:numShards:shardId:config:)`

Load Adagrad embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingAdagradParametersGradAccumDebug(parameters: Tensor<Float>, accumulators: Tensor<Float>, gradientAccumulators: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the Adagrad optimization algorithm.
  - accumulators: - accumulators: Value of accumulators used in the Adagrad optimization algorithm.

### `loadTPUEmbeddingCenteredRMSPropParameters(parameters:ms:mom:mg:tableId:tableName:numShards:shardId:config:)`

Load centered RMSProp embedding parameters.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingCenteredRMSPropParameters(parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>, mg: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the centered RMSProp optimization algorithm.
  - ms: - ms: Value of ms used in the centered RMSProp optimization algorithm.
  - mom: - mom: Value of mom used in the centered RMSProp optimization algorithm.
  - mg: - mg: Value of mg used in the centered RMSProp optimization algorithm.

### `loadTPUEmbeddingFTRLParameters(parameters:accumulators:linears:tableId:tableName:numShards:shardId:config:)`

Load FTRL embedding parameters.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingFTRLParameters(parameters: Tensor<Float>, accumulators: Tensor<Float>, linears: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the FTRL optimization algorithm.
  - accumulators: - accumulators: Value of accumulators used in the FTRL optimization algorithm.
  - linears: - linears: Value of linears used in the FTRL optimization algorithm.

### `loadTPUEmbeddingFTRLParametersGradAccumDebug(parameters:accumulators:linears:gradientAccumulators:tableId:tableName:numShards:shardId:config:)`

Load FTRL embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingFTRLParametersGradAccumDebug(parameters: Tensor<Float>, accumulators: Tensor<Float>, linears: Tensor<Float>, gradientAccumulators: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the FTRL optimization algorithm.
  - accumulators: - accumulators: Value of accumulators used in the FTRL optimization algorithm.
  - linears: - linears: Value of linears used in the FTRL optimization algorithm.

### `loadTPUEmbeddingMDLAdagradLightParameters(parameters:accumulators:weights:benefits:tableId:tableName:numShards:shardId:config:)`

Load MDL Adagrad Light embedding parameters.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingMDLAdagradLightParameters(parameters: Tensor<Float>, accumulators: Tensor<Float>, weights: Tensor<Float>, benefits: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the MDL Adagrad Light optimization algorithm.
  - accumulators: - accumulators: Value of accumulators used in the MDL Adagrad Light optimization algorithm.
  - weights: - weights: Value of weights used in the MDL Adagrad Light optimization algorithm.
  - benefits: - benefits: Value of benefits used in the MDL Adagrad Light optimization algorithm.

### `loadTPUEmbeddingMomentumParameters(parameters:momenta:tableId:tableName:numShards:shardId:config:)`

Load Momentum embedding parameters.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingMomentumParameters(parameters: Tensor<Float>, momenta: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the Momentum optimization algorithm.
  - momenta: - momenta: Value of momenta used in the Momentum optimization algorithm.

### `loadTPUEmbeddingMomentumParametersGradAccumDebug(parameters:momenta:gradientAccumulators:tableId:tableName:numShards:shardId:config:)`

Load Momentum embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingMomentumParametersGradAccumDebug(parameters: Tensor<Float>, momenta: Tensor<Float>, gradientAccumulators: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the Momentum optimization algorithm.
  - momenta: - momenta: Value of momenta used in the Momentum optimization algorithm.

### `loadTPUEmbeddingProximalAdagradParameters(parameters:accumulators:tableId:tableName:numShards:shardId:config:)`

Load proximal Adagrad embedding parameters.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingProximalAdagradParameters(parameters: Tensor<Float>, accumulators: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the proximal Adagrad optimization algorithm.
  - accumulators: - accumulators: Value of accumulators used in the proximal Adagrad optimization algorithm.

### `loadTPUEmbeddingProximalAdagradParametersGradAccumDebug(parameters:accumulators:gradientAccumulators:tableId:tableName:numShards:shardId:config:)`

Load proximal Adagrad embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingProximalAdagradParametersGradAccumDebug(parameters: Tensor<Float>, accumulators: Tensor<Float>, gradientAccumulators: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the proximal Adagrad optimization algorithm.
  - accumulators: - accumulators: Value of accumulators used in the proximal Adagrad optimization algorithm.

### `loadTPUEmbeddingRMSPropParameters(parameters:ms:mom:tableId:tableName:numShards:shardId:config:)`

Load RMSProp embedding parameters.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingRMSPropParameters(parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the RMSProp optimization algorithm.
  - ms: - ms: Value of ms used in the RMSProp optimization algorithm.
  - mom: - mom: Value of mom used in the RMSProp optimization algorithm.

### `loadTPUEmbeddingRMSPropParametersGradAccumDebug(parameters:ms:mom:gradientAccumulators:tableId:tableName:numShards:shardId:config:)`

Load RMSProp embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingRMSPropParametersGradAccumDebug(parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>, gradientAccumulators: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the RMSProp optimization algorithm.
  - ms: - ms: Value of ms used in the RMSProp optimization algorithm.
  - mom: - mom: Value of mom used in the RMSProp optimization algorithm.

### `loadTPUEmbeddingStochasticGradientDescentParameters(parameters:tableId:tableName:numShards:shardId:config:)`

Load SGD embedding parameters.

``` swift
@inlinable @inline(__always) public static func loadTPUEmbeddingStochasticGradientDescentParameters(parameters: Tensor<Float>, tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String)
```

An op that loads optimization parameters into HBM for embedding. Must be
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to install
parameters that are loaded from a checkpoint before a training loop is
executed.

#### Parameters

  - parameters: - parameters: Value of parameters used in the stochastic gradient descent optimization algorithm.

### `log(_:)`

Computes natural logarithm of x element-wise.

``` swift
@inlinable @inline(__always) public static func log<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

I.e., \\(y = \\log\_e x\\).

Example:

``` python
x = tf.constant([0, 0.5, 1, 5])
tf.math.log(x) ==> [-inf, -0.6931472,  0. ,  1.609438]
```

### `log1p(_:)`

Computes natural logarithm of (1 + x) element-wise.

``` swift
@inlinable @inline(__always) public static func log1p<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

I.e., \\(y = \\log\_e (1 + x)\\).

Example:

``` python
x = tf.constant([0, 0.5, 1, 5])
tf.math.log1p(x) ==> [0., 0.4054651, 0.6931472, 1.7917595]
```

### `logMatrixDeterminant(_:)`

Computes the sign and the log of the absolute value of the determinant of

``` swift
@inlinable @inline(__always) public static func logMatrixDeterminant<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> (sign: Tensor<T>, logAbsDeterminant: Tensor<T>)
```

one or more square matrices.

The input is a tensor of shape `[N, M, M]` whose inner-most 2 dimensions
form square matrices. The outputs are two tensors containing the signs and
absolute values of the log determinants for all N input submatrices
`[..., :, :]` such that the determinant = sign\*exp(log\_abs\_determinant).
The log\_abs\_determinant is computed as det(P)\*sum(log(diag(LU))) where LU
is the LU decomposition of the input and P is the corresponding
permutation matrix.

#### Parameters

  - input: - input: Shape is `[N, M, M]`.

### `logSoftmax(logits:)`

Computes log softmax activations.

``` swift
@inlinable @inline(__always) public static func logSoftmax<T: FloatingPoint & TensorFlowScalar>(logits: Tensor<T>) -> Tensor<T>
```

For each batch `i` and class `j` we have

``` 
logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
```

#### Parameters

  - logits: - logits: 2-D with shape `[batch_size, num_classes]`.

### `logUniformCandidateSampler(trueClasses:numTrue:numSampled:unique:rangeMax:seed:seed2:)`

Generates labels for candidate sampling with a log-uniform distribution.

``` swift
@inlinable @inline(__always) public static func logUniformCandidateSampler(trueClasses: Tensor<Int64>, numTrue: Int64, numSampled: Int64, unique: Bool, rangeMax: Int64, seed: Int64 = 0, seed2: Int64 = 0) -> (
    sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>,
    sampledExpectedCount: Tensor<Float>
  )
```

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

### `logicalAnd(_:_:)`

Returns the truth value of x AND y element-wise.

``` swift
@inlinable @inline(__always) public static func logicalAnd(_ x: Tensor<Bool>, _ y: Tensor<Bool>) -> Tensor<Bool>
```

*NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `logicalNot(_:)`

Returns the truth value of `NOT x` element-wise.

``` swift
@inlinable @inline(__always) public static func logicalNot(_ x: Tensor<Bool>) -> Tensor<Bool>
```

#### Parameters

  - x: - x: A `Tensor` of type `bool`.

### `logicalOr(_:_:)`

Returns the truth value of x OR y element-wise.

``` swift
@inlinable @inline(__always) public static func logicalOr(_ x: Tensor<Bool>, _ y: Tensor<Bool>) -> Tensor<Bool>
```

*NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `lookupTableExportV2(tableHandle:)`

Outputs all keys and values in the table.

``` swift
@inlinable @inline(__always) public static func lookupTableExportV2<Tkeys: TensorFlowScalar, Tvalues: TensorFlowScalar>(tableHandle: ResourceHandle) -> (keys: Tensor<Tkeys>, values: Tensor<Tvalues>)
```

### `lookupTableFindV2(tableHandle:keys:defaultValue:)`

Looks up keys in a table, outputs the corresponding values.

``` swift
@inlinable @inline(__always) public static func lookupTableFindV2<Tin: TensorFlowScalar, Tout: TensorFlowScalar>(tableHandle: ResourceHandle, keys: Tensor<Tin>, defaultValue: Tensor<Tout>) -> Tensor<Tout>
```

The tensor `keys` must of the same type as the keys of the table.
The output `values` is of the type of the table values.

The scalar `default_value` is the value output for keys not present in the
table. It must also be of the same type as the table values.

#### Parameters

  - keys: - keys: Any shape.  Keys to look up.

### `lookupTableImportV2(tableHandle:keys:_:)`

Replaces the contents of the table with the specified keys and values.

``` swift
@inlinable @inline(__always) public static func lookupTableImportV2<Tin: TensorFlowScalar, Tout: TensorFlowScalar>(tableHandle: ResourceHandle, keys: Tensor<Tin>, _ values: Tensor<Tout>)
```

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

#### Parameters

  - keys: - keys: Any shape.  Keys to look up.
  - values: - values: Values to associate with keys.

### `lookupTableInsertV2(tableHandle:keys:_:)`

Updates the table to associates keys with values.

``` swift
@inlinable @inline(__always) public static func lookupTableInsertV2<Tin: TensorFlowScalar, Tout: TensorFlowScalar>(tableHandle: ResourceHandle, keys: Tensor<Tin>, _ values: Tensor<Tout>)
```

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

#### Parameters

  - keys: - keys: Any shape.  Keys to look up.
  - values: - values: Values to associate with keys.

### `lookupTableRemoveV2(tableHandle:keys:)`

Removes keys and its associated values from a table.

``` swift
@inlinable @inline(__always) public static func lookupTableRemoveV2<Tin: TensorFlowScalar>(tableHandle: ResourceHandle, keys: Tensor<Tin>)
```

The tensor `keys` must of the same type as the keys of the table. Keys not
already in the table are silently ignored.

#### Parameters

  - keys: - keys: Any shape.  Keys of the elements to remove.

### `lookupTableSizeV2(tableHandle:)`

Computes the number of elements in the given table.

``` swift
@inlinable @inline(__always) public static func lookupTableSizeV2(tableHandle: ResourceHandle) -> Tensor<Int64>
```

### `loopCond(_:)`

Forwards the input to the output.

``` swift
@inlinable @inline(__always) public static func loopCond(_ input: Tensor<Bool>) -> Tensor<Bool>
```

This operator represents the loop termination condition used by the
"pivot" switches of a loop.

#### Parameters

  - input: - input: A boolean scalar, representing the branch predicate of the Switch op.

### `lowerBound(sortedInputs:_:)`

Applies lower\_bound(sorted\_search\_values, values) along each row.

``` swift
@inlinable @inline(__always) public static func lowerBound<T: TensorFlowScalar, OutType: TensorFlowIndex>(sortedInputs: Tensor<T>, _ values: Tensor<T>) -> Tensor<OutType>
```

Each set of rows with the same index in (sorted\_inputs, values) is treated
independently.  The resulting row is the equivalent of calling
`np.searchsorted(sorted_inputs, values, side='left')`.

The result is not a global index to the entire
`Tensor`, but rather just the index in the last dimension.

A 2-D example:
sorted\_sequence = \[\[0, 3, 9, 9, 10\],
\[1, 2, 3, 4, 5\]\]
values = \[\[2, 4, 9\],
\[0, 2, 6\]\]

result = LowerBound(sorted\_sequence, values)

result == \[\[1, 2, 2\],
\[0, 1, 5\]\]

#### Parameters

  - values: - values: 2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains the values that will be searched for in `sorted_search_values`.

### `lu(_:)`

Computes the LU decomposition of one or more square matrices.

``` swift
@inlinable @inline(__always) public static func lu<T: FloatingPoint & TensorFlowScalar, OutputIdxType: TensorFlowIndex>(_ input: Tensor<T>) -> (lu: Tensor<T>, p: Tensor<OutputIdxType>)
```

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices.

The input has to be invertible.

The output consists of two tensors LU and P containing the LU decomposition
of all input submatrices `[..., :, :]`. LU encodes the lower triangular and
upper triangular factors.

For each input submatrix of shape `[M, M]`, L is a lower triangular matrix of
shape `[M, M]` with unit diagonal whose entries correspond to the strictly lower
triangular part of LU. U is a upper triangular matrix of shape `[M, M]` whose
entries correspond to the upper triangular part, including the diagonal, of LU.

P represents a permutation matrix encoded as a list of indices each between `0`
and `M-1`, inclusive. If P\_mat denotes the permutation matrix corresponding to
P, then the L, U and P satisfies P\_mat \* input = L \* U.

#### Parameters

  - input: - input: A tensor of shape `[..., M, M]` whose inner-most 2 dimensions form matrices of size `[M, M]`.

### `makeIterator(dataset:iterator:)`

Makes a new iterator from the given `dataset` and stores it in `iterator`.

``` swift
@inlinable @inline(__always) public static func makeIterator(dataset: VariantHandle, iterator: ResourceHandle)
```

This operation may be executed multiple times. Each execution will reset the
iterator in `iterator` to the first element of `dataset`.

### `mapAndBatchDataset(inputDataset:otherArguments:batchSize:numParallelCalls:dropRemainder:f:outputTypes:outputShapes:preserveCardinality:)`

Creates a dataset that fuses mapping with batching.

``` swift
@inlinable @inline(__always) public static func mapAndBatchDataset<FIn: TensorGroup, FOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, batchSize: Tensor<Int64>, numParallelCalls: Tensor<Int64>, dropRemainder: Tensor<Bool>, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?], preserveCardinality: Bool = false) -> VariantHandle
```

Creates a dataset that applies `f` to the outputs of `input_dataset` and then
batches `batch_size` of them.

Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
to `batch_size * num_parallel_batches` copies of `f` in parallel.

### `mapClear(capacity:memoryLimit:dtypes:container:sharedName:)`

Op removes all elements in the underlying container.

``` swift
@inlinable @inline(__always) public static func mapClear(capacity: Int64 = 0, memoryLimit: Int64 = 0, dtypes: [TensorDataType], container: String, sharedName: String)
```

### `mapDataset(inputDataset:otherArguments:f:outputTypes:outputShapes:useInterOpParallelism:preserveCardinality:)`

Creates a dataset that applies `f` to the outputs of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func mapDataset<FIn: TensorGroup, FOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?], useInterOpParallelism: Bool = true, preserveCardinality: Bool = false) -> VariantHandle
```

### `mapDefun(arguments:capturedInputs:outputShapes:f:maxIntraOpParallelism:)`

Maps a function on the list of tensors unpacked from arguments on dimension 0.
The function given by `f` is assumed to be stateless, and is executed
concurrently on all the slices; up to batch\_size (i.e. the size of the 0th
dimension of each argument) functions will be scheduled at once.

``` swift
@inlinable @inline(__always) public static func mapDefun<Targuments: TensorArrayProtocol, Tcaptured: TensorArrayProtocol, OutputTypes: TensorGroup, FIn: TensorGroup, FOut: TensorGroup>(arguments: Targuments, capturedInputs: Tcaptured, outputShapes: [TensorShape?], f: (FIn) -> FOut, maxIntraOpParallelism: Int64 = 1) -> OutputTypes
```

The `max_intra_op_parallelism` attr, which defaults to 1, can be used to
limit the intra op parallelism. To limit inter-op parallelism, a user can
set a private threadpool on the dataset using `tf.data.Options`'s
`ThreadingOptions`.

Note that this op is not exposed to users directly, but is invoked in tf.data
rewrites.

#### Parameters

  - arguments: - arguments: A list of tensors whose types are `Targuments`, corresponding to the inputs the function should be mapped over.

### `mapIncompleteSize(capacity:memoryLimit:dtypes:container:sharedName:)`

Op returns the number of incomplete elements in the underlying container.

``` swift
@inlinable @inline(__always) public static func mapIncompleteSize(capacity: Int64 = 0, memoryLimit: Int64 = 0, dtypes: [TensorDataType], container: String, sharedName: String) -> Tensor<Int32>
```

### `mapPeek(key:indices:capacity:memoryLimit:container:sharedName:)`

Op peeks at the values at the specified key.  If the

``` swift
@inlinable @inline(__always) public static func mapPeek<Dtypes: TensorGroup>(key: Tensor<Int64>, indices: Tensor<Int32>, capacity: Int64 = 0, memoryLimit: Int64 = 0, container: String, sharedName: String) -> Dtypes
```

underlying container does not contain this key
this op will block until it does.

### `mapSize(capacity:memoryLimit:dtypes:container:sharedName:)`

Op returns the number of elements in the underlying container.

``` swift
@inlinable @inline(__always) public static func mapSize(capacity: Int64 = 0, memoryLimit: Int64 = 0, dtypes: [TensorDataType], container: String, sharedName: String) -> Tensor<Int32>
```

### `mapStage(key:indices:_:capacity:memoryLimit:dtypes:container:sharedName:)`

Stage (key, values) in the underlying container which behaves like a hashtable.

``` swift
@inlinable @inline(__always) public static func mapStage<FakeDtypes: TensorArrayProtocol>(key: Tensor<Int64>, indices: Tensor<Int32>, _ values: FakeDtypes, capacity: Int64 = 0, memoryLimit: Int64 = 0, dtypes: [TensorDataType], container: String, sharedName: String)
```

#### Parameters

  - key: - key: int64
  - values: - values: a list of tensors dtypes A list of data types that inserted values should adhere to.

### `mapUnstage(key:indices:capacity:memoryLimit:container:sharedName:)`

Op removes and returns the values associated with the key

``` swift
@inlinable @inline(__always) public static func mapUnstage<Dtypes: TensorGroup>(key: Tensor<Int64>, indices: Tensor<Int32>, capacity: Int64 = 0, memoryLimit: Int64 = 0, container: String, sharedName: String) -> Dtypes
```

from the underlying container.   If the underlying container
does not contain this key, the op will block until it does.

### `mapUnstageNoKey(indices:capacity:memoryLimit:container:sharedName:)`

Op removes and returns a random (key, value)

``` swift
@inlinable @inline(__always) public static func mapUnstageNoKey<Dtypes: TensorGroup>(indices: Tensor<Int32>, capacity: Int64 = 0, memoryLimit: Int64 = 0, container: String, sharedName: String) -> (key: Tensor<Int64>, values: Dtypes)
```

from the underlying container.   If the underlying container
does not contain elements, the op will block until it does.

### `matMul(_:_:transposeA:transposeB:)`

Multiply the matrix "a" by the matrix "b".

``` swift
@inlinable @inline(__always) public static func matMul<T: TensorFlowNumeric>(_ a: Tensor<T>, _ b: Tensor<T>, transposeA: Bool = false, transposeB: Bool = false) -> Tensor<T>
```

The inputs must be two-dimensional matrices and the inner dimension of
"a" (after being transposed if transpose\_a is true) must match the
outer dimension of "b" (after being transposed if transposed\_b is
true).

*Note*: The default kernel implementation for MatMul on GPUs uses
cublas.

### `matchingFiles(pattern:)`

Returns the set of files matching one or more glob patterns.

``` swift
@inlinable @inline(__always) public static func matchingFiles(pattern: StringTensor) -> StringTensor
```

Note that this routine only supports wildcard characters in the
basename portion of the pattern, not in the directory portion.
Note also that the order of filenames returned is deterministic.

#### Parameters

  - pattern: - pattern: Shell wildcard pattern(s). Scalar or vector of type string.

### `matchingFilesDataset(patterns:)`

``` swift
@inlinable @inline(__always) public static func matchingFilesDataset(patterns: StringTensor) -> VariantHandle
```

### `matrixBandPart(_:numLower:numUpper:)`

Copy a tensor setting everything outside a central band in each innermost matrix

``` swift
@inlinable @inline(__always) public static func matrixBandPart<T: TensorFlowScalar, Tindex: TensorFlowIndex>(_ input: Tensor<T>, numLower: Tensor<Tindex>, numUpper: Tensor<Tindex>) -> Tensor<T>
```

to zero.

The `band` part is computed as follows:
Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor with the same shape where

`band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

The indicator function

`in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) && (num_upper < 0 || (n-m) <= num_upper)`.

For example:

``` 
# if 'input' is [[ 0,  1,  2, 3]
                 [-1,  0,  1, 2]
                 [-2, -1,  0, 1]
                 [-3, -2, -1, 0]],
 
tf.matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                       [-1,  0,  1, 2]
                                       [ 0, -1,  0, 1]
                                       [ 0,  0, -1, 0]],
 
tf.matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                      [-1,  0,  1, 0]
                                      [-2, -1,  0, 1]
                                      [ 0, -2, -1, 0]]
```

Useful special cases:

``` 
 tf.matrix_band_part(input, 0, -1) ==> Upper triangular part.
 tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
 tf.matrix_band_part(input, 0, 0) ==> Diagonal.
```

#### Parameters

  - input: - input: Rank `k` tensor.

### `matrixDeterminant(_:)`

Computes the determinant of one or more square matrices.

``` swift
@inlinable @inline(__always) public static func matrixDeterminant<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor containing the determinants
for all input submatrices `[..., :, :]`.

#### Parameters

  - input: - input: Shape is `[..., M, M]`.

### `matrixDiag(diagonal:)`

Returns a batched diagonal tensor with a given batched diagonal values.

``` swift
@inlinable @inline(__always) public static func matrixDiag<T: TensorFlowScalar>(diagonal: Tensor<T>) -> Tensor<T>
```

Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:

Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
tensor of rank `k+1` with dimensions \[I, J, K, ..., N, N\]\` where:

`output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.

For example:

``` 
# 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]
 
and diagonal.shape = (2, 4)
 
tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
                                     [0, 2, 0, 0]
                                     [0, 0, 3, 0]
                                     [0, 0, 0, 4]],
                                    [[5, 0, 0, 0]
                                     [0, 6, 0, 0]
                                     [0, 0, 7, 0]
                                     [0, 0, 0, 8]]]
 
which has shape (2, 4, 4)
```

#### Parameters

  - diagonal: - diagonal: Rank `k`, where `k >= 1`.

### `matrixDiagPart(_:)`

Returns the batched diagonal part of a batched tensor.

``` swift
@inlinable @inline(__always) public static func matrixDiagPart<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

This operation returns a tensor with the `diagonal` part
of the batched `input`. The `diagonal` part is computed as follows:

Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]` where:

`diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.

The input must be at least a matrix.

For example:

``` 
# 'input' is [[[1, 0, 0, 0]
               [0, 2, 0, 0]
               [0, 0, 3, 0]
               [0, 0, 0, 4]],
              [[5, 0, 0, 0]
               [0, 6, 0, 0]
               [0, 0, 7, 0]
               [0, 0, 0, 8]]]
 
and input.shape = (2, 4, 4)
 
tf.matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]
 
which has shape (2, 4)
```

#### Parameters

  - input: - input: Rank `k` tensor where `k >= 2`.

### `matrixDiagPartV2(_:k:paddingValue:)`

Returns the batched diagonal part of a batched tensor.

``` swift
@inlinable @inline(__always) public static func matrixDiagPartV2<T: TensorFlowScalar>(_ input: Tensor<T>, k: Tensor<Int32>, paddingValue: Tensor<T>) -> Tensor<T>
```

Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched
`input`.

Assume `input` has `r` dimensions `[I, J, ..., L, M, N]`.
Let `max_diag_len` be the maximum length among all diagonals to be extracted,
`max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
Let `num_diags` be the number of diagonals to extract,
`num_diags = k[1] - k[0] + 1`.

If `num_diags == 1`, the output tensor is of rank `r - 1` with shape
`[I, J, ..., L, max_diag_len]` and values:

``` 
diagonal[i, j, ..., l, n]
  = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
    padding_value                 ; otherwise.
```

where `y = max(-k[1], 0)`, `x = max(k[1], 0)`.

Otherwise, the output tensor has rank `r` with dimensions
`[I, J, ..., L, num_diags, max_diag_len]` with values:

``` 
diagonal[i, j, ..., l, m, n]
  = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
    padding_value                 ; otherwise.
```

where `d = k[1] - m`, `y = max(-d, 0)`, and `x = max(d, 0)`.

The input must be at least a matrix.

For example:

``` 
input = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
                   [5, 6, 7, 8],
                   [9, 8, 7, 6]],
                  [[5, 4, 3, 2],
                   [1, 2, 3, 4],
                   [5, 6, 7, 8]]])
 
# A main diagonal from each batch.
tf.matrix_diag_part(input) ==> [[1, 6, 7],  # Output shape: (2, 3)
                                [5, 2, 7]]
 
# A superdiagonal from each batch.
tf.matrix_diag_part(input, k = 1)
  ==> [[2, 7, 6],  # Output shape: (2, 3)
       [4, 3, 8]]
 
# A tridiagonal band from each batch.
tf.matrix_diag_part(input, k = (-1, 1))
  ==> [[[2, 7, 6],  # Output shape: (2, 3, 3)
        [1, 6, 7],
        [5, 8, 0]],
       [[4, 3, 8],
        [5, 2, 7],
        [1, 6, 0]]]
 
# Padding value = 9
tf.matrix_diag_part(input, k = (1, 3), padding_value = 9)
  ==> [[[4, 9, 9],  # Output shape: (2, 3, 3)
        [3, 8, 9],
        [2, 7, 6]],
       [[2, 9, 9],
        [3, 4, 9],
        [4, 3, 8]]]
```

#### Parameters

  - input: - input: Rank `r` tensor where `r >= 2`.
  - k: - k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main diagonal, and negative value means subdiagonals. `k` can be a single integer (for a single diagonal) or a pair of integers specifying the low and high ends of a matrix band. `k[0]` must not be larger than `k[1]`.

### `matrixDiagV2(diagonal:k:numRows:numCols:paddingValue:)`

Returns a batched diagonal tensor with given batched diagonal values.

``` swift
@inlinable @inline(__always) public static func matrixDiagV2<T: TensorFlowScalar>(diagonal: Tensor<T>, k: Tensor<Int32>, numRows: Tensor<Int32>, numCols: Tensor<Int32>, paddingValue: Tensor<T>) -> Tensor<T>
```

Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
diagonals of a matrix, with everything else padded with `padding`. `num_rows`
and `num_cols` specify the dimension of the innermost matrix of the output. If
both are not specified, the op assumes the innermost matrix is square and infers
its size from `k` and the innermost dimension of `diagonal`. If only one of them
is specified, the op assumes the unspecified value is the smallest possible
based on other criteria.

Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor has
rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only one
diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has rank
`r` with shape `[I, J, ..., L, num_rows, num_cols]`.

The second innermost dimension of `diagonal` has double meaning.
When `k` is scalar or `k[0] == k[1]`, `M` is part of the batch size
\[I, J, ..., M\], and the output tensor is:

``` 
output[i, j, ..., l, m, n]
  = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
    padding_value                             ; otherwise
```

Otherwise, `M` is treated as the number of diagonals for the matrix in the
same batch (`M = k[1]-k[0]+1`), and the output tensor is:

``` 
output[i, j, ..., l, m, n]
  = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
    padding_value                                     ; otherwise
```

where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.

For example:

``` 
# The main diagonal.
diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
                     [5, 6, 7, 8]])
tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
                               [0, 2, 0, 0],
                               [0, 0, 3, 0],
                               [0, 0, 0, 4]],
                              [[5, 0, 0, 0],
                               [0, 6, 0, 0],
                               [0, 0, 7, 0],
                               [0, 0, 0, 8]]]
 
# A superdiagonal (per batch).
diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
                     [4, 5, 6]])
tf.matrix_diag(diagonal, k = 1)
  ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
        [0, 0, 2, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 0]],
       [[0, 4, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 6],
        [0, 0, 0, 0]]]
 
# A band of diagonals.
diagonals = np.array([[[1, 2, 3],  # Input shape: (2, 2, 3)
                       [4, 5, 0]],
                      [[6, 7, 9],
                       [9, 1, 0]]])
tf.matrix_diag(diagonals, k = (-1, 0))
  ==> [[[1, 0, 0],  # Output shape: (2, 3, 3)
        [4, 2, 0],
        [0, 5, 3]],
       [[6, 0, 0],
        [9, 7, 0],
        [0, 1, 9]]]
 
# Rectangular matrix.
diagonal = np.array([1, 2])  # Input shape: (2)
tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
  ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
       [1, 0, 0, 0],
       [0, 2, 0, 0]]
 
# Rectangular matrix with inferred num_cols and padding_value = 9.
tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding_value = 9)
  ==> [[9, 9],  # Output shape: (3, 2)
       [1, 9],
       [9, 2]]
```

#### Parameters

  - diagonal: - diagonal: Rank `r`, where `r >= 1`
  - k: - k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main diagonal, and negative value means subdiagonals. `k` can be a single integer (for a single diagonal) or a pair of integers specifying the low and high ends of a matrix band. `k[0]` must not be larger than `k[1]`.

### `matrixExponential(_:)`

Deprecated, use python implementation tf.linalg.matrix\_exponential.

``` swift
@inlinable @inline(__always) public static func matrixExponential<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

### `matrixInverse(_:adjoint:)`

Computes the inverse of one or more square invertible matrices or their

``` swift
@inlinable @inline(__always) public static func matrixInverse<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, adjoint: Bool = false) -> Tensor<T>
```

adjoints (conjugate transposes).

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the inverse for all input submatrices `[..., :, :]`.

The op uses LU decomposition with partial pivoting to compute the inverses.

If a matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result.

#### Parameters

  - input: - input: Shape is `[..., M, M]`.

### `matrixLogarithm(_:)`

Computes the matrix logarithm of one or more square matrices:

``` swift
@inlinable @inline(__always) public static func matrixLogarithm<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

\\(log(exp(A)) = A\\)

This op is only defined for complex matrices. If A is positive-definite and
real, then casting to a complex matrix, taking the logarithm and casting back
to a real matrix will give the correct result.

This function computes the matrix logarithm using the Schur-Parlett algorithm.
Details of the algorithm can be found in Section 11.6.2 of:
Nicholas J. Higham, Functions of Matrices: Theory and Computation, SIAM 2008.
ISBN 978-0-898716-46-7.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the exponential for all input submatrices `[..., :, :]`.

#### Parameters

  - input: - input: Shape is `[..., M, M]`.

### `matrixSetDiag(_:diagonal:)`

Returns a batched matrix tensor with new batched diagonal values.

``` swift
@inlinable @inline(__always) public static func matrixSetDiag<T: TensorFlowScalar>(_ input: Tensor<T>, diagonal: Tensor<T>) -> Tensor<T>
```

Given `input` and `diagonal`, this operation returns a tensor with the
same shape and values as `input`, except for the main diagonal of the
innermost matrices.  These will be overwritten by the values in `diagonal`.

The output is computed as follows:

Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
`k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:

#### Parameters

  - input: - input: Rank `k+1`, where `k >= 1`.
  - diagonal: - diagonal: Rank `k`, where `k >= 1`.

### `matrixSetDiagV2(_:diagonal:k:)`

Returns a batched matrix tensor with new batched diagonal values.

``` swift
@inlinable @inline(__always) public static func matrixSetDiagV2<T: TensorFlowScalar>(_ input: Tensor<T>, diagonal: Tensor<T>, k: Tensor<Int32>) -> Tensor<T>
```

Given `input` and `diagonal`, this operation returns a tensor with the
same shape and values as `input`, except for the specified diagonals of the
innermost matrices. These will be overwritten by the values in `diagonal`.

`input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or
`k[0] == k[1]`, `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`.
Otherwise, it has `r+1` dimensions `[I, J, ..., L, num_diags, max_diag_len]`.
`num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
`max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
`max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`

The output is a tensor of rank `k+1` with dimensions `[I, J, ..., L, M, N]`.
If `k` is scalar or `k[0] == k[1]`:

``` 
output[i, j, ..., l, m, n]
  = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
    input[i, j, ..., l, m, n]              ; otherwise
```

Otherwise,

``` 
output[i, j, ..., l, m, n]
  = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
    input[i, j, ..., l, m, n]                         ; otherwise
```

where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.

For example:

``` 
# The main diagonal.
input = np.array([[[7, 7, 7, 7],              # Input shape: (2, 3, 4)
                   [7, 7, 7, 7],
                   [7, 7, 7, 7]],
                  [[7, 7, 7, 7],
                   [7, 7, 7, 7],
                   [7, 7, 7, 7]]])
diagonal = np.array([[1, 2, 3],               # Diagonal shape: (2, 3)
                     [4, 5, 6]])
tf.matrix_set_diag(diagonal) ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
                                   [7, 2, 7, 7],
                                   [7, 7, 3, 7]],
                                  [[4, 7, 7, 7],
                                   [7, 5, 7, 7],
                                   [7, 7, 6, 7]]]
 
# A superdiagonal (per batch).
tf.matrix_set_diag(diagonal, k = 1)
  ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
        [7, 7, 2, 7],
        [7, 7, 7, 3]],
       [[7, 4, 7, 7],
        [7, 7, 5, 7],
        [7, 7, 7, 6]]]
 
# A band of diagonals.
diagonals = np.array([[[1, 2, 3],  # Diagonal shape: (2, 2, 3)
                       [4, 5, 0]],
                      [[6, 1, 2],
                       [3, 4, 0]]])
tf.matrix_set_diag(diagonals, k = (-1, 0))
  ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
        [4, 2, 7, 7],
        [0, 5, 3, 7]],
       [[6, 7, 7, 7],
        [3, 1, 7, 7],
        [7, 4, 2, 7]]]
 
```

#### Parameters

  - input: - input: Rank `r+1`, where `r >= 1`.
  - diagonal: - diagonal: Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`. `k >= 1`.
  - k: - k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main diagonal, and negative value means subdiagonals. `k` can be a single integer (for a single diagonal) or a pair of integers specifying the low and high ends of a matrix band. `k[0]` must not be larger than `k[1]`.

### `matrixSolve(matrix:rhs:adjoint:)`

Solves systems of linear equations.

``` swift
@inlinable @inline(__always) public static func matrixSolve<T: FloatingPoint & TensorFlowScalar>(matrix: Tensor<T>, rhs: Tensor<T>, adjoint: Bool = false) -> Tensor<T>
```

`Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `True` then each output matrix satisfies
`adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.

#### Parameters

  - matrix: - matrix: Shape is `[..., M, M]`.
  - rhs: - rhs: Shape is `[..., M, K]`.

### `matrixSolveLs(matrix:rhs:l2Regularizer:fast:)`

Solves one or more linear least-squares problems.

``` swift
@inlinable @inline(__always) public static func matrixSolveLs<T: FloatingPoint & TensorFlowScalar>(matrix: Tensor<T>, rhs: Tensor<T>, l2Regularizer: Tensor<Double>, fast: Bool = true) -> Tensor<T>
```

`matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
form real or complex matrices of size `[M, N]`. `Rhs` is a tensor of the same
type as `matrix` and shape `[..., M, K]`.
The output is a tensor shape `[..., N, K]` where each output matrix solves
each of the equations
`matrix[..., :, :]` \* `output[..., :, :]` = `rhs[..., :, :]`
in the least squares sense.

We use the following notation for (complex) matrix and right-hand sides
in the batch:

`matrix`=\\(A \\in \\mathbb{C}^{m \\times n}\\),
`rhs`=\\(B  \\in \\mathbb{C}^{m \\times k}\\),
`output`=\\(X  \\in \\mathbb{C}^{n \\times k}\\),
`l2_regularizer`=\\(\\lambda \\in \\mathbb{R}\\).

If `fast` is `True`, then the solution is computed by solving the normal
equations using Cholesky decomposition. Specifically, if \\(m \\ge n\\) then
\\(X = (A^H A + \\lambda I)^{-1} A^H B\\), which solves the least-squares
problem \\(X = \\mathrm{argmin}\_{Z \\in \\Re^{n \\times k} } ||A Z - B||\_F^2 + \\lambda ||Z||*F^2\\).
If \\(m \\lt n\\) then `output` is computed as
\\(X = A^H (A A^H + \\lambda I)^{-1} B\\), which (for \\(\\lambda = 0\\)) is the
minimum-norm solution to the under-determined linear system, i.e.
\\(X = \\mathrm{argmin}*{Z \\in \\mathbb{C}^{n \\times k} } ||Z||*F^2 \\),
subject to \\(A Z = B\\). Notice that the fast path is only numerically stable
when \\(A\\) is numerically full rank and has a condition number
\\(\\mathrm{cond}(A) \\lt \\frac{1}{\\sqrt{\\epsilon*{mach} } }\\) or \\(\\lambda\\) is
sufficiently large.

If `fast` is `False` an algorithm based on the numerically robust complete
orthogonal decomposition is used. This computes the minimum-norm
least-squares solution, even when \\(A\\) is rank deficient. This path is
typically 6-7 times slower than the fast path. If `fast` is `False` then
`l2_regularizer` is ignored.

#### Parameters

  - matrix: - matrix: Shape is `[..., M, N]`.
  - rhs: - rhs: Shape is `[..., M, K]`.

### `matrixSquareRoot(_:)`

Computes the matrix square root of one or more square matrices:

``` swift
@inlinable @inline(__always) public static func matrixSquareRoot<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

matmul(sqrtm(A), sqrtm(A)) = A

The input matrix should be invertible. If the input matrix is real, it should
have no eigenvalues which are real and negative (pairs of complex conjugate
eigenvalues are allowed).

The matrix square root is computed by first reducing the matrix to
quasi-triangular form with the real Schur decomposition. The square root
of the quasi-triangular matrix is then computed directly. Details of
the algorithm can be found in: Nicholas J. Higham, "Computing real
square roots of a real matrix", Linear Algebra Appl., 1987.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the matrix square root for all input submatrices `[..., :, :]`.

#### Parameters

  - input: - input: Shape is `[..., M, M]`.

### `matrixTriangularSolve(matrix:rhs:lower:adjoint:)`

Solves systems of linear equations with upper or lower triangular matrices by backsubstitution.

``` swift
@inlinable @inline(__always) public static func matrixTriangularSolve<T: FloatingPoint & TensorFlowScalar>(matrix: Tensor<T>, rhs: Tensor<T>, lower: Bool = true, adjoint: Bool = false) -> Tensor<T>
```

`matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
square matrices. If `lower` is `True` then the strictly upper triangular part
of each inner-most matrix is assumed to be zero and not accessed.
If `lower` is False then the strictly lower triangular part of each inner-most
matrix is assumed to be zero and not accessed.
`rhs` is a tensor of shape `[..., M, K]`.

The output is a tensor of shape `[..., M, K]`. If `adjoint` is
`True` then the innermost matrices in `output` satisfy matrix equations
`matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `False` then the strictly then the  innermost matrices in
`output` satisfy matrix equations
`adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.

Example:

``` python
 
a = tf.constant([[3,  0,  0,  0],
                 [2,  1,  0,  0],
                 [1,  0,  1,  0],
                 [1,  1,  1,  1]], dtype=tf.float32)
 
b = tf.constant([[4],
                 [2],
                 [4],
                 [2]], dtype=tf.float32)
 
x = tf.linalg.triangular_solve(a, b, lower=True)
x
# <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
# array([[ 1.3333334 ],
#        [-0.66666675],
#        [ 2.6666665 ],
#        [-1.3333331 ]], dtype=float32)>
 
# in python3 one can use `a@x`
tf.matmul(a, x)
# <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
# array([[4.       ],
#        [2.       ],
#        [4.       ],
#        [1.9999999]], dtype=float32)>
```

#### Parameters

  - matrix: - matrix: Shape is `[..., M, M]`.
  - rhs: - rhs: Shape is `[..., M, K]`.

### `max(_:reductionIndices:keepDims:)`

Computes the maximum of elements across dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func max<T: TensorFlowNumeric, Tidx: TensorFlowIndex>(_ input: Tensor<T>, reductionIndices: Tensor<Tidx>, keepDims: Bool = false) -> Tensor<T>
```

Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

#### Parameters

  - input: - input: The tensor to reduce.

### `maxIntraOpParallelismDataset(inputDataset:maxIntraOpParallelism:outputTypes:outputShapes:)`

Creates a dataset that overrides the maximum intra-op parallelism.

``` swift
@inlinable @inline(__always) public static func maxIntraOpParallelismDataset(inputDataset: VariantHandle, maxIntraOpParallelism: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `maxPool(_:ksize:strides:padding:dataFormat:)`

Performs max pooling on the input.

``` swift
@inlinable @inline(__always) public static func maxPool<T: TensorFlowNumeric>(_ input: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, dataFormat: DataFormat5 = .nhwc) -> Tensor<T>
```

#### Parameters

  - input: - input: 4-D input to pool over.

### `maxPool3D(_:ksize:strides:padding:dataFormat:)`

Performs 3D max pooling on the input.

``` swift
@inlinable @inline(__always) public static func maxPool3D<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, dataFormat: DataFormat1 = .ndhwc) -> Tensor<T>
```

#### Parameters

  - input: - input: Shape `[batch, depth, rows, cols, channels]` tensor to pool over.

### `maxPool3DGrad(origInput:origOutput:grad:ksize:strides:padding:dataFormat:)`

Computes gradients of max pooling function.

``` swift
@inlinable @inline(__always) public static func maxPool3DGrad<T: FloatingPoint & TensorFlowScalar, Tinput: FloatingPoint & TensorFlowScalar>(origInput: Tensor<Tinput>, origOutput: Tensor<Tinput>, grad: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, dataFormat: DataFormat1 = .ndhwc) -> Tensor<T>
```

#### Parameters

  - grad: - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.

### `maxPool3DGradGrad(origInput:origOutput:grad:ksize:strides:padding:dataFormat:)`

Computes second-order gradients of the maxpooling function.

``` swift
@inlinable @inline(__always) public static func maxPool3DGradGrad<T: TensorFlowNumeric>(origInput: Tensor<T>, origOutput: Tensor<T>, grad: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, dataFormat: DataFormat1 = .ndhwc) -> Tensor<T>
```

#### Parameters

  - grad: - grad: Output backprop of shape `[batch, depth, rows, cols, channels]`.

### `maxPoolGrad(origInput:origOutput:grad:ksize:strides:padding:dataFormat:)`

Computes gradients of the maxpooling function.

``` swift
@inlinable @inline(__always) public static func maxPoolGrad<T: TensorFlowNumeric>(origInput: Tensor<T>, origOutput: Tensor<T>, grad: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, dataFormat: DataFormat = .nhwc) -> Tensor<T>
```

#### Parameters

  - grad: - grad: 4-D.  Gradients w.r.t. the output of `max_pool`.

### `maxPoolGradGrad(origInput:origOutput:grad:ksize:strides:padding:dataFormat:)`

Computes second-order gradients of the maxpooling function.

``` swift
@inlinable @inline(__always) public static func maxPoolGradGrad<T: TensorFlowNumeric>(origInput: Tensor<T>, origOutput: Tensor<T>, grad: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, dataFormat: DataFormat = .nhwc) -> Tensor<T>
```

#### Parameters

  - grad: - grad: 4-D.  Gradients of gradients w.r.t. the input of `max_pool`.

### `maxPoolGradGradV2(origInput:origOutput:grad:ksize:strides:padding:dataFormat:)`

Computes second-order gradients of the maxpooling function.

``` swift
@inlinable @inline(__always) public static func maxPoolGradGradV2<T: TensorFlowNumeric>(origInput: Tensor<T>, origOutput: Tensor<T>, grad: Tensor<T>, ksize: Tensor<Int32>, strides: Tensor<Int32>, padding: Padding, dataFormat: DataFormat = .nhwc) -> Tensor<T>
```

#### Parameters

  - grad: - grad: 4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
  - ksize: - ksize: The size of the window for each dimension of the input tensor.
  - strides: - strides: The stride of the sliding window for each dimension of the input tensor.

### `maxPoolGradGradWithArgmax(_:grad:argmax:ksize:strides:padding:includeBatchInIndex:)`

Computes second-order gradients of the maxpooling function.

``` swift
@inlinable @inline(__always) public static func maxPoolGradGradWithArgmax<Targmax: TensorFlowIndex, T: TensorFlowNumeric>(_ input: Tensor<T>, grad: Tensor<T>, argmax: Tensor<Targmax>, ksize: [Int32], strides: [Int32], padding: Padding, includeBatchInIndex: Bool = false) -> Tensor<T>
```

#### Parameters

  - input: - input: The original input.
  - grad: - grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the input of `max_pool`.
  - argmax: - argmax: The indices of the maximum values chosen for each output of `max_pool`.

### `maxPoolGradV2(origInput:origOutput:grad:ksize:strides:padding:dataFormat:)`

Computes gradients of the maxpooling function.

``` swift
@inlinable @inline(__always) public static func maxPoolGradV2<T: TensorFlowNumeric>(origInput: Tensor<T>, origOutput: Tensor<T>, grad: Tensor<T>, ksize: Tensor<Int32>, strides: Tensor<Int32>, padding: Padding, dataFormat: DataFormat = .nhwc) -> Tensor<T>
```

#### Parameters

  - grad: - grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
  - ksize: - ksize: The size of the window for each dimension of the input tensor.
  - strides: - strides: The stride of the sliding window for each dimension of the input tensor.

### `maxPoolGradWithArgmax(_:grad:argmax:ksize:strides:padding:includeBatchInIndex:)`

Computes gradients of the maxpooling function.

``` swift
@inlinable @inline(__always) public static func maxPoolGradWithArgmax<Targmax: TensorFlowIndex, T: TensorFlowNumeric>(_ input: Tensor<T>, grad: Tensor<T>, argmax: Tensor<Targmax>, ksize: [Int32], strides: [Int32], padding: Padding, includeBatchInIndex: Bool = false) -> Tensor<T>
```

#### Parameters

  - input: - input: The original input.
  - grad: - grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the output of `max_pool`.
  - argmax: - argmax: The indices of the maximum values chosen for each output of `max_pool`.

### `maxPoolV2(_:ksize:strides:padding:dataFormat:)`

Performs max pooling on the input.

``` swift
@inlinable @inline(__always) public static func maxPoolV2<T: TensorFlowNumeric>(_ input: Tensor<T>, ksize: Tensor<Int32>, strides: Tensor<Int32>, padding: Padding, dataFormat: DataFormat5 = .nhwc) -> Tensor<T>
```

#### Parameters

  - input: - input: 4-D input to pool over.
  - ksize: - ksize: The size of the window for each dimension of the input tensor.
  - strides: - strides: The stride of the sliding window for each dimension of the input tensor.

### `maxPoolWithArgmax(_:ksize:strides:padding:includeBatchInIndex:)`

Performs max pooling on the input and outputs both max values and indices.

``` swift
@inlinable @inline(__always) public static func maxPoolWithArgmax<Targmax: TensorFlowIndex, T: TensorFlowNumeric>(_ input: Tensor<T>, ksize: [Int32], strides: [Int32], padding: Padding, includeBatchInIndex: Bool = false) -> (output: Tensor<T>, argmax: Tensor<Targmax>)
```

The indices in `argmax` are flattened, so that a maximum value at position
`[b, y, x, c]` becomes flattened index:
`(y * width + x) * channels + c` if `include_batch_in_index` is False;
`((b * height + y) * width + x) * channels + c` if `include_batch_in_index` is True.

The indices returned are always in `[0, height) x [0, width)` before flattening,
even if padding is involved and the mathematically correct answer is outside
(either negative or too large).  This is a bug, but fixing it is difficult to do
in a safe backwards compatible way, especially due to flattening.

#### Parameters

  - input: - input: 4-D with shape `[batch, height, width, channels]`.  Input to pool over.

### `maximum(_:_:)`

Returns the max of x and y (i.e. x \> y ? x : y) element-wise.

``` swift
@inlinable @inline(__always) public static func maximum<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `Maximum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `mean(_:reductionIndices:keepDims:)`

Computes the mean of elements across dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func mean<T: TensorFlowNumeric, Tidx: TensorFlowIndex>(_ input: Tensor<T>, reductionIndices: Tensor<Tidx>, keepDims: Bool = false) -> Tensor<T>
```

Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

#### Parameters

  - input: - input: The tensor to reduce.

### `merge(inputs:)`

Forwards the value of an available tensor from `inputs` to `output`.

``` swift
@inlinable @inline(__always) public static func merge<T: TensorFlowScalar>(inputs: [Tensor<T>]) -> (output: Tensor<T>, valueIndex: Tensor<Int32>)
```

`Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.

`Merge` forwards the first tensor to become available to `output`, and sets
`value_index` to its index in `inputs`.

#### Parameters

  - inputs: - inputs: The input tensors, exactly one of which will become available.

### `mergeSummary(inputs:)`

Merges summaries.

``` swift
@inlinable @inline(__always) public static func mergeSummary(inputs: [StringTensor]) -> StringTensor
```

This op creates a
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.

When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag.

#### Parameters

  - inputs: - inputs: Can be of any shape.  Each must contain serialized `Summary` protocol buffers.

### `mergeV2Checkpoints(checkpointPrefixes:destinationPrefix:deleteOldDirs:)`

V2 format specific: merges the metadata files of sharded checkpoints.  The

``` swift
@inlinable @inline(__always) public static func mergeV2Checkpoints(checkpointPrefixes: StringTensor, destinationPrefix: StringTensor, deleteOldDirs: Bool = true)
```

result is one logical checkpoint, with one physical metadata file and renamed
data files.

Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.

If delete\_old\_dirs is true, attempts to delete recursively the dirname of each
path in the input checkpoint\_prefixes.  This is useful when those paths are non
user-facing temporary locations.

### `mfcc(spectrogram:sampleRate:upperFrequencyLimit:lowerFrequencyLimit:filterbankChannelCount:dctCoefficientCount:)`

Transforms a spectrogram into a form that's useful for speech recognition.

``` swift
@inlinable @inline(__always) public static func mfcc(spectrogram: Tensor<Float>, sampleRate: Tensor<Int32>, upperFrequencyLimit: Double = 4000, lowerFrequencyLimit: Double = 20, filterbankChannelCount: Int64 = 40, dctCoefficientCount: Int64 = 13) -> Tensor<Float>
```

Mel Frequency Cepstral Coefficients are a way of representing audio data that's
been effective as an input feature for machine learning. They are created by
taking the spectrum of a spectrogram (a 'cepstrum'), and discarding some of the
higher frequencies that are less significant to the human ear. They have a long
history in the speech recognition world, and https://en.wikipedia.org/wiki/Mel-frequency\_cepstrum
is a good resource to learn more.

#### Parameters

  - spectrogram: - spectrogram: Typically produced by the Spectrogram op, with magnitude\_squared set to true.

### `min(_:reductionIndices:keepDims:)`

Computes the minimum of elements across dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func min<T: TensorFlowNumeric, Tidx: TensorFlowIndex>(_ input: Tensor<T>, reductionIndices: Tensor<Tidx>, keepDims: Bool = false) -> Tensor<T>
```

Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

#### Parameters

  - input: - input: The tensor to reduce.

### `minimum(_:_:)`

Returns the min of x and y (i.e. x \< y ? x : y) element-wise.

``` swift
@inlinable @inline(__always) public static func minimum<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `Minimum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `mirrorPad(_:paddings:mode:)`

Pads a tensor with mirrored values.

``` swift
@inlinable @inline(__always) public static func mirrorPad<T: TensorFlowScalar, Tpaddings: TensorFlowIndex>(_ input: Tensor<T>, paddings: Tensor<Tpaddings>, mode: Mode6) -> Tensor<T>
```

This operation pads a `input` with mirrored values according to the `paddings`
you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many values to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many values to add after the contents of `input`
in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
(if false, respectively).

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

``` 
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1]], [2, 2]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
                      [2, 1, 1, 2, 3, 3, 2]
                      [5, 4, 4, 5, 6, 6, 5]
                      [5, 4, 4, 5, 6, 6, 5]]
```

#### Parameters

  - input: - input: The input tensor to be padded.
  - paddings: - paddings: A two-column matrix specifying the padding sizes. The number of rows must be the same as the rank of `input`.

### `mirrorPadGrad(_:paddings:mode:)`

Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.

``` swift
@inlinable @inline(__always) public static func mirrorPadGrad<T: TensorFlowScalar, Tpaddings: TensorFlowIndex>(_ input: Tensor<T>, paddings: Tensor<Tpaddings>, mode: Mode6) -> Tensor<T>
```

This operation folds the padded areas of `input` by `MirrorPad` according to the
`paddings` you specify. `paddings` must be the same as `paddings` argument
given to the corresponding `MirrorPad` op.

The folded size of each dimension D of the output is:

`input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`

For example:

``` 
# 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
# 'paddings' is [[0, 1]], [0, 1]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[ 1,  5]
                      [11, 28]]
```

#### Parameters

  - input: - input: The input tensor to be folded.
  - paddings: - paddings: A two-column matrix specifying the padding sizes. The number of rows must be the same as the rank of `input`.

### `mixedStruct(nA:)`

``` swift
@inlinable @inline(__always) public static func mixedStruct(nA: Int64) -> (a: [Tensor<Int32>], b: Tensor<Float>)
```

### `mlirPassthroughOp(inputs:mlirModule:)`

Wraps an arbitrary MLIR computation expressed as a module with a main() function.

``` swift
@inlinable @inline(__always) public static func mlirPassthroughOp<Tinputs: TensorArrayProtocol, Toutputs: TensorGroup>(inputs: Tinputs, mlirModule: String) -> Toutputs
```

This operation does not have an associated kernel and is not intended to be
executed in a regular TensorFlow session. Instead it is intended to be used for
testing or for special case where a user intends to pass custom MLIR computation
through a TensorFlow graph with the intent of having custom tooling processing
it downstream (when targeting a different environment, like TensorFlow lite for
example).
The MLIR module is expected to have a main() function that will be used as an
entry point. The inputs to the operations will be passed as argument to the
main() function and the returned values of the main function mapped to the
outputs.
Example usage:

``` 
import tensorflow as tf
from tensorflow.compiler.mlir.tensorflow.gen_mlir_passthrough_op import mlir_passthrough_op
 
mlir_module = '''python
func @main(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10x10xf32> {
   %add = "magic.op"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
   return %ret : tensor<10x10xf32>
}
'''
 
@tf.function
def foo(x, y):
  return = mlir_passthrough_op([x, y], mlir_module, Toutputs=[tf.float32])
 
graph_def = foo.get_concrete_function(tf.TensorSpec([10], tf.float32), tf.TensorSpec([10], tf.float32)).graph.as_graph_def()
```

### `mod(_:_:)`

Returns element-wise remainder of division. This emulates C semantics in that

``` swift
@inlinable @inline(__always) public static func mod<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

the result here is consistent with a truncating divide. E.g.
`tf.truncatediv(x, y) * y + truncate_mod(x, y) = x`.

*NOTE*: `Mod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `modelDataset(inputDataset:algorithm:cpuBudget:outputTypes:outputShapes:)`

Identity transformation that models performance.

``` swift
@inlinable @inline(__always) public static func modelDataset(inputDataset: VariantHandle, algorithm: Int64 = 0, cpuBudget: Int64 = 0, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

Identity transformation that models performance.

### `mul(_:_:)`

Returns x \* y element-wise.

``` swift
@inlinable @inline(__always) public static func mul<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `Multiply` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `mulNoNan(_:_:)`

Returns x \* y element-wise. Returns zero if y is zero, even if x if infinite or NaN.

``` swift
@inlinable @inline(__always) public static func mulNoNan<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `MulNoNan` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `multiDeviceIterator(devices:sharedName:container:outputTypes:outputShapes:)`

Creates a MultiDeviceIterator resource.

``` swift
@inlinable @inline(__always) public static func multiDeviceIterator(devices: [String], sharedName: String, container: String, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> ResourceHandle
```

### `multiDeviceIteratorFromStringHandle(stringHandle:outputTypes:outputShapes:)`

Generates a MultiDeviceIterator resource from its provided string handle.

``` swift
@inlinable @inline(__always) public static func multiDeviceIteratorFromStringHandle(stringHandle: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> ResourceHandle
```

### `multiDeviceIteratorGetNextFromShard(multiDeviceIterator:shardNum:incarnationId:outputShapes:)`

Gets next element for the provided shard number.

``` swift
@inlinable @inline(__always) public static func multiDeviceIteratorGetNextFromShard<OutputTypes: TensorGroup>(multiDeviceIterator: ResourceHandle, shardNum: Tensor<Int32>, incarnationId: Tensor<Int64>, outputShapes: [TensorShape?]) -> OutputTypes
```

### `multiDeviceIteratorInit(dataset:multiDeviceIterator:maxBufferSize:)`

Initializes the multi device iterator with the given dataset.

``` swift
@inlinable @inline(__always) public static func multiDeviceIteratorInit(dataset: VariantHandle, multiDeviceIterator: ResourceHandle, maxBufferSize: Tensor<Int64>) -> Tensor<Int64>
```

#### Parameters

  - dataset: - dataset: Dataset to be iterated upon.

### `multiDeviceIteratorToStringHandle(multiDeviceIterator:)`

Produces a string handle for the given MultiDeviceIterator.

``` swift
@inlinable @inline(__always) public static func multiDeviceIteratorToStringHandle(multiDeviceIterator: ResourceHandle) -> StringTensor
```

### `multinomial(logits:numSamples:seed:seed2:)`

Draws samples from a multinomial distribution.

``` swift
@inlinable @inline(__always) public static func multinomial<T: TensorFlowNumeric, OutputDtype: TensorFlowIndex>(logits: Tensor<T>, numSamples: Tensor<Int32>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<OutputDtype>
```

#### Parameters

  - logits: - logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]` represents the unnormalized log probabilities for all classes.

### `mutableDenseHashTableV2(emptyKey:deletedKey:container:sharedName:useNodeNameSharing:valueDtype:valueShape:initialNumBuckets:maxLoadFactor:)`

Creates an empty hash table that uses tensors as the backing store.

``` swift
@inlinable @inline(__always) public static func mutableDenseHashTableV2<KeyDtype: TensorFlowScalar>(emptyKey: Tensor<KeyDtype>, deletedKey: Tensor<KeyDtype>, container: String, sharedName: String, useNodeNameSharing: Bool = false, valueDtype: TensorDataType, valueShape: TensorShape?, initialNumBuckets: Int64 = 131072, maxLoadFactor: Double = 0.8) -> ResourceHandle
```

It uses "open addressing" with quadratic reprobing to resolve
collisions.

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

### `mutableHashTableOfTensorsV2(container:sharedName:useNodeNameSharing:keyDtype:valueDtype:valueShape:)`

Creates an empty hash table.

``` swift
@inlinable @inline(__always) public static func mutableHashTableOfTensorsV2(container: String, sharedName: String, useNodeNameSharing: Bool = false, keyDtype: TensorDataType, valueDtype: TensorDataType, valueShape: TensorShape?) -> ResourceHandle
```

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a vector. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

### `mutableHashTableV2(container:sharedName:useNodeNameSharing:keyDtype:valueDtype:)`

Creates an empty hash table.

``` swift
@inlinable @inline(__always) public static func mutableHashTableV2(container: String, sharedName: String, useNodeNameSharing: Bool = false, keyDtype: TensorDataType, valueDtype: TensorDataType) -> ResourceHandle
```

This op creates a mutable hash table, specifying the type of its keys and
values. Each value must be a scalar. Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

### `mutexLock(mutex:)`

Locks a mutex resource.  The output is the lock.  So long as the lock tensor

``` swift
@inlinable @inline(__always) public static func mutexLock(mutex: ResourceHandle) -> VariantHandle
```

is alive, any other request to use `MutexLock` with this mutex will wait.

This is particularly useful for creating a critical section when used in
conjunction with `MutexLockIdentity`:

``` python
 
mutex = mutex_v2(
  shared_name=handle_name, container=container, name=name)
 
def execute_in_critical_section(fn, *args, **kwargs):
  lock = gen_resource_variable_ops.mutex_lock(mutex)
 
  with ops.control_dependencies([lock]):
    r = fn(*args, **kwargs)
 
  with ops.control_dependencies(nest.flatten(r)):
    with ops.colocate_with(mutex):
      ensure_lock_exists = mutex_lock_identity(lock)
 
    # Make sure that if any element of r is accessed, all of
    # them are executed together.
    r = nest.map_structure(tf.identity, r)
 
  with ops.control_dependencies([ensure_lock_exists]):
    return nest.map_structure(tf.identity, r)
```

While `fn` is running in the critical section, no other functions which wish to
use this critical section may run.

Often the use case is that two executions of the same graph, in parallel,
wish to run `fn`; and we wish to ensure that only one of them executes
at a time.  This is especially important if `fn` modifies one or more
variables at a time.

It is also useful if two separate functions must share a resource, but we
wish to ensure the usage is exclusive.

#### Parameters

  - mutex: - mutex: The mutex resource to lock.

### `mutexV2(container:sharedName:)`

Creates a Mutex resource that can be locked by `MutexLock`.

``` swift
@inlinable @inline(__always) public static func mutexV2(container: String, sharedName: String) -> ResourceHandle
```

### `nInPolymorphicTwice(_:_:)`

``` swift
@inlinable @inline(__always) public static func nInPolymorphicTwice<T: TensorFlowScalar>(_ a: [Tensor<T>], _ b: [Tensor<T>])
```

### `nInTwice(_:_:)`

``` swift
@inlinable @inline(__always) public static func nInTwice(_ a: [Tensor<Int32>], _ b: [StringTensor])
```

### `nInTwoTypeVariables(_:_:)`

``` swift
@inlinable @inline(__always) public static func nInTwoTypeVariables<S: TensorFlowScalar, T: TensorFlowScalar>(_ a: [Tensor<S>], _ b: [Tensor<T>])
```

### `nIntsIn(_:)`

``` swift
@inlinable @inline(__always) public static func nIntsIn(_ a: [Tensor<Int32>])
```

### `nIntsOut(n:)`

``` swift
@inlinable @inline(__always) public static func nIntsOut(n: Int64) -> [Tensor<Int32>]
```

### `nIntsOutDefault(n:)`

``` swift
@inlinable @inline(__always) public static func nIntsOutDefault(n: Int64 = 3) -> [Tensor<Int32>]
```

### `nPolymorphicIn(_:)`

``` swift
@inlinable @inline(__always) public static func nPolymorphicIn<T: TensorFlowScalar>(_ a: [Tensor<T>])
```

### `nPolymorphicOut(n:)`

``` swift
@inlinable @inline(__always) public static func nPolymorphicOut<T: TensorFlowScalar>(n: Int64) -> [Tensor<T>]
```

### `nPolymorphicOutDefault(n:)`

``` swift
@inlinable @inline(__always) public static func nPolymorphicOutDefault<T: TensorFlowScalar>(n: Int64 = 2) -> [Tensor<T>]
```

### `nPolymorphicRestrictIn(_:)`

``` swift
@inlinable @inline(__always) public static func nPolymorphicRestrictIn<T: TensorFlowScalar>(_ a: [Tensor<T>])
```

### `nPolymorphicRestrictIn(_:)`

``` swift
@inlinable @inline(__always) public static func nPolymorphicRestrictIn(_ a: [StringTensor])
```

### `nPolymorphicRestrictOut(n:)`

``` swift
@inlinable @inline(__always) public static func nPolymorphicRestrictOut<T: TensorFlowScalar>(n: Int64) -> [Tensor<T>]
```

### `nPolymorphicRestrictOut(n:)`

``` swift
@inlinable @inline(__always) public static func nPolymorphicRestrictOut(n: Int64) -> [StringTensor]
```

### `namespaceTestStringOutput(_:)`

``` swift
@inlinable @inline(__always) public static func namespaceTestStringOutput(_ input: Tensor<Float>) -> (output1: Tensor<Float>, output2: StringTensor)
```

### `ncclAllReduce(_:reduction:numDevices:sharedName:)`

Outputs a tensor containing the reduction across all input tensors.

``` swift
@inlinable @inline(__always) public static func ncclAllReduce<T: TensorFlowNumeric>(_ input: Tensor<T>, reduction: Reduction, numDevices: Int64, sharedName: String) -> Tensor<T>
```

Outputs a tensor containing the reduction across all input tensors passed to ops
within the same \`shared\_name.

The graph should be constructed so if one op runs with shared\_name value `c`,
then `num_devices` ops will run with shared\_name value `c`.  Failure to do so
will cause the graph execution to fail to complete.

input: the input to the reduction
data: the value of the reduction across all `num_devices` devices.
reduction: the reduction operation to perform.
num\_devices: The number of devices participating in this reduction.
shared\_name: Identifier that shared between ops of the same reduction.

### `ncclBroadcast(_:shape:)`

Sends `input` to all devices that are connected to the output.

``` swift
@inlinable @inline(__always) public static func ncclBroadcast<T: TensorFlowNumeric>(_ input: Tensor<T>, shape: TensorShape?) -> Tensor<T>
```

Sends `input` to all devices that are connected to the output.

The graph should be constructed so that all ops connected to the output have a
valid device assignment, and the op itself is assigned one of these devices.

input: The input to the broadcast.
output: The same as input.
shape: The shape of the input tensor.

### `ncclReduce(_:reduction:)`

Reduces `input` from `num_devices` using `reduction` to a single device.

``` swift
@inlinable @inline(__always) public static func ncclReduce<T: TensorFlowNumeric>(_ input: [Tensor<T>], reduction: Reduction) -> Tensor<T>
```

Reduces `input` from `num_devices` using `reduction` to a single device.

The graph should be constructed so that all inputs have a valid device
assignment, and the op itself is assigned one of these devices.

input: The input to the reduction.
data: the value of the reduction across all `num_devices` devices.
reduction: the reduction operation to perform.

### `ndtri(_:)`

``` swift
@inlinable @inline(__always) public static func ndtri<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

### `nearestNeighbors(points:centers:k:)`

Selects the k nearest centers for each point.

``` swift
@inlinable @inline(__always) public static func nearestNeighbors(points: Tensor<Float>, centers: Tensor<Float>, k: Tensor<Int64>) -> (nearestCenterIndices: Tensor<Int64>, nearestCenterDistances: Tensor<Float>)
```

Rows of points are assumed to be input points. Rows of centers are assumed to be
the list of candidate centers. For each point, the k centers that have least L2
distance to it are computed.

#### Parameters

  - points: - points: Matrix of shape (n, d). Rows are assumed to be input points.
  - centers: - centers: Matrix of shape (m, d). Rows are assumed to be centers.
  - k: - k: Number of nearest centers to return for each point. If k is larger than m, then only m centers are returned.

### `neg(_:)`

Computes numerical negative value element-wise.

``` swift
@inlinable @inline(__always) public static func neg<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

I.e., \\(y = -x\\).

### `nextAfter(x1:x2:)`

Returns the next representable value of `x1` in the direction of `x2`, element-wise.

``` swift
@inlinable @inline(__always) public static func nextAfter<T: FloatingPoint & TensorFlowScalar>(x1: Tensor<T>, x2: Tensor<T>) -> Tensor<T>
```

This operation returns the same result as the C++ std::nextafter function.

It can also return a subnormal number.

@compatibility(cpp)
Equivalent to C++ std::nextafter function.
@end\_compatibility

### `nextIteration(data:)`

Makes its input available to the next iteration.

``` swift
@inlinable @inline(__always) public static func nextIteration<T: TensorFlowScalar>(data: Tensor<T>) -> Tensor<T>
```

#### Parameters

  - data: - data: The tensor to be made available to the next iteration.

### `noOp()`

Does nothing. Only useful as a placeholder for control edges.

``` swift
@inlinable @inline(__always) public static func noOp()
```

### `nonDeterministicInts(shape:)`

Non-deterministically generates some integers.

``` swift
@inlinable @inline(__always) public static func nonDeterministicInts<Dtype: TensorFlowScalar, ShapeDtype: TensorFlowScalar>(shape: Tensor<ShapeDtype>) -> Tensor<Dtype>
```

This op may use some OS-provided source of non-determinism (e.g. an RNG), so each execution will give different results.

#### Parameters

  - shape: - shape: The shape of the output tensor.

### `nonMaxSuppression(boxes:scores:maxOutputSize:iouThreshold:)`

Greedily selects a subset of bounding boxes in descending order of score,

``` swift
@inlinable @inline(__always) public static func nonMaxSuppression(boxes: Tensor<Float>, scores: Tensor<Float>, maxOutputSize: Tensor<Int32>, iouThreshold: Double = 0.5) -> Tensor<Int32>
```

pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes are supplied as
\[y1, x1, y2, x2\], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval \[0, 1\]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system.  Note that this
algorithm is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
selected\_indices = tf.image.non\_max\_suppression(
boxes, scores, max\_output\_size, iou\_threshold)
selected\_boxes = tf.gather(boxes, selected\_indices)

#### Parameters

  - boxes: - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
  - scores: - scores: A 1-D float tensor of shape `[num_boxes]` representing a single score corresponding to each box (each row of boxes).

### `nonMaxSuppressionV2(boxes:scores:maxOutputSize:iouThreshold:)`

Greedily selects a subset of bounding boxes in descending order of score,

``` swift
@inlinable @inline(__always) public static func nonMaxSuppressionV2<T: FloatingPoint & TensorFlowScalar, TThreshold: FloatingPoint & TensorFlowScalar>(boxes: Tensor<T>, scores: Tensor<T>, maxOutputSize: Tensor<Int32>, iouThreshold: Tensor<TThreshold>) -> Tensor<Int32>
```

pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes are supplied as
\[y1, x1, y2, x2\], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval \[0, 1\]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system.  Note that this
algorithm is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.

The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:

selected\_indices = tf.image.non\_max\_suppression\_v2(
boxes, scores, max\_output\_size, iou\_threshold)
selected\_boxes = tf.gather(boxes, selected\_indices)

#### Parameters

  - boxes: - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
  - scores: - scores: A 1-D float tensor of shape `[num_boxes]` representing a single score corresponding to each box (each row of boxes).

### `nonMaxSuppressionV3(boxes:scores:maxOutputSize:iouThreshold:scoreThreshold:)`

Greedily selects a subset of bounding boxes in descending order of score,

``` swift
@inlinable @inline(__always) public static func nonMaxSuppressionV3<T: FloatingPoint & TensorFlowScalar, TThreshold: FloatingPoint & TensorFlowScalar>(boxes: Tensor<T>, scores: Tensor<T>, maxOutputSize: Tensor<Int32>, iouThreshold: Tensor<TThreshold>, scoreThreshold: Tensor<TThreshold>) -> Tensor<Int32>
```

pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes with score less than
`score_threshold` are removed.  Bounding boxes are supplied as
\[y1, x1, y2, x2\], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval \[0, 1\]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system and more
generally is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
selected\_indices = tf.image.non\_max\_suppression\_v2(
boxes, scores, max\_output\_size, iou\_threshold, score\_threshold)
selected\_boxes = tf.gather(boxes, selected\_indices)

#### Parameters

  - boxes: - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
  - scores: - scores: A 1-D float tensor of shape `[num_boxes]` representing a single score corresponding to each box (each row of boxes).

### `nonMaxSuppressionV4(boxes:scores:maxOutputSize:iouThreshold:scoreThreshold:padToMaxOutputSize:)`

Greedily selects a subset of bounding boxes in descending order of score,

``` swift
@inlinable @inline(__always) public static func nonMaxSuppressionV4<T: FloatingPoint & TensorFlowScalar, TThreshold: FloatingPoint & TensorFlowScalar>(boxes: Tensor<T>, scores: Tensor<T>, maxOutputSize: Tensor<Int32>, iouThreshold: Tensor<TThreshold>, scoreThreshold: Tensor<TThreshold>, padToMaxOutputSize: Bool = false) -> (selectedIndices: Tensor<Int32>, validOutputs: Tensor<Int32>)
```

pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes with score less than
`score_threshold` are removed.  Bounding boxes are supplied as
\[y1, x1, y2, x2\], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval \[0, 1\]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system and more
generally is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
selected\_indices = tf.image.non\_max\_suppression\_v2(
boxes, scores, max\_output\_size, iou\_threshold, score\_threshold)
selected\_boxes = tf.gather(boxes, selected\_indices)

#### Parameters

  - boxes: - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
  - scores: - scores: A 1-D float tensor of shape `[num_boxes]` representing a single score corresponding to each box (each row of boxes).

### `nonMaxSuppressionV5(boxes:scores:maxOutputSize:iouThreshold:scoreThreshold:softNmsSigma:padToMaxOutputSize:)`

Greedily selects a subset of bounding boxes in descending order of score,

``` swift
@inlinable @inline(__always) public static func nonMaxSuppressionV5<T: FloatingPoint & TensorFlowScalar>(boxes: Tensor<T>, scores: Tensor<T>, maxOutputSize: Tensor<Int32>, iouThreshold: Tensor<T>, scoreThreshold: Tensor<T>, softNmsSigma: Tensor<T>, padToMaxOutputSize: Bool = false) -> (selectedIndices: Tensor<Int32>, selectedScores: Tensor<T>, validOutputs: Tensor<Int32>)
```

pruning away boxes that have high intersection-over-union (IOU) overlap
with previously selected boxes.  Bounding boxes with score less than
`score_threshold` are removed.  Bounding boxes are supplied as
\[y1, x1, y2, x2\], where (y1, x1) and (y2, x2) are the coordinates of any
diagonal pair of box corners and the coordinates can be provided as normalized
(i.e., lying in the interval \[0, 1\]) or absolute.  Note that this algorithm
is agnostic to where the origin is in the coordinate system and more
generally is invariant to orthogonal transformations and translations
of the coordinate system; thus translating or reflections of the coordinate
system result in the same boxes being selected by the algorithm.
The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:
selected\_indices = tf.image.non\_max\_suppression\_v2(
boxes, scores, max\_output\_size, iou\_threshold, score\_threshold)
selected\_boxes = tf.gather(boxes, selected\_indices)
This op also supports a Soft-NMS (with Gaussian weighting) mode (c.f.
Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
of other overlapping boxes instead of directly causing them to be pruned.
To enable this Soft-NMS mode, set the `soft_nms_sigma` parameter to be
larger than 0.

#### Parameters

  - boxes: - boxes: A 2-D float tensor of shape `[num_boxes, 4]`.
  - scores: - scores: A 1-D float tensor of shape `[num_boxes]` representing a single score corresponding to each box (each row of boxes).

### `nonMaxSuppressionWithOverlaps(overlaps:scores:maxOutputSize:overlapThreshold:scoreThreshold:)`

Greedily selects a subset of bounding boxes in descending order of score,

``` swift
@inlinable @inline(__always) public static func nonMaxSuppressionWithOverlaps(overlaps: Tensor<Float>, scores: Tensor<Float>, maxOutputSize: Tensor<Int32>, overlapThreshold: Tensor<Float>, scoreThreshold: Tensor<Float>) -> Tensor<Int32>
```

pruning away boxes that have high overlaps
with previously selected boxes.  Bounding boxes with score less than
`score_threshold` are removed. N-by-n overlap values are supplied as square matrix,
which allows for defining a custom overlap criterium (eg. intersection over union,
intersection over area, etc.).

The output of this operation is a set of integers indexing into the input
collection of bounding boxes representing the selected boxes.  The bounding
box coordinates corresponding to the selected indices can then be obtained
using the `tf.gather operation`.  For example:

selected\_indices = tf.image.non\_max\_suppression\_with\_overlaps(
overlaps, scores, max\_output\_size, overlap\_threshold, score\_threshold)
selected\_boxes = tf.gather(boxes, selected\_indices)

#### Parameters

  - overlaps: - overlaps: A 2-D float tensor of shape `[num_boxes, num_boxes]` representing the n-by-n box overlap values.
  - scores: - scores: A 1-D float tensor of shape `[num_boxes]` representing a single score corresponding to each box (each row of boxes).

### `nonSerializableDataset(inputDataset:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func nonSerializableDataset(inputDataset: VariantHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `none()`

``` swift
@inlinable @inline(__always) public static func none()
```

### `notEqual(_:_:incompatibleShapeError:)`

Returns the truth value of (x \!= y) element-wise.

``` swift
@inlinable @inline(__always) public static func notEqual<T: TensorFlowScalar>(_ x: Tensor<T>, _ y: Tensor<T>, incompatibleShapeError: Bool = true) -> Tensor<Bool>
```

*NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `notEqual(_:_:incompatibleShapeError:)`

Returns the truth value of (x \!= y) element-wise.

``` swift
@inlinable @inline(__always) public static func notEqual(_ x: StringTensor, _ y: StringTensor, incompatibleShapeError: Bool = true) -> Tensor<Bool>
```

*NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `nthElement(_:n:reverse:)`

Finds values of the `n`-th order statistic for the last dimension.

``` swift
@inlinable @inline(__always) public static func nthElement<T: TensorFlowNumeric>(_ input: Tensor<T>, n: Tensor<Int32>, reverse: Bool = false) -> Tensor<T>
```

If the input is a vector (rank-1), finds the entries which is the nth-smallest
value in the vector and outputs their values as scalar tensor.

For matrices (resp. higher rank input), computes the entries which is the
nth-smallest value in each row (resp. vector along the last dimension). Thus,

``` 
values.shape = input.shape[:-1]
```

#### Parameters

  - input: - input: 1-D or higher with last dimension at least `n+1`.
  - n: - n: 0-D. Position of sorted vector to select along the last dimension (along each row for matrices). Valid range of n is `[0, input.shape[:-1])`

### `old()`

``` swift
@inlinable @inline(__always) public static func old()
```

### `oneHot(indices:depth:onValue:offValue:axis:)`

Returns a one-hot tensor.

``` swift
@inlinable @inline(__always) public static func oneHot<T: TensorFlowScalar, Ti: TensorFlowInteger>(indices: Tensor<Ti>, depth: Tensor<Int32>, onValue: Tensor<T>, offValue: Tensor<T>, axis: Int64 = -1) -> Tensor<T>
```

The locations represented by indices in `indices` take value `on_value`,
while all other locations take value `off_value`.

If the input `indices` is rank `N`, the output will have rank `N+1`,
The new axis is created at dimension `axis` (default: the new axis is
appended at the end).

If `indices` is a scalar the output shape will be a vector of length `depth`.

If `indices` is a vector of length `features`, the output shape will be:

``` 
  features x depth if axis == -1
  depth x features if axis == 0
```

If `indices` is a matrix (batch) with shape `[batch, features]`,
the output shape will be:

``` 
  batch x features x depth if axis == -1
  batch x depth x features if axis == 1
  depth x batch x features if axis == 0
```

### Examples

Suppose that

``` 
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 5.0
  off_value = 0.0
  axis = -1
```

Then output is `[4 x 3]`:

``` 
output =
  [5.0 0.0 0.0]  // one_hot(0)
  [0.0 0.0 5.0]  // one_hot(2)
  [0.0 0.0 0.0]  // one_hot(-1)
  [0.0 5.0 0.0]  // one_hot(1)
```

Suppose that

``` 
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 0.0
  off_value = 3.0
  axis = 0
```

Then output is `[3 x 4]`:

``` 
output =
  [0.0 3.0 3.0 3.0]
  [3.0 3.0 3.0 0.0]
  [3.0 3.0 3.0 3.0]
  [3.0 0.0 3.0 3.0]
//  ^                one_hot(0)
//      ^            one_hot(2)
//          ^        one_hot(-1)
//              ^    one_hot(1)
```

Suppose that

``` 
  indices = [[0, 2], [1, -1]]
  depth = 3
  on_value = 1.0
  off_value = 0.0
  axis = -1
```

Then output is `[2 x 2 x 3]`:

``` 
output =
  [
    [1.0, 0.0, 0.0]  // one_hot(0)
    [0.0, 0.0, 1.0]  // one_hot(2)
  ][
    [0.0, 1.0, 0.0]  // one_hot(1)
    [0.0, 0.0, 0.0]  // one_hot(-1)
  ]
```

#### Parameters

  - indices: - indices: A tensor of indices.
  - depth: - depth: A scalar defining the depth of the one hot dimension.

### `oneShotIterator(datasetFactory:outputTypes:outputShapes:container:sharedName:)`

Makes a "one-shot" iterator that can be iterated only once.

``` swift
@inlinable @inline(__always) public static func oneShotIterator<DatasetfactoryIn: TensorGroup, DatasetfactoryOut: TensorGroup>(datasetFactory: (DatasetfactoryIn) -> DatasetfactoryOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?], container: String, sharedName: String) -> ResourceHandle
```

A one-shot iterator bundles the logic for defining the dataset and
the state of the iterator in a single op, which allows simple input
pipelines to be defined without an additional initialization
("MakeIterator") step.

One-shot iterators have the following limitations:

For greater flexibility, use "Iterator" and "MakeIterator" to define
an iterator using an arbitrary subgraph, which may capture tensors
(including fed values) as parameters, and which may be reset multiple
times by rerunning "MakeIterator".

### `onesLike(_:)`

Returns a tensor of ones with the same shape and type as x.

``` swift
@inlinable @inline(__always) public static func onesLike<T: TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

#### Parameters

  - x: - x: a tensor of type T.

### `opWithDefaultAttr(defaultFloat:)`

``` swift
@inlinable @inline(__always) public static func opWithDefaultAttr(defaultFloat: Double = 123) -> Tensor<Int32>
```

### `opWithFutureDefaultAttr()`

``` swift
@inlinable @inline(__always) public static func opWithFutureDefaultAttr()
```

### `optimizeDataset(inputDataset:optimizations:outputTypes:outputShapes:optimizationConfigs:)`

Creates a dataset by applying optimizations to `input_dataset`.

``` swift
@inlinable @inline(__always) public static func optimizeDataset(inputDataset: VariantHandle, optimizations: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?], optimizationConfigs: [String]) -> VariantHandle
```

Creates a dataset by applying optimizations to `input_dataset`.

#### Parameters

  - optimizations: - optimizations: A `tf.string` vector `tf.Tensor` identifying optimizations to use.

### `optionalFromValue(components:)`

Constructs an Optional variant from a tuple of tensors.

``` swift
@inlinable @inline(__always) public static func optionalFromValue<ToutputTypes: TensorArrayProtocol>(components: ToutputTypes) -> VariantHandle
```

### `optionalGetValue(optional:outputShapes:)`

Returns the value stored in an Optional variant or raises an error if none exists.

``` swift
@inlinable @inline(__always) public static func optionalGetValue<OutputTypes: TensorGroup>(optional: VariantHandle, outputShapes: [TensorShape?]) -> OutputTypes
```

### `optionalHasValue(optional:)`

Returns true if and only if the given Optional variant has a value.

``` swift
@inlinable @inline(__always) public static func optionalHasValue(optional: VariantHandle) -> Tensor<Bool>
```

### `optionalNone()`

Creates an Optional variant with no value.

``` swift
@inlinable @inline(__always) public static func optionalNone() -> VariantHandle
```

### `orderedMapClear(capacity:memoryLimit:dtypes:container:sharedName:)`

Op removes all elements in the underlying container.

``` swift
@inlinable @inline(__always) public static func orderedMapClear(capacity: Int64 = 0, memoryLimit: Int64 = 0, dtypes: [TensorDataType], container: String, sharedName: String)
```

### `orderedMapIncompleteSize(capacity:memoryLimit:dtypes:container:sharedName:)`

Op returns the number of incomplete elements in the underlying container.

``` swift
@inlinable @inline(__always) public static func orderedMapIncompleteSize(capacity: Int64 = 0, memoryLimit: Int64 = 0, dtypes: [TensorDataType], container: String, sharedName: String) -> Tensor<Int32>
```

### `orderedMapPeek(key:indices:capacity:memoryLimit:container:sharedName:)`

Op peeks at the values at the specified key.  If the

``` swift
@inlinable @inline(__always) public static func orderedMapPeek<Dtypes: TensorGroup>(key: Tensor<Int64>, indices: Tensor<Int32>, capacity: Int64 = 0, memoryLimit: Int64 = 0, container: String, sharedName: String) -> Dtypes
```

underlying container does not contain this key
this op will block until it does.   This Op is optimized for
performance.

### `orderedMapSize(capacity:memoryLimit:dtypes:container:sharedName:)`

Op returns the number of elements in the underlying container.

``` swift
@inlinable @inline(__always) public static func orderedMapSize(capacity: Int64 = 0, memoryLimit: Int64 = 0, dtypes: [TensorDataType], container: String, sharedName: String) -> Tensor<Int32>
```

### `orderedMapStage(key:indices:_:capacity:memoryLimit:dtypes:container:sharedName:)`

Stage (key, values) in the underlying container which behaves like a ordered

``` swift
@inlinable @inline(__always) public static func orderedMapStage<FakeDtypes: TensorArrayProtocol>(key: Tensor<Int64>, indices: Tensor<Int32>, _ values: FakeDtypes, capacity: Int64 = 0, memoryLimit: Int64 = 0, dtypes: [TensorDataType], container: String, sharedName: String)
```

associative container.   Elements are ordered by key.

#### Parameters

  - key: - key: int64
  - values: - values: a list of tensors dtypes A list of data types that inserted values should adhere to.

### `orderedMapUnstage(key:indices:capacity:memoryLimit:container:sharedName:)`

Op removes and returns the values associated with the key

``` swift
@inlinable @inline(__always) public static func orderedMapUnstage<Dtypes: TensorGroup>(key: Tensor<Int64>, indices: Tensor<Int32>, capacity: Int64 = 0, memoryLimit: Int64 = 0, container: String, sharedName: String) -> Dtypes
```

from the underlying container.   If the underlying container
does not contain this key, the op will block until it does.

### `orderedMapUnstageNoKey(indices:capacity:memoryLimit:container:sharedName:)`

Op removes and returns the (key, value) element with the smallest

``` swift
@inlinable @inline(__always) public static func orderedMapUnstageNoKey<Dtypes: TensorGroup>(indices: Tensor<Int32>, capacity: Int64 = 0, memoryLimit: Int64 = 0, container: String, sharedName: String) -> (key: Tensor<Int64>, values: Dtypes)
```

key from the underlying container.   If the underlying container
does not contain elements, the op will block until it does.

### `outT()`

``` swift
@inlinable @inline(__always) public static func outT<T: TensorFlowScalar>() -> Tensor<T>
```

### `outTypeList()`

``` swift
@inlinable @inline(__always) public static func outTypeList<T: TensorGroup>() -> T
```

### `outTypeListRestrict()`

``` swift
@inlinable @inline(__always) public static func outTypeListRestrict<T: TensorGroup>() -> T
```

### `outfeedDequeue(shape:deviceOrdinal:)`

Retrieves a single tensor from the computation outfeed.

``` swift
@inlinable @inline(__always) public static func outfeedDequeue<Dtype: TensorFlowScalar>(shape: TensorShape?, deviceOrdinal: Int64 = -1) -> Tensor<Dtype>
```

This operation will block indefinitely until data is available.

### `outfeedDequeueTuple(shapes:deviceOrdinal:)`

Retrieve multiple values from the computation outfeed.

``` swift
@inlinable @inline(__always) public static func outfeedDequeueTuple<Dtypes: TensorGroup>(shapes: [TensorShape?], deviceOrdinal: Int64 = -1) -> Dtypes
```

This operation will block indefinitely until data is available. Output `i`
corresponds to XLA tuple element `i`.

### `outfeedEnqueue(_:)`

Enqueue a Tensor on the computation outfeed.

``` swift
@inlinable @inline(__always) public static func outfeedEnqueue<Dtype: TensorFlowScalar>(_ input: Tensor<Dtype>)
```

#### Parameters

  - input: - input: A tensor that will be inserted into the outfeed queue.

### `outfeedEnqueueTuple(inputs:)`

Enqueue multiple Tensor values on the computation outfeed.

``` swift
@inlinable @inline(__always) public static func outfeedEnqueueTuple<Dtypes: TensorArrayProtocol>(inputs: Dtypes)
```

#### Parameters

  - inputs: - inputs: A list of tensors that will be inserted into the outfeed queue as an XLA tuple.

### `pack(_:axis:)`

Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.

``` swift
@inlinable @inline(__always) public static func pack<T: TensorFlowScalar>(_ values: [Tensor<T>], axis: Int64 = 0) -> Tensor<T>
```

Packs the `N` tensors in `values` into a tensor with rank one higher than each
tensor in `values`, by packing them along the `axis` dimension.
Given a list of tensors of shape `(A, B, C)`;

if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
Etc.

For example:

``` 
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
```

This is the opposite of `unpack`.

#### Parameters

  - values: - values: Must be of same shape and type.

### `pad(_:paddings:)`

Pads a tensor with zeros.

``` swift
@inlinable @inline(__always) public static func pad<T: TensorFlowScalar, Tpaddings: TensorFlowIndex>(_ input: Tensor<T>, paddings: Tensor<Tpaddings>) -> Tensor<T>
```

This operation pads a `input` with zeros according to the `paddings` you
specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many zeros to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many zeros to add after the contents of `input`
in that dimension.

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

``` 
# 't' is [[1, 1], [2, 2]]
# 'paddings' is [[1, 1], [2, 2]]
# rank of 't' is 2
pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                      [0, 0, 1, 1, 0, 0]
                      [0, 0, 2, 2, 0, 0]
                      [0, 0, 0, 0, 0, 0]]
```

### `padV2(_:paddings:constantValues:)`

Pads a tensor.

``` swift
@inlinable @inline(__always) public static func padV2<T: TensorFlowScalar, Tpaddings: TensorFlowIndex>(_ input: Tensor<T>, paddings: Tensor<Tpaddings>, constantValues: Tensor<T>) -> Tensor<T>
```

This operation pads `input` according to the `paddings` and `constant_values`
you specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many padding values to add before the contents of `input` in that dimension,
and `paddings[D, 1]` indicates how many padding values to add after the contents
of `input` in that dimension. `constant_values` is a scalar tensor of the same
type as `input` that indicates the value to use for padding `input`.

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

``` 
# 't' is [[1, 1], [2, 2]]
# 'paddings' is [[1, 1], [2, 2]]
# 'constant_values' is 0
# rank of 't' is 2
pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                      [0, 0, 1, 1, 0, 0]
                      [0, 0, 2, 2, 0, 0]
                      [0, 0, 0, 0, 0, 0]]
```

### `paddedBatchDataset(inputDataset:batchSize:paddedShapes:paddingValues:outputShapes:)`

Creates a dataset that batches and pads `batch_size` elements from the input.

``` swift
@inlinable @inline(__always) public static func paddedBatchDataset<ToutputTypes: TensorArrayProtocol>(inputDataset: VariantHandle, batchSize: Tensor<Int64>, paddedShapes: [Tensor<Int64>], paddingValues: ToutputTypes, outputShapes: [TensorShape?]) -> VariantHandle
```

### `paddedBatchDatasetV2(inputDataset:batchSize:paddedShapes:paddingValues:dropRemainder:parallelCopy:outputShapes:)`

Creates a dataset that batches and pads `batch_size` elements from the input.

``` swift
@inlinable @inline(__always) public static func paddedBatchDatasetV2<ToutputTypes: TensorArrayProtocol>(inputDataset: VariantHandle, batchSize: Tensor<Int64>, paddedShapes: [Tensor<Int64>], paddingValues: ToutputTypes, dropRemainder: Tensor<Bool>, parallelCopy: Bool = false, outputShapes: [TensorShape?]) -> VariantHandle
```

### `paddingFIFOQueueV2(componentTypes:shapes:capacity:container:sharedName:)`

A queue that produces elements in first-in first-out order.

``` swift
@inlinable @inline(__always) public static func paddingFIFOQueueV2(componentTypes: [TensorDataType], shapes: [TensorShape?], capacity: Int64 = -1, container: String, sharedName: String) -> ResourceHandle
```

Variable-size shapes are allowed by setting the corresponding shape dimensions
to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
size of any given element in the minibatch.  See below for details.

### `parallelConcat(_:shape:)`

Concatenates a list of `N` tensors along the first dimension.

``` swift
@inlinable @inline(__always) public static func parallelConcat<T: TensorFlowScalar>(_ values: [Tensor<T>], shape: TensorShape?) -> Tensor<T>
```

The input tensors are all required to have size 1 in the first dimension.

For example:

``` 
# 'x' is [[1, 4]]
# 'y' is [[2, 5]]
# 'z' is [[3, 6]]
parallel_concat([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
```

The difference between concat and parallel\_concat is that concat requires all
of the inputs be computed before the operation will begin but doesn't require
that the input shapes be known during graph construction.  Parallel concat
will copy pieces of the input into the output as they become available, in
some situations this can provide a performance benefit.

#### Parameters

  - values: - values: Tensors to be concatenated. All must have size 1 in the first dimension and same shape.

### `parallelDynamicStitch(indices:data:)`

Interleave the values from the `data` tensors into a single tensor.

``` swift
@inlinable @inline(__always) public static func parallelDynamicStitch<T: TensorFlowScalar>(indices: [Tensor<Int32>], data: [Tensor<T>]) -> Tensor<T>
```

Builds a merged tensor such that

``` python
    merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
```

For example, if each `indices[m]` is scalar or vector, we have

``` python
    # Scalar indices:
    merged[indices[m], ...] = data[m][...]
 
    # Vector indices:
    merged[indices[m][i], ...] = data[m][i, ...]
```

Each `data[i].shape` must start with the corresponding `indices[i].shape`,
and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
must have `data[i].shape = indices[i].shape + constant`.  In terms of this
`constant`, the output shape is

``` 
merged.shape = [max(indices)] + constant
```

Values may be merged in parallel, so if an index appears in both `indices[m][i]`
and `indices[n][j]`, the result may be invalid. This differs from the normal
DynamicStitch operator that defines the behavior in that case.

For example:

``` python
    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]
```

This method can be used to merge partitions created by `dynamic_partition`
as illustrated on the following example:

``` python
    # Apply function (increments x_i) on elements for which a certain condition
    # apply (x_i != -1 in this example).
    x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
    condition_mask=tf.not_equal(x,tf.constant(-1.))
    partitioned_data = tf.dynamic_partition(
        x, tf.cast(condition_mask, tf.int32) , 2)
    partitioned_data[1] = partitioned_data[1] + 1.0
    condition_indices = tf.dynamic_partition(
        tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
    x = tf.dynamic_stitch(condition_indices, partitioned_data)
    # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
    # unchanged.
```

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
</div>

### `parallelInterleaveDataset(inputDataset:otherArguments:cycleLength:blockLength:sloppy:bufferOutputElements:prefetchInputElements:f:outputTypes:outputShapes:)`

Creates a dataset that applies `f` to the outputs of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func parallelInterleaveDataset<FIn: TensorGroup, FOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, cycleLength: Tensor<Int64>, blockLength: Tensor<Int64>, sloppy: Tensor<Bool>, bufferOutputElements: Tensor<Int64>, prefetchInputElements: Tensor<Int64>, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

The resulting dataset is similar to the `InterleaveDataset`, with the exception
that if retrieving the next value from a dataset would cause the requester to
block, it will skip that input dataset. This dataset is especially useful
when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
allows the training step to proceed so long as some data is available.

\!\! WARNING \!\! If the `sloppy` parameter is set to `True`, the operation of this
dataset will not be deterministic\!

This dataset has been superseded by `ParallelInterleaveDatasetV2`.  New code
should use `ParallelInterleaveDatasetV2`.

The Python API `tf.data.experimental.parallel_interleave` creates instances of
this op. `tf.data.experimental.parallel_interleave` is a deprecated API.

#### Parameters

  - sloppy: - sloppy: If `True`, return elements as they become available, even if that means returning these elements in a non-deterministic order. Sloppy operation may result in better performance in the presence of stragglers, but the dataset will still block if all of its open streams are blocked. If `False`, always return elements in a deterministic order.

### `parallelInterleaveDatasetV2(inputDataset:otherArguments:cycleLength:blockLength:numParallelCalls:f:outputTypes:outputShapes:sloppy:)`

Creates a dataset that applies `f` to the outputs of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func parallelInterleaveDatasetV2<FIn: TensorGroup, FOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, cycleLength: Tensor<Int64>, blockLength: Tensor<Int64>, numParallelCalls: Tensor<Int64>, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?], sloppy: Bool = false) -> VariantHandle
```

The resulting dataset is similar to the `InterleaveDataset`, except that the
dataset will fetch records from the interleaved datasets in parallel.

The `tf.data` Python API creates instances of this op from
`Dataset.interleave()` when the `num_parallel_calls` parameter of that method
is set to any value other than `None`.

By default, the output of this dataset will be deterministic, which may result
in the dataset blocking if the next data item to be returned isn't available.
In order to avoid head-of-line blocking, one can set the
`experimental_deterministic` parameter of `tf.data.Options` to `False`,
which can improve performance at the expense of non-determinism.

### `parallelMapDataset(inputDataset:otherArguments:numParallelCalls:f:outputTypes:outputShapes:useInterOpParallelism:sloppy:preserveCardinality:)`

Creates a dataset that applies `f` to the outputs of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func parallelMapDataset<FIn: TensorGroup, FOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, numParallelCalls: Tensor<Int32>, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?], useInterOpParallelism: Bool = true, sloppy: Bool = false, preserveCardinality: Bool = false) -> VariantHandle
```

Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
to `num_parallel_calls` copies of `f` in parallel.

### `parameterizedTruncatedNormal(shape:means:stdevs:minvals:maxvals:seed:seed2:)`

Outputs random values from a normal distribution. The parameters may each be a

``` swift
@inlinable @inline(__always) public static func parameterizedTruncatedNormal<Dtype: FloatingPoint & TensorFlowScalar, T: TensorFlowIndex>(shape: Tensor<T>, means: Tensor<Dtype>, stdevs: Tensor<Dtype>, minvals: Tensor<Dtype>, maxvals: Tensor<Dtype>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<Dtype>
```

scalar which applies to the entire output, or a vector of length shape\[0\] which
stores the parameters for each batch.

#### Parameters

  - shape: - shape: The shape of the output tensor. Batches are indexed by the 0th dimension.
  - means: - means: The mean parameter of each batch.
  - stdevs: - stdevs: The standard deviation parameter of each batch. Must be greater than 0.
  - minvals: - minvals: The minimum cutoff. May be -infinity.
  - maxvals: - maxvals: The maximum cutoff. May be +infinity, and must be more than the minval for each batch.

### `parseExample(serialized:names:sparseKeys:denseKeys:denseDefaults:denseShapes:)`

Transforms a vector of brain.Example protos (as strings) into typed tensors.

``` swift
@inlinable @inline(__always) public static func parseExample<SparseTypes: TensorGroup, Tdense: TensorArrayProtocol>(serialized: StringTensor, names: StringTensor, sparseKeys: [StringTensor], denseKeys: [StringTensor], denseDefaults: Tdense, denseShapes: [TensorShape?]) -> (
    sparseIndices: [Tensor<Int64>], sparseValues: SparseTypes, sparseShapes: [Tensor<Int64>],
    denseValues: Tdense
  )
```

#### Parameters

  - serialized: - serialized: A vector containing a batch of binary serialized Example protos.
  - names: - names: A vector containing the names of the serialized protos. May contain, for example, table key (descriptive) names for the corresponding serialized protos.  These are purely useful for debugging purposes, and the presence of values here has no effect on the output. May also be an empty vector if no names are available. If non-empty, this vector must be the same length as "serialized".

### `parseExampleDataset(inputDataset:numParallelCalls:denseDefaults:sparseKeys:denseKeys:sparseTypes:denseShapes:outputTypes:outputShapes:sloppy:raggedKeys:raggedValueTypes:raggedSplitTypes:)`

Transforms `input_dataset` containing `Example` protos as vectors of DT\_STRING into a dataset of `Tensor` or `SparseTensor` objects representing the parsed features.

``` swift
@inlinable @inline(__always) public static func parseExampleDataset<Tdense: TensorArrayProtocol>(inputDataset: VariantHandle, numParallelCalls: Tensor<Int64>, denseDefaults: Tdense, sparseKeys: [String], denseKeys: [String], sparseTypes: [TensorDataType], denseShapes: [TensorShape?], outputTypes: [TensorDataType], outputShapes: [TensorShape?], sloppy: Bool = false, raggedKeys: [String], raggedValueTypes: [TensorDataType], raggedSplitTypes: [TensorDataType]) -> VariantHandle
```

### `parseExampleV2(serialized:names:sparseKeys:denseKeys:raggedKeys:denseDefaults:numSparse:denseShapes:)`

Transforms a vector of tf.Example protos (as strings) into typed tensors.

``` swift
@inlinable @inline(__always) public static func parseExampleV2<Tdense: TensorArrayProtocol, SparseTypes: TensorGroup, RaggedValueTypes: TensorGroup, RaggedSplitTypes: TensorGroup>(serialized: StringTensor, names: StringTensor, sparseKeys: StringTensor, denseKeys: StringTensor, raggedKeys: StringTensor, denseDefaults: Tdense, numSparse: Int64, denseShapes: [TensorShape?]) -> (
    sparseIndices: [Tensor<Int64>], sparseValues: SparseTypes, sparseShapes: [Tensor<Int64>],
    denseValues: Tdense, raggedValues: RaggedValueTypes, raggedRowSplits: RaggedSplitTypes
  )
```

#### Parameters

  - serialized: - serialized: A scalar or vector containing binary serialized Example protos.
  - names: - names: A tensor containing the names of the serialized protos. Corresponds 1:1 with the `serialized` tensor. May contain, for example, table key (descriptive) names for the corresponding serialized protos.  These are purely useful for debugging purposes, and the presence of values here has no effect on the output. May also be an empty vector if no names are available. If non-empty, this tensor must have the same shape as "serialized".

### `parseSequenceExample(serialized:debugName:contextDenseDefaults:featureListDenseMissingAssumedEmpty:contextSparseKeys:contextDenseKeys:featureListSparseKeys:featureListDenseKeys:ncontextSparse:ncontextDense:nfeatureListSparse:nfeatureListDense:contextDenseShapes:featureListDenseShapes:)`

Transforms a vector of brain.SequenceExample protos (as strings) into typed tensors.

``` swift
@inlinable @inline(__always) public static func parseSequenceExample<ContextSparseTypes: TensorGroup, TcontextDense: TensorArrayProtocol, FeatureListDenseTypes: TensorGroup, FeatureListSparseTypes: TensorGroup>(serialized: StringTensor, debugName: StringTensor, contextDenseDefaults: TcontextDense, featureListDenseMissingAssumedEmpty: [String], contextSparseKeys: [String], contextDenseKeys: [String], featureListSparseKeys: [String], featureListDenseKeys: [String], ncontextSparse: Int64 = 0, ncontextDense: Int64 = 0, nfeatureListSparse: Int64 = 0, nfeatureListDense: Int64 = 0, contextDenseShapes: [TensorShape?], featureListDenseShapes: [TensorShape?]) -> (
    contextSparseIndices: [Tensor<Int64>], contextSparseValues: ContextSparseTypes,
    contextSparseShapes: [Tensor<Int64>], contextDenseValues: TcontextDense,
    featureListSparseIndices: [Tensor<Int64>], featureListSparseValues: FeatureListSparseTypes,
    featureListSparseShapes: [Tensor<Int64>], featureListDenseValues: FeatureListDenseTypes,
    featureListDenseLengths: [Tensor<Int64>]
  )
```

#### Parameters

  - serialized: - serialized: A vector containing binary serialized SequenceExample protos.

### `parseSequenceExampleV2(serialized:debugName:contextSparseKeys:contextDenseKeys:contextRaggedKeys:featureListSparseKeys:featureListDenseKeys:featureListRaggedKeys:featureListDenseMissingAssumedEmpty:contextDenseDefaults:ncontextSparse:contextDenseShapes:nfeatureListSparse:nfeatureListDense:featureListDenseShapes:)`

Transforms a vector of tf.io.SequenceExample protos (as strings) into
typed tensors.

``` swift
@inlinable @inline(__always) public static func parseSequenceExampleV2<TcontextDense: TensorArrayProtocol, ContextSparseTypes: TensorGroup, ContextRaggedValueTypes: TensorGroup, ContextRaggedSplitTypes: TensorGroup, FeatureListDenseTypes: TensorGroup, FeatureListSparseTypes: TensorGroup, FeatureListRaggedValueTypes: TensorGroup, FeatureListRaggedSplitTypes: TensorGroup>(serialized: StringTensor, debugName: StringTensor, contextSparseKeys: StringTensor, contextDenseKeys: StringTensor, contextRaggedKeys: StringTensor, featureListSparseKeys: StringTensor, featureListDenseKeys: StringTensor, featureListRaggedKeys: StringTensor, featureListDenseMissingAssumedEmpty: Tensor<Bool>, contextDenseDefaults: TcontextDense, ncontextSparse: Int64 = 0, contextDenseShapes: [TensorShape?], nfeatureListSparse: Int64 = 0, nfeatureListDense: Int64 = 0, featureListDenseShapes: [TensorShape?]) -> (
    contextSparseIndices: [Tensor<Int64>], contextSparseValues: ContextSparseTypes,
    contextSparseShapes: [Tensor<Int64>], contextDenseValues: TcontextDense,
    contextRaggedValues: ContextRaggedValueTypes,
    contextRaggedRowSplits: ContextRaggedSplitTypes, featureListSparseIndices: [Tensor<Int64>],
    featureListSparseValues: FeatureListSparseTypes, featureListSparseShapes: [Tensor<Int64>],
    featureListDenseValues: FeatureListDenseTypes, featureListDenseLengths: [Tensor<Int64>],
    featureListRaggedValues: FeatureListRaggedValueTypes,
    featureListRaggedOuterSplits: FeatureListRaggedSplitTypes,
    featureListRaggedInnerSplits: FeatureListRaggedSplitTypes
  )
```

#### Parameters

  - serialized: - serialized: A scalar or vector containing binary serialized SequenceExample protos.

### `parseSingleExample(serialized:denseDefaults:numSparse:sparseKeys:denseKeys:denseShapes:)`

Transforms a tf.Example proto (as a string) into typed tensors.

``` swift
@inlinable @inline(__always) public static func parseSingleExample<SparseTypes: TensorGroup, Tdense: TensorArrayProtocol>(serialized: StringTensor, denseDefaults: Tdense, numSparse: Int64, sparseKeys: [String], denseKeys: [String], denseShapes: [TensorShape?]) -> (
    sparseIndices: [Tensor<Int64>], sparseValues: SparseTypes, sparseShapes: [Tensor<Int64>],
    denseValues: Tdense
  )
```

#### Parameters

  - serialized: - serialized: A vector containing a batch of binary serialized Example protos.

### `parseSingleSequenceExample(serialized:featureListDenseMissingAssumedEmpty:contextSparseKeys:contextDenseKeys:featureListSparseKeys:featureListDenseKeys:contextDenseDefaults:debugName:contextDenseShapes:featureListDenseShapes:)`

Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.

``` swift
@inlinable @inline(__always) public static func parseSingleSequenceExample<ContextSparseTypes: TensorGroup, TcontextDense: TensorArrayProtocol, FeatureListDenseTypes: TensorGroup, FeatureListSparseTypes: TensorGroup>(serialized: StringTensor, featureListDenseMissingAssumedEmpty: StringTensor, contextSparseKeys: [StringTensor], contextDenseKeys: [StringTensor], featureListSparseKeys: [StringTensor], featureListDenseKeys: [StringTensor], contextDenseDefaults: TcontextDense, debugName: StringTensor, contextDenseShapes: [TensorShape?], featureListDenseShapes: [TensorShape?]) -> (
    contextSparseIndices: [Tensor<Int64>], contextSparseValues: ContextSparseTypes,
    contextSparseShapes: [Tensor<Int64>], contextDenseValues: TcontextDense,
    featureListSparseIndices: [Tensor<Int64>], featureListSparseValues: FeatureListSparseTypes,
    featureListSparseShapes: [Tensor<Int64>], featureListDenseValues: FeatureListDenseTypes
  )
```

#### Parameters

  - serialized: - serialized: A scalar containing a binary serialized SequenceExample proto.

### `parseTensor(serialized:)`

Transforms a serialized tensorflow.TensorProto proto into a Tensor.

``` swift
@inlinable @inline(__always) public static func parseTensor<OutType: TensorFlowScalar>(serialized: StringTensor) -> Tensor<OutType>
```

#### Parameters

  - serialized: - serialized: A scalar string containing a serialized TensorProto proto.

### `partitionedCall(args:f:config:configProto:executorType:)`

returns `f(inputs)`, where `f`'s body is placed and partitioned.

``` swift
@inlinable @inline(__always) public static func partitionedCall<Tin: TensorArrayProtocol, Tout: TensorGroup, FIn: TensorGroup, FOut: TensorGroup>(args: Tin, f: (FIn) -> FOut, config: String, configProto: String, executorType: String) -> Tout
```

#### Parameters

  - args: - args: A list of input tensors.

### `placeholder(shape:)`

A placeholder op for a value that will be fed into the computation.

``` swift
@inlinable @inline(__always) public static func placeholder<Dtype: TensorFlowScalar>(shape: TensorShape?) -> Tensor<Dtype>
```

N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime.

### `placeholderV2(shape:)`

A placeholder op for a value that will be fed into the computation.

``` swift
@inlinable @inline(__always) public static func placeholderV2<Dtype: TensorFlowScalar>(shape: TensorShape?) -> Tensor<Dtype>
```

N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime.

### `placeholderWithDefault(_:shape:)`

A placeholder op that passes through `input` when its output is not fed.

``` swift
@inlinable @inline(__always) public static func placeholderWithDefault<Dtype: TensorFlowScalar>(_ input: Tensor<Dtype>, shape: TensorShape?) -> Tensor<Dtype>
```

#### Parameters

  - input: - input: The default value to produce when `output` is not fed.

### `polygamma(_:_:)`

Compute the polygamma function \\(\\psi^{(n)}(x)\\).

``` swift
@inlinable @inline(__always) public static func polygamma<T: FloatingPoint & TensorFlowScalar>(_ a: Tensor<T>, _ x: Tensor<T>) -> Tensor<T>
```

The polygamma function is defined as:

\\(\\psi^{(a)}(x) = \\frac{d^a}{dx^a} \\psi(x)\\)

where \\(\\psi(x)\\) is the digamma function.
The polygamma function is defined only for non-negative integer orders \\a\\.

### `polymorphic(_:)`

``` swift
@inlinable @inline(__always) public static func polymorphic<T: TensorFlowScalar>(_ a: Tensor<T>) -> Tensor<T>
```

### `polymorphicDefaultOut()`

``` swift
@inlinable @inline(__always) public static func polymorphicDefaultOut<T: TensorFlowScalar>() -> Tensor<T>
```

### `polymorphicOut()`

``` swift
@inlinable @inline(__always) public static func polymorphicOut<T: TensorFlowScalar>() -> Tensor<T>
```

### `populationCount(_:)`

Computes element-wise population count (a.k.a. popcount, bitsum, bitcount).

``` swift
@inlinable @inline(__always) public static func populationCount<T: TensorFlowInteger>(_ x: Tensor<T>) -> Tensor<UInt8>
```

For each entry in `x`, calculates the number of `1` (on) bits in the binary
representation of that entry.

**NOTE**: It is more efficient to first `tf.bitcast` your tensors into
`int32` or `int64` and perform the bitcount on the result, than to feed in
8- or 16-bit inputs and then aggregate the resulting counts.

### `pow(_:_:)`

Computes the power of one value to another.

``` swift
@inlinable @inline(__always) public static func pow<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:

``` 
# tensor 'x' is [[2, 2]], [3, 3]]
# tensor 'y' is [[8, 16], [2, 3]]
tf.pow(x, y) ==> [[256, 65536], [9, 27]]
```

### `prefetchDataset(inputDataset:bufferSize:outputTypes:outputShapes:slackPeriod:legacyAutotune:)`

Creates a dataset that asynchronously prefetches elements from `input_dataset`.

``` swift
@inlinable @inline(__always) public static func prefetchDataset(inputDataset: VariantHandle, bufferSize: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?], slackPeriod: Int64 = 0, legacyAutotune: Bool = true) -> VariantHandle
```

### `prelinearize(_:shape:layout:)`

An op which linearizes one Tensor value to an opaque variant tensor.

``` swift
@inlinable @inline(__always) public static func prelinearize<Dtype: TensorFlowScalar>(_ input: Tensor<Dtype>, shape: TensorShape?, layout: [Int32]) -> VariantHandle
```

#### Parameters

  - input: - input: A tensor that will be linearized.

### `prelinearizeTuple(inputs:shapes:layouts:)`

An op which linearizes multiple Tensor values to an opaque variant tensor.

``` swift
@inlinable @inline(__always) public static func prelinearizeTuple<Dtypes: TensorArrayProtocol>(inputs: Dtypes, shapes: [TensorShape?], layouts: [Int32]) -> VariantHandle
```

#### Parameters

  - inputs: - inputs: A list of tensors that will be provided using the infeed mechanism.

### `preventGradient(_:message:)`

An identity op that triggers an error if a gradient is requested.

``` swift
@inlinable @inline(__always) public static func preventGradient<T: TensorFlowScalar>(_ input: Tensor<T>, message: String) -> Tensor<T>
```

When executed in a graph, this op outputs its input tensor as-is.

When building ops to compute gradients, the TensorFlow gradient system
will return an error when trying to lookup the gradient of this op,
because no gradient must ever be registered for this function.  This
op exists to prevent subtle bugs from silently returning unimplemented
gradients in some corner cases.

#### Parameters

  - input: - input: any tensor.

### `print(_:data:message:firstN:summarize:)`

Prints a list of tensors.

``` swift
@inlinable @inline(__always) public static func print<T: TensorFlowScalar, U: TensorArrayProtocol>(_ input: Tensor<T>, data: U, message: String, firstN: Int64 = -1, summarize: Int64 = 3) -> Tensor<T>
```

Passes `input` through to `output` and prints `data` when evaluating.

#### Parameters

  - input: - input: The tensor passed to `output`
  - data: - data: A list of tensors to print out when op is evaluated.

### `printV2(_:outputStream:end:)`

Prints a string scalar.

``` swift
@inlinable @inline(__always) public static func printV2(_ input: StringTensor, outputStream: String = "stderr", end: String = "")
```

Prints a string scalar to the desired output\_stream.

#### Parameters

  - input: - input: The string scalar to print.

### `priorityQueueV2(componentTypes:shapes:capacity:container:sharedName:)`

A queue that produces elements sorted by the first component value.

``` swift
@inlinable @inline(__always) public static func priorityQueueV2(componentTypes: [TensorDataType], shapes: [TensorShape?], capacity: Int64 = -1, container: String, sharedName: String) -> ResourceHandle
```

Note that the PriorityQueue requires the first component of any element
to be a scalar int64, in addition to the other elements declared by
component\_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
entry in their input (resp. output) lists.

### `privateThreadPoolDataset(inputDataset:numThreads:outputTypes:outputShapes:)`

Creates a dataset that uses a custom thread pool to compute `input_dataset`.

``` swift
@inlinable @inline(__always) public static func privateThreadPoolDataset(inputDataset: VariantHandle, numThreads: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `prod(_:reductionIndices:keepDims:)`

Computes the product of elements across dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func prod<T: TensorFlowNumeric, Tidx: TensorFlowIndex>(_ input: Tensor<T>, reductionIndices: Tensor<Tidx>, keepDims: Bool = false) -> Tensor<T>
```

Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

#### Parameters

  - input: - input: The tensor to reduce.

### `pyFunc(_:token:)`

Invokes a python function to compute func(input)-\>output.

``` swift
@inlinable @inline(__always) public static func pyFunc<Tin: TensorArrayProtocol, Tout: TensorGroup>(_ input: Tin, token: String) -> Tout
```

This operation is considered stateful. For a stateless version, see
PyFuncStateless.

#### Parameters

  - input: - input: List of Tensors that will provide input to the Op.

### `pyFuncStateless(_:token:)`

A stateless version of PyFunc.

``` swift
@inlinable @inline(__always) public static func pyFuncStateless<Tin: TensorArrayProtocol, Tout: TensorGroup>(_ input: Tin, token: String) -> Tout
```

### `qr(_:fullMatrices:)`

Computes the QR decompositions of one or more matrices.

``` swift
@inlinable @inline(__always) public static func qr<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, fullMatrices: Bool = false) -> (q: Tensor<T>, r: Tensor<T>)
```

Computes the QR decomposition of each inner matrix in `tensor` such that
`tensor[..., :, :] = q[..., :, :] * r[..., :,:])`

``` python
# a is a tensor.
# q is a tensor of orthonormal matrices.
# r is a tensor of upper triangular matrices.
q, r = qr(a)
q_full, r_full = qr(a, full_matrices=True)
```

#### Parameters

  - input: - input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.

### `quantizeAndDequantize(_:signedInput:numBits:rangeGiven:inputMin:inputMax:)`

Use QuantizeAndDequantizeV2 instead.

``` swift
@inlinable @inline(__always) public static func quantizeAndDequantize<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, signedInput: Bool = true, numBits: Int64 = 8, rangeGiven: Bool = false, inputMin: Double = 0, inputMax: Double = 0) -> Tensor<T>
```

### `quantizeAndDequantizeV2(_:inputMin:inputMax:signedInput:numBits:rangeGiven:roundMode:narrowRange:axis:)`

Quantizes then dequantizes a tensor.

``` swift
@inlinable @inline(__always) public static func quantizeAndDequantizeV2<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, inputMin: Tensor<T>, inputMax: Tensor<T>, signedInput: Bool = true, numBits: Int64 = 8, rangeGiven: Bool = false, roundMode: RoundMode = .halfToEven, narrowRange: Bool = false, axis: Int64 = -1) -> Tensor<T>
```

This op simulates the precision loss from the quantized forward pass by:

1.  Quantizing the tensor to fixed point numbers, which should match the target
    quantization method when it is used in inference.
2.  Dequantizing it back to floating point numbers for the following ops, most
    likely matmul.

There are different ways to quantize. This version uses only scaling, so 0.0
maps to 0.

From the specified 'num\_bits' in the quantized output type, it determines
minimum and maximum representable quantized values.

e.g.

If range\_given == False, the initial input\_min, input\_max will be determined
automatically as the minimum and maximum values in the input tensor, otherwise
the specified values of input\_min, input\_max are used.

Note: If the input\_min, input\_max are specified, they do not need to equal the
actual minimum and maximum values in the tensor. e.g. in some cases it may be
beneficial to specify these values such that the low probability extremes of the
input distribution are clipped.

This op determines the maximum scale\_factor that would map the initial
\[input\_min, input\_max\] range to a range that lies within the representable
quantized range.

It determines the scale from one of input\_min and input\_max, then updates the
other one to maximize the representable range.

e.g.

After determining the scale\_factor and updating the input range, it applies the
following to each value in the 'input' tensor.

output = round(clamp(value, input\_min, input\_max) \* scale\_factor) / scale\_factor.

The above round function rounds the value based on the given round\_mode.

#### Parameters

  - input: - input: Tensor to quantize and then dequantize.

### `quantizeAndDequantizeV3(_:inputMin:inputMax:numBits:signedInput:rangeGiven:narrowRange:axis:)`

Quantizes then dequantizes a tensor.

``` swift
@inlinable @inline(__always) public static func quantizeAndDequantizeV3<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, inputMin: Tensor<T>, inputMax: Tensor<T>, numBits: Tensor<Int32>, signedInput: Bool = true, rangeGiven: Bool = true, narrowRange: Bool = false, axis: Int64 = -1) -> Tensor<T>
```

This is almost identical to QuantizeAndDequantizeV2, except that num\_bits is a
tensor, so its value can change during training.

### `quantizeDownAndShrinkRange(_:inputMin:inputMax:)`

Convert the quantized 'input' tensor into a lower-precision 'output', using the

``` swift
@inlinable @inline(__always) public static func quantizeDownAndShrinkRange<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, inputMin: Tensor<Float>, inputMax: Tensor<Float>) -> (output: Tensor<OutType>, outputMin: Tensor<Float>, outputMax: Tensor<Float>)
```

actual distribution of the values to maximize the usage of the lower bit depth
and adjusting the output min and max ranges accordingly.

\[input\_min, input\_max\] are scalar floats that specify the range for the float
interpretation of the 'input' data. For example, if input\_min is -1.0f and
input\_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

This operator tries to squeeze as much precision as possible into an output with
a lower bit depth by calculating the actual min and max values found in the
data. For example, maybe that quint16 input has no values lower than 16,384 and
none higher than 49,152. That means only half the range is actually needed, all
the float interpretations are between -0.5f and 0.5f, so if we want to compress
the data into a quint8 output, we can use that range rather than the theoretical
\-1.0f to 1.0f that is suggested by the input min and max.

In practice, this is most useful for taking output from operations like
QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
may have large potential output ranges, but in practice have a distribution of
input values that only uses a small fraction of the possible range. By feeding
that output into this operator, we can reduce it from 32 bits down to 8 with
minimal loss of accuracy.

### `quantizeV2(_:minRange:maxRange:mode:roundMode:narrowRange:axis:ensureMinimumRange:)`

Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.

``` swift
@inlinable @inline(__always) public static func quantizeV2<T: TensorFlowScalar>(_ input: Tensor<Float>, minRange: Tensor<Float>, maxRange: Tensor<Float>, mode: Mode = .minCombined, roundMode: RoundMode7 = .halfAwayFromZero, narrowRange: Bool = false, axis: Int64 = -1, ensureMinimumRange: Double = 0.01) -> (output: Tensor<T>, outputMin: Tensor<Float>, outputMax: Tensor<Float>)
```

\[min\_range, max\_range\] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.  The
'round\_mode' attribute controls which rounding tie-breaking algorithm is used
when rounding float values to their quantized equivalents.

In 'MIN\_COMBINED' mode, each value of the tensor will undergo the following:

``` 
out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
if T == qint8: out[i] -= (range(T) + 1) / 2.0
```

here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

*MIN\_COMBINED Mode Example*

Assume the input is type float and has a possible range of \[0.0, 6.0\] and the
output type is quint8 (\[0, 255\]). The min\_range and max\_range values should be
specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
value of the input by 255/6 and cast to quint8.

If the output type was qint8 (\[-128, 127\]), the operation will additionally
subtract each value by 128 prior to casting, so that the range of values aligns
with the range of qint8.

If the mode is 'MIN\_FIRST', then this approach is used:

``` 
num_discrete_values = 1 << (# of bits in T)
range_adjust = num_discrete_values / (num_discrete_values - 1)
range = (range_max - range_min) * range_adjust
range_scale = num_discrete_values / range
quantized = round(input * range_scale) - round(range_min * range_scale) +
  numeric_limits<T>::min()
quantized = max(quantized, numeric_limits<T>::min())
quantized = min(quantized, numeric_limits<T>::max())
```

The biggest difference between this and MIN\_COMBINED is that the minimum range
is rounded first, before it's subtracted from the rounded value. With
MIN\_COMBINED, a small bias is introduced where repeated iterations of quantizing
and dequantizing will introduce a larger and larger error.

*SCALED mode Example*

`SCALED` mode matches the quantization approach used in
`QuantizeAndDequantize{V2|V3}`.

If the mode is `SCALED`, the quantization is performed by multiplying each
input value by a scaling\_factor.
The scaling\_factor is determined from `min_range` and `max_range` to be as large
as possible such that the range from `min_range` to `max_range` is representable
within values of type T.

``` c++
 
  const int min_T = std::numeric_limits<T>::min();
  const int max_T = std::numeric_limits<T>::max();
  const float max_float = std::numeric_limits<float>::max();
 
  const float scale_factor_from_min_side =
      (min_T * min_range > 0) ? min_T / min_range : max_float;
  const float scale_factor_from_max_side =
      (max_T * max_range > 0) ? max_T / max_range : max_float;
 
  const float scale_factor = std::min(scale_factor_from_min_side,
                                      scale_factor_from_max_side);
```

We next use the scale\_factor to adjust min\_range and max\_range as follows:

``` c++
      min_range = min_T / scale_factor;
      max_range = max_T / scale_factor;
```

e.g. if T = qint8, and initially min\_range = -10, and max\_range = 9, we would
compare -128/-10.0 = 12.8 to 127/9.0 = 14.11, and set scaling\_factor = 12.8
In this case, min\_range would remain -10, but max\_range would be adjusted to
127 / 12.8 = 9.921875

So we will quantize input values in the range (-10, 9.921875) to (-128, 127).

The input tensor can now be quantized by clipping values to the range
`min_range` to `max_range`, then multiplying by scale\_factor as follows:

``` c++
result = round(min(max_range, max(min_range, input)) * scale_factor)
```

The adjusted `min_range` and `max_range` are returned as outputs 2 and 3 of
this operation. These outputs should be used as the range for any further
calculations.

*narrow\_range (bool) attribute*

If true, we do not use the minimum quantized value.
i.e. for int8 the quantized output, it would be restricted to the range
\-127..127 instead of the full -128..127 range.
This is provided for compatibility with certain inference backends.
(Only applies to SCALED mode)

*axis (int) attribute*

An optional `axis` attribute can specify a dimension index of the input tensor,
such that quantization ranges will be calculated and applied separately for each
slice of the tensor along that dimension. This is useful for per-channel
quantization.

If axis is specified, min\_range and max\_range

if `axis`=None, per-tensor quantization is performed as normal.

*ensure\_minimum\_range (float) attribute*

Ensures the minimum quantization range is at least this value.
The legacy default value for this is 0.01, but it is strongly suggested to
set it to 0 for new uses.

### `quantizedAdd(_:_:minX:maxX:minY:maxY:)`

Returns x + y element-wise, working on quantized buffers.

``` swift
@inlinable @inline(__always) public static func quantizedAdd<T1: TensorFlowScalar, T2: TensorFlowScalar, Toutput: TensorFlowScalar>(_ x: Tensor<T1>, _ y: Tensor<T2>, minX: Tensor<Float>, maxX: Tensor<Float>, minY: Tensor<Float>, maxY: Tensor<Float>) -> (z: Tensor<Toutput>, minZ: Tensor<Float>, maxZ: Tensor<Float>)
```

### `quantizedAvgPool(_:minInput:maxInput:ksize:strides:padding:)`

Produces the average pool of the input tensor for quantized types.

``` swift
@inlinable @inline(__always) public static func quantizedAvgPool<T: TensorFlowScalar>(_ input: Tensor<T>, minInput: Tensor<Float>, maxInput: Tensor<Float>, ksize: [Int32], strides: [Int32], padding: Padding) -> (output: Tensor<T>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

#### Parameters

  - input: - input: 4-D with shape `[batch, height, width, channels]`.

### `quantizedBatchNormWithGlobalNormalization(t:tMin:tMax:m:mMin:mMax:v:vMin:vMax:beta:betaMin:betaMax:gamma:gammaMin:gammaMax:varianceEpsilon:scaleAfterNormalization:)`

Quantized Batch normalization.

``` swift
@inlinable @inline(__always) public static func quantizedBatchNormWithGlobalNormalization<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(t: Tensor<Tinput>, tMin: Tensor<Float>, tMax: Tensor<Float>, m: Tensor<Tinput>, mMin: Tensor<Float>, mMax: Tensor<Float>, v: Tensor<Tinput>, vMin: Tensor<Float>, vMax: Tensor<Float>, beta: Tensor<Tinput>, betaMin: Tensor<Float>, betaMax: Tensor<Float>, gamma: Tensor<Tinput>, gammaMin: Tensor<Float>, gammaMax: Tensor<Float>, varianceEpsilon: Double, scaleAfterNormalization: Bool) -> (result: Tensor<OutType>, resultMin: Tensor<Float>, resultMax: Tensor<Float>)
```

This op is deprecated and will be removed in the future. Prefer
`tf.nn.batch_normalization`.

#### Parameters

  - t: - t: A 4D input Tensor.
  - m: - m: A 1D mean Tensor with size matching the last dimension of t. This is the first output from tf.nn.moments, or a saved moving average thereof.
  - v: - v: A 1D variance Tensor with size matching the last dimension of t. This is the second output from tf.nn.moments, or a saved moving average thereof.
  - beta: - beta: A 1D beta Tensor with size matching the last dimension of t. An offset to be added to the normalized tensor.
  - gamma: - gamma: A 1D gamma Tensor with size matching the last dimension of t. If "scale\_after\_normalization" is true, this tensor will be multiplied with the normalized tensor.

### `quantizedBiasAdd(_:bias:minInput:maxInput:minBias:maxBias:)`

Adds Tensor 'bias' to Tensor 'input' for Quantized types.

``` swift
@inlinable @inline(__always) public static func quantizedBiasAdd<T1: TensorFlowScalar, T2: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<T1>, bias: Tensor<T2>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minBias: Tensor<Float>, maxBias: Tensor<Float>) -> (output: Tensor<OutType>, minOut: Tensor<Float>, maxOut: Tensor<Float>)
```

Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

#### Parameters

  - bias: - bias: A 1D bias Tensor with size matching the last dimension of 'input'.

### `quantizedConcat(concatDim:_:inputMins:inputMaxes:)`

Concatenates quantized tensors along one dimension.

``` swift
@inlinable @inline(__always) public static func quantizedConcat<T: TensorFlowScalar>(concatDim: Tensor<Int32>, _ values: [Tensor<T>], inputMins: [Tensor<Float>], inputMaxes: [Tensor<Float>]) -> (output: Tensor<T>, outputMin: Tensor<Float>, outputMax: Tensor<Float>)
```

#### Parameters

  - values: - values: The `N` Tensors to concatenate. Their ranks and types must match, and their sizes must match in all dimensions except `concat_dim`.

### `quantizedConv2D(_:filter:minInput:maxInput:minFilter:maxFilter:strides:padding:dilations:)`

Computes a 2D convolution given quantized 4D input and filter tensors.

``` swift
@inlinable @inline(__always) public static func quantizedConv2D<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

The inputs are quantized tensors where the lowest value represents the real
number of the associated minimum, and the highest represents the maximum.
This means that you can only interpret the quantized output in the same way, by
taking the returned minimum and maximum values into account.

#### Parameters

  - filter: - filter: filter's input\_depth dimension must match input's depth dimensions.

### `quantizedConv2DAndRelu(_:filter:minInput:maxInput:minFilter:maxFilter:strides:padding:dilations:paddingList:)`

``` swift
@inlinable @inline(__always) public static func quantizedConv2DAndRelu<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1], paddingList: [Int32]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

### `quantizedConv2DAndReluAndRequantize(_:filter:minInput:maxInput:minFilter:maxFilter:minFreezedOutput:maxFreezedOutput:strides:padding:dilations:paddingList:)`

``` swift
@inlinable @inline(__always) public static func quantizedConv2DAndReluAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, minFreezedOutput: Tensor<Float>, maxFreezedOutput: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1], paddingList: [Int32]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

### `quantizedConv2DAndRequantize(_:filter:minInput:maxInput:minFilter:maxFilter:minFreezedOutput:maxFreezedOutput:strides:padding:dilations:paddingList:)`

``` swift
@inlinable @inline(__always) public static func quantizedConv2DAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, minFreezedOutput: Tensor<Float>, maxFreezedOutput: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1], paddingList: [Int32]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

### `quantizedConv2DPerChannel(_:filter:minInput:maxInput:minFilter:maxFilter:strides:padding:dilations:)`

Computes QuantizedConv2D per channel.

``` swift
@inlinable @inline(__always) public static func quantizedConv2DPerChannel<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

#### Parameters

  - input: - input: The original input tensor.
  - filter: - filter: The original filter tensor.

### `quantizedConv2DWithBias(_:filter:bias:minInput:maxInput:minFilter:maxFilter:strides:padding:dilations:paddingList:)`

``` swift
@inlinable @inline(__always) public static func quantizedConv2DWithBias<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, bias: Tensor<Float>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1], paddingList: [Int32]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

### `quantizedConv2DWithBiasAndRelu(_:filter:bias:minInput:maxInput:minFilter:maxFilter:strides:padding:dilations:paddingList:)`

``` swift
@inlinable @inline(__always) public static func quantizedConv2DWithBiasAndRelu<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, bias: Tensor<Float>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1], paddingList: [Int32]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

### `quantizedConv2DWithBiasAndReluAndRequantize(_:filter:bias:minInput:maxInput:minFilter:maxFilter:minFreezedOutput:maxFreezedOutput:strides:padding:dilations:paddingList:)`

``` swift
@inlinable @inline(__always) public static func quantizedConv2DWithBiasAndReluAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, bias: Tensor<Tbias>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, minFreezedOutput: Tensor<Float>, maxFreezedOutput: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1], paddingList: [Int32]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

### `quantizedConv2DWithBiasAndRequantize(_:filter:bias:minInput:maxInput:minFilter:maxFilter:minFreezedOutput:maxFreezedOutput:strides:padding:dilations:paddingList:)`

``` swift
@inlinable @inline(__always) public static func quantizedConv2DWithBiasAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, bias: Tensor<Tbias>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, minFreezedOutput: Tensor<Float>, maxFreezedOutput: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1], paddingList: [Int32]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

### `quantizedConv2DWithBiasSignedSumAndReluAndRequantize(_:filter:bias:minInput:maxInput:minFilter:maxFilter:minFreezedOutput:maxFreezedOutput:summand:minSummand:maxSummand:strides:padding:dilations:paddingList:)`

``` swift
@inlinable @inline(__always) public static func quantizedConv2DWithBiasSignedSumAndReluAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, Tsummand: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, bias: Tensor<Tbias>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, minFreezedOutput: Tensor<Float>, maxFreezedOutput: Tensor<Float>, summand: Tensor<Tsummand>, minSummand: Tensor<Float>, maxSummand: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1], paddingList: [Int32]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

### `quantizedConv2DWithBiasSumAndRelu(_:filter:bias:minInput:maxInput:minFilter:maxFilter:summand:strides:padding:dilations:paddingList:)`

``` swift
@inlinable @inline(__always) public static func quantizedConv2DWithBiasSumAndRelu<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, bias: Tensor<Float>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, summand: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1], paddingList: [Int32]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

### `quantizedConv2DWithBiasSumAndReluAndRequantize(_:filter:bias:minInput:maxInput:minFilter:maxFilter:minFreezedOutput:maxFreezedOutput:summand:minSummand:maxSummand:strides:padding:dilations:paddingList:)`

``` swift
@inlinable @inline(__always) public static func quantizedConv2DWithBiasSumAndReluAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, Tsummand: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, bias: Tensor<Tbias>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, minFreezedOutput: Tensor<Float>, maxFreezedOutput: Tensor<Float>, summand: Tensor<Tsummand>, minSummand: Tensor<Float>, maxSummand: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1], paddingList: [Int32]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

### `quantizedDepthwiseConv2D(_:filter:minInput:maxInput:minFilter:maxFilter:strides:padding:dilations:)`

Computes quantized depthwise Conv2D.

``` swift
@inlinable @inline(__always) public static func quantizedDepthwiseConv2D<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

#### Parameters

  - input: - input: The original input tensor.
  - filter: - filter: The original filter tensor.

### `quantizedDepthwiseConv2DWithBias(_:filter:bias:minInput:maxInput:minFilter:maxFilter:strides:padding:dilations:)`

Computes quantized depthwise Conv2D with Bias.

``` swift
@inlinable @inline(__always) public static func quantizedDepthwiseConv2DWithBias<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, bias: Tensor<Float>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

#### Parameters

  - input: - input: The original input tensor.
  - filter: - filter: The original filter tensor.
  - bias: - bias: The original bias tensor.

### `quantizedDepthwiseConv2DWithBiasAndRelu(_:filter:bias:minInput:maxInput:minFilter:maxFilter:strides:padding:dilations:)`

Computes quantized depthwise Conv2D with Bias and Relu.

``` swift
@inlinable @inline(__always) public static func quantizedDepthwiseConv2DWithBiasAndRelu<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, bias: Tensor<Float>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

#### Parameters

  - input: - input: The original input tensor.
  - filter: - filter: The original filter tensor.
  - bias: - bias: The original bias tensor.

### `quantizedDepthwiseConv2DWithBiasAndReluAndRequantize(_:filter:bias:minInput:maxInput:minFilter:maxFilter:minFreezedOutput:maxFreezedOutput:strides:padding:dilations:)`

Computes quantized depthwise Conv2D with Bias, Relu and Requantize.

``` swift
@inlinable @inline(__always) public static func quantizedDepthwiseConv2DWithBiasAndReluAndRequantize<Tinput: TensorFlowScalar, Tfilter: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, filter: Tensor<Tfilter>, bias: Tensor<Tbias>, minInput: Tensor<Float>, maxInput: Tensor<Float>, minFilter: Tensor<Float>, maxFilter: Tensor<Float>, minFreezedOutput: Tensor<Float>, maxFreezedOutput: Tensor<Float>, strides: [Int32], padding: Padding, dilations: [Int32] = [1, 1, 1, 1]) -> (output: Tensor<OutType>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

#### Parameters

  - input: - input: The original input tensor.
  - filter: - filter: The original filter tensor.
  - bias: - bias: The original bias tensor.

### `quantizedInstanceNorm(_:xMin:xMax:outputRangeGiven:givenYMin:givenYMax:varianceEpsilon:minSeparation:)`

Quantized Instance normalization.

``` swift
@inlinable @inline(__always) public static func quantizedInstanceNorm<T: TensorFlowScalar>(_ x: Tensor<T>, xMin: Tensor<Float>, xMax: Tensor<Float>, outputRangeGiven: Bool = false, givenYMin: Double = 0, givenYMax: Double = 0, varianceEpsilon: Double = 1e-05, minSeparation: Double = 0.001) -> (y: Tensor<T>, yMin: Tensor<Float>, yMax: Tensor<Float>)
```

#### Parameters

  - x: - x: A 4D input Tensor.

### `quantizedMatMul(_:_:minA:maxA:minB:maxB:transposeA:transposeB:tactivation:)`

Perform a quantized matrix multiplication of  `a` by the matrix `b`.

``` swift
@inlinable @inline(__always) public static func quantizedMatMul<T1: TensorFlowScalar, T2: TensorFlowScalar, Toutput: TensorFlowScalar>(_ a: Tensor<T1>, _ b: Tensor<T2>, minA: Tensor<Float>, maxA: Tensor<Float>, minB: Tensor<Float>, maxB: Tensor<Float>, transposeA: Bool = false, transposeB: Bool = false, tactivation: TensorDataType) -> (out: Tensor<Toutput>, minOut: Tensor<Float>, maxOut: Tensor<Float>)
```

The inputs must be two-dimensional matrices and the inner dimension of
`a` (after being transposed if `transpose_a` is non-zero) must match the
outer dimension of `b` (after being transposed if `transposed_b` is
non-zero).

#### Parameters

  - a: - a: Must be a two-dimensional tensor.
  - b: - b: Must be a two-dimensional tensor.

### `quantizedMatMulWithBias(_:_:bias:minA:maxA:minB:maxB:transposeA:transposeB:inputQuantMode:)`

Performs a quantized matrix multiplication of `a` by the matrix `b` with bias
add.

``` swift
@inlinable @inline(__always) public static func quantizedMatMulWithBias<T1: TensorFlowScalar, T2: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, Toutput: TensorFlowScalar>(_ a: Tensor<T1>, _ b: Tensor<T2>, bias: Tensor<Tbias>, minA: Tensor<Float>, maxA: Tensor<Float>, minB: Tensor<Float>, maxB: Tensor<Float>, transposeA: Bool = false, transposeB: Bool = false, inputQuantMode: InputQuantMode = .minFirst) -> (out: Tensor<Toutput>, minOut: Tensor<Float>, maxOut: Tensor<Float>)
```

The inputs must be two-dimensional matrices and 1D bias vector. And the inner
dimension of `a` (after being transposed if `transpose_a` is non-zero) must
match the outer dimension of `b` (after being transposed if `transposed_b` is
non-zero). Then do broadcast add operation with bias values on the matrix
mulplication result. The bias size must match inner dimension of `b`.

#### Parameters

  - a: - a: A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
  - b: - b: A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
  - bias: - bias: A 1D bias tensor with size matching inner dimension of `b` (after being transposed if `transposed_b` is non-zero).

### `quantizedMatMulWithBiasAndRelu(_:_:bias:minA:maxA:minB:maxB:transposeA:transposeB:inputQuantMode:)`

Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias
add and relu fusion.

``` swift
@inlinable @inline(__always) public static func quantizedMatMulWithBiasAndRelu<T1: TensorFlowScalar, T2: TensorFlowScalar, Toutput: TensorFlowScalar>(_ a: Tensor<T1>, _ b: Tensor<T2>, bias: Tensor<Float>, minA: Tensor<Float>, maxA: Tensor<Float>, minB: Tensor<Float>, maxB: Tensor<Float>, transposeA: Bool = false, transposeB: Bool = false, inputQuantMode: InputQuantMode = .minFirst) -> (out: Tensor<Toutput>, minOut: Tensor<Float>, maxOut: Tensor<Float>)
```

The inputs must be two-dimensional matrices and 1D bias vector. And the inner
dimension of `a` (after being transposed if `transpose_a` is non-zero) must
match the outer dimension of `b` (after being transposed if `transposed_b` is
non-zero). Then do broadcast add operation with bias values on the matrix
mulplication result. The bias size must match inner dimension of `b`. Then do
relu activation to get non-negative result.

#### Parameters

  - a: - a: A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
  - b: - b: A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
  - bias: - bias: A 1D bias tensor with size matching with inner dimension of `b` (after being transposed if `transposed_b` is non-zero).

### `quantizedMatMulWithBiasAndReluAndRequantize(_:_:bias:minA:maxA:minB:maxB:minFreezedOutput:maxFreezedOutput:transposeA:transposeB:inputQuantMode:)`

Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias
add and relu and requantize fusion.

``` swift
@inlinable @inline(__always) public static func quantizedMatMulWithBiasAndReluAndRequantize<T1: TensorFlowScalar, T2: TensorFlowScalar, Tbias: FloatingPoint & TensorFlowScalar, Toutput: TensorFlowScalar>(_ a: Tensor<T1>, _ b: Tensor<T2>, bias: Tensor<Tbias>, minA: Tensor<Float>, maxA: Tensor<Float>, minB: Tensor<Float>, maxB: Tensor<Float>, minFreezedOutput: Tensor<Float>, maxFreezedOutput: Tensor<Float>, transposeA: Bool = false, transposeB: Bool = false, inputQuantMode: InputQuantMode = .minFirst) -> (out: Tensor<Toutput>, minOut: Tensor<Float>, maxOut: Tensor<Float>)
```

The inputs must be two-dimensional matrices and 1D bias vector. And the inner
dimension of `a` (after being transposed if `transpose_a` is non-zero) must
match the outer dimension of `b` (after being transposed if `transposed_b` is
non-zero). Then do broadcast add operation with bias values on the matrix
mulplication result. The bias size must match inner dimension of `b`.  Then do
relu activation to get non-negative result. Then do requantize operation to get
final uint8 result.

#### Parameters

  - a: - a: A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
  - b: - b: A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
  - bias: - bias: A 1D bias tensor with size matching with inner dimension of `b` (after being transposed if `transposed_b` is non-zero).

### `quantizedMaxPool(_:minInput:maxInput:ksize:strides:padding:)`

Produces the max pool of the input tensor for quantized types.

``` swift
@inlinable @inline(__always) public static func quantizedMaxPool<T: TensorFlowScalar>(_ input: Tensor<T>, minInput: Tensor<Float>, maxInput: Tensor<Float>, ksize: [Int32], strides: [Int32], padding: Padding) -> (output: Tensor<T>, minOutput: Tensor<Float>, maxOutput: Tensor<Float>)
```

#### Parameters

  - input: - input: The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.

### `quantizedMul(_:_:minX:maxX:minY:maxY:)`

Returns x \* y element-wise, working on quantized buffers.

``` swift
@inlinable @inline(__always) public static func quantizedMul<T1: TensorFlowScalar, T2: TensorFlowScalar, Toutput: TensorFlowScalar>(_ x: Tensor<T1>, _ y: Tensor<T2>, minX: Tensor<Float>, maxX: Tensor<Float>, minY: Tensor<Float>, maxY: Tensor<Float>) -> (z: Tensor<Toutput>, minZ: Tensor<Float>, maxZ: Tensor<Float>)
```

### `quantizedRelu(features:minFeatures:maxFeatures:)`

Computes Quantized Rectified Linear: `max(features, 0)`

``` swift
@inlinable @inline(__always) public static func quantizedRelu<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(features: Tensor<Tinput>, minFeatures: Tensor<Float>, maxFeatures: Tensor<Float>) -> (
    activations: Tensor<OutType>, minActivations: Tensor<Float>, maxActivations: Tensor<Float>
  )
```

### `quantizedRelu6(features:minFeatures:maxFeatures:)`

Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`

``` swift
@inlinable @inline(__always) public static func quantizedRelu6<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(features: Tensor<Tinput>, minFeatures: Tensor<Float>, maxFeatures: Tensor<Float>) -> (
    activations: Tensor<OutType>, minActivations: Tensor<Float>, maxActivations: Tensor<Float>
  )
```

### `quantizedReluX(features:maxValue:minFeatures:maxFeatures:)`

Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`

``` swift
@inlinable @inline(__always) public static func quantizedReluX<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(features: Tensor<Tinput>, maxValue: Tensor<Float>, minFeatures: Tensor<Float>, maxFeatures: Tensor<Float>) -> (
    activations: Tensor<OutType>, minActivations: Tensor<Float>, maxActivations: Tensor<Float>
  )
```

### `quantizedReshape(_:shape:inputMin:inputMax:)`

Reshapes a quantized tensor as per the Reshape op.

``` swift
@inlinable @inline(__always) public static func quantizedReshape<T: TensorFlowScalar, Tshape: TensorFlowIndex>(_ tensor: Tensor<T>, shape: Tensor<Tshape>, inputMin: Tensor<Float>, inputMax: Tensor<Float>) -> (output: Tensor<T>, outputMin: Tensor<Float>, outputMax: Tensor<Float>)
```

``` 

- Parameters:
    - shape: Defines the shape of the output tensor.
    - input_min: The minimum value of the input.
    - input_max: The maximum value of the input.

- Outputs:
    - output_min: This value is copied from input_min.
    - output_max: This value is copied from input_max.
```

### `quantizedResizeBilinear(images:size:min:max:alignCorners:halfPixelCenters:)`

Resize quantized `images` to `size` using quantized bilinear interpolation.

``` swift
@inlinable @inline(__always) public static func quantizedResizeBilinear<T: FloatingPoint & TensorFlowScalar>(images: Tensor<T>, size: Tensor<Int32>, min: Tensor<Float>, max: Tensor<Float>, alignCorners: Bool = false, halfPixelCenters: Bool = false) -> (resizedImages: Tensor<T>, outMin: Tensor<Float>, outMax: Tensor<Float>)
```

Input images and output images must be quantized types.

#### Parameters

  - images: - images: 4-D with shape `[batch, height, width, channels]`.
  - size: - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new size for the images.

### `queueCloseV2(handle:cancelPendingEnqueues:)`

Closes the given queue.

``` swift
@inlinable @inline(__always) public static func queueCloseV2(handle: ResourceHandle, cancelPendingEnqueues: Bool = false)
```

This operation signals that no more elements will be enqueued in the
given queue. Subsequent Enqueue(Many) operations will fail.
Subsequent Dequeue(Many) operations will continue to succeed if
sufficient elements remain in the queue. Subsequent Dequeue(Many)
operations that would block will fail immediately.

#### Parameters

  - handle: - handle: The handle to a queue.

### `queueDequeueManyV2(handle:n:timeoutMs:)`

Dequeues `n` tuples of one or more tensors from the given queue.

``` swift
@inlinable @inline(__always) public static func queueDequeueManyV2<ComponentTypes: TensorGroup>(handle: ResourceHandle, n: Tensor<Int32>, timeoutMs: Int64 = -1) -> ComponentTypes
```

If the queue is closed and there are fewer than `n` elements, then an
OutOfRange error is returned.

This operation concatenates queue-element component tensors along the
0th dimension to make a single component tensor.  All of the components
in the dequeued tuple will have size `n` in the 0th dimension.

This operation has `k` outputs, where `k` is the number of components in
the tuples stored in the given queue, and output `i` is the ith
component of the dequeued tuple.

N.B. If the queue is empty, this operation will block until `n` elements
have been dequeued (or 'timeout\_ms' elapses, if specified).

#### Parameters

  - handle: - handle: The handle to a queue.
  - n: - n: The number of tuples to dequeue.

### `queueDequeueUpToV2(handle:n:timeoutMs:)`

Dequeues `n` tuples of one or more tensors from the given queue.

``` swift
@inlinable @inline(__always) public static func queueDequeueUpToV2<ComponentTypes: TensorGroup>(handle: ResourceHandle, n: Tensor<Int32>, timeoutMs: Int64 = -1) -> ComponentTypes
```

This operation is not supported by all queues.  If a queue does not support
DequeueUpTo, then an Unimplemented error is returned.

If the queue is closed and there are more than 0 but less than `n`
elements remaining, then instead of returning an OutOfRange error like
QueueDequeueMany, less than `n` elements are returned immediately.  If
the queue is closed and there are 0 elements left in the queue, then
an OutOfRange error is returned just like in QueueDequeueMany.
Otherwise the behavior is identical to QueueDequeueMany:

This operation concatenates queue-element component tensors along the
0th dimension to make a single component tensor.  All of the components
in the dequeued tuple will have size n in the 0th dimension.

This operation has `k` outputs, where `k` is the number of components in
the tuples stored in the given queue, and output `i` is the ith
component of the dequeued tuple.

#### Parameters

  - handle: - handle: The handle to a queue.
  - n: - n: The number of tuples to dequeue.

### `queueDequeueV2(handle:timeoutMs:)`

Dequeues a tuple of one or more tensors from the given queue.

``` swift
@inlinable @inline(__always) public static func queueDequeueV2<ComponentTypes: TensorGroup>(handle: ResourceHandle, timeoutMs: Int64 = -1) -> ComponentTypes
```

This operation has k outputs, where k is the number of components
in the tuples stored in the given queue, and output i is the ith
component of the dequeued tuple.

N.B. If the queue is empty, this operation will block until an element
has been dequeued (or 'timeout\_ms' elapses, if specified).

#### Parameters

  - handle: - handle: The handle to a queue.

### `queueEnqueueManyV2(handle:components:timeoutMs:)`

Enqueues zero or more tuples of one or more tensors in the given queue.

``` swift
@inlinable @inline(__always) public static func queueEnqueueManyV2<Tcomponents: TensorArrayProtocol>(handle: ResourceHandle, components: Tcomponents, timeoutMs: Int64 = -1)
```

This operation slices each component tensor along the 0th dimension to
make multiple queue elements. All of the tuple components must have the
same size in the 0th dimension.

The components input has k elements, which correspond to the components of
tuples stored in the given queue.

N.B. If the queue is full, this operation will block until the given
elements have been enqueued (or 'timeout\_ms' elapses, if specified).

#### Parameters

  - handle: - handle: The handle to a queue.
  - components: - components: One or more tensors from which the enqueued tensors should be taken.

### `queueEnqueueV2(handle:components:timeoutMs:)`

Enqueues a tuple of one or more tensors in the given queue.

``` swift
@inlinable @inline(__always) public static func queueEnqueueV2<Tcomponents: TensorArrayProtocol>(handle: ResourceHandle, components: Tcomponents, timeoutMs: Int64 = -1)
```

The components input has k elements, which correspond to the components of
tuples stored in the given queue.

N.B. If the queue is full, this operation will block until the given
element has been enqueued (or 'timeout\_ms' elapses, if specified).

#### Parameters

  - handle: - handle: The handle to a queue.
  - components: - components: One or more tensors from which the enqueued tensors should be taken.

### `queueIsClosedV2(handle:)`

Returns true if queue is closed.

``` swift
@inlinable @inline(__always) public static func queueIsClosedV2(handle: ResourceHandle) -> Tensor<Bool>
```

This operation returns true if the queue is closed and false if the queue
is open.

#### Parameters

  - handle: - handle: The handle to a queue.

### `queueSizeV2(handle:)`

Computes the number of elements in the given queue.

``` swift
@inlinable @inline(__always) public static func queueSizeV2(handle: ResourceHandle) -> Tensor<Int32>
```

#### Parameters

  - handle: - handle: The handle to a queue.

### `rFFT(_:fftLength:)`

Real-valued fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func rFFT<Treal: FloatingPoint & TensorFlowScalar, Tcomplex: TensorFlowScalar>(_ input: Tensor<Treal>, fftLength: Tensor<Int32>) -> Tensor<Tcomplex>
```

Computes the 1-dimensional discrete Fourier transform of a real-valued signal
over the inner-most dimension of `input`.

Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
`fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
followed by the `fft_length / 2` positive-frequency terms.

Along the axis `RFFT` is computed on, if `fft_length` is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.

#### Parameters

  - input: - input: A float32 tensor.

### `rFFT2D(_:fftLength:)`

2D real-valued fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func rFFT2D<Treal: FloatingPoint & TensorFlowScalar, Tcomplex: TensorFlowScalar>(_ input: Tensor<Treal>, fftLength: Tensor<Int32>) -> Tensor<Tcomplex>
```

Computes the 2-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 2 dimensions of `input`.

Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

Along each axis `RFFT2D` is computed on, if `fft_length` is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.

#### Parameters

  - input: - input: A float32 tensor.

### `rFFT3D(_:fftLength:)`

3D real-valued fast Fourier transform.

``` swift
@inlinable @inline(__always) public static func rFFT3D<Treal: FloatingPoint & TensorFlowScalar, Tcomplex: TensorFlowScalar>(_ input: Tensor<Treal>, fftLength: Tensor<Int32>) -> Tensor<Tcomplex>
```

Computes the 3-dimensional discrete Fourier transform of a real-valued signal
over the inner-most 3 dimensions of `input`.

Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
`fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
of `output`: the zero-frequency term, followed by the `fft_length / 2`
positive-frequency terms.

Along each axis `RFFT3D` is computed on, if `fft_length` is smaller than the
corresponding dimension of `input`, the dimension is cropped. If it is larger,
the dimension is padded with zeros.

#### Parameters

  - input: - input: A float32 tensor.

### `rGBToHSV(images:)`

Converts one or more images from RGB to HSV.

``` swift
@inlinable @inline(__always) public static func rGBToHSV<T: FloatingPoint & TensorFlowScalar>(images: Tensor<T>) -> Tensor<T>
```

Outputs a tensor of the same shape as the `images` tensor, containing the HSV
value of the pixels. The output is only well defined if the value in `images`
are in `[0,1]`.

`output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
`output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

#### Parameters

  - images: - images: 1-D or higher rank. RGB data to convert. Last dimension must be size 3.

### `raggedGather(paramsNestedSplits:paramsDenseValues:indices:oUTPUTRAGGEDRANK:)`

Gather ragged slices from `params` axis `0` according to `indices`.

``` swift
@inlinable @inline(__always) public static func raggedGather<Tvalues: TensorFlowScalar, Tindices: TensorFlowIndex, Tsplits: TensorFlowIndex>(paramsNestedSplits: [Tensor<Tsplits>], paramsDenseValues: Tensor<Tvalues>, indices: Tensor<Tindices>, oUTPUTRAGGEDRANK: Int64) -> (outputNestedSplits: [Tensor<Tsplits>], outputDenseValues: Tensor<Tvalues>)
```

Outputs a `RaggedTensor` output composed from `output_dense_values` and
`output_nested_splits`, such that:

``` python
output.shape = indices.shape + params.shape[1:]
output.ragged_rank = indices.shape.ndims + params.ragged_rank
output[i...j, d0...dn] = params[indices[i...j], d0...dn]
```

where

(Note: This c++ op is used to implement the higher-level python
`tf.ragged.gather` op, which also supports ragged indices.)

#### Parameters

  - indices: - indices: Indices in the outermost dimension of `params` of the values that should be gathered.

### `raggedRange(starts:limits:deltas:)`

Returns a `RaggedTensor` containing the specified sequences of numbers.

``` swift
@inlinable @inline(__always) public static func raggedRange<T: TensorFlowNumeric, Tsplits: TensorFlowIndex>(starts: Tensor<T>, limits: Tensor<T>, deltas: Tensor<T>) -> (rtNestedSplits: Tensor<Tsplits>, rtDenseValues: Tensor<T>)
```

Returns a `RaggedTensor` `result` composed from `rt_dense_values` and
`rt_nested_splits`, such that
`result[i] = range(starts[i], limits[i], deltas[i])`.

``` python
(rt_nested_splits, rt_dense_values) = ragged_range(
      starts=[2, 5, 8], limits=[3, 5, 12], deltas=1)
result = tf.ragged.from_row_splits(rt_dense_values, rt_nested_splits)
print(result)
<tf.RaggedTensor [[2], [], [8, 9, 10, 11]] >
```

The input tensors `starts`, `limits`, and `deltas` may be scalars or vectors.
The vector inputs must all have the same size.  Scalar inputs are broadcast
to match the size of the vector inputs.

#### Parameters

  - starts: - starts: The starts of each range.
  - limits: - limits: The limits of each range.
  - deltas: - deltas: The deltas of each range.

### `raggedTensorFromVariant(encodedRagged:inputRaggedRank:outputRaggedRank:)`

Decodes a `variant` Tensor into a `RaggedTensor`.

``` swift
@inlinable @inline(__always) public static func raggedTensorFromVariant<Tvalues: TensorFlowScalar, Tsplits: TensorFlowIndex>(encodedRagged: VariantHandle, inputRaggedRank: Int64, outputRaggedRank: Int64) -> (outputNestedSplits: [Tensor<Tsplits>], outputDenseValues: Tensor<Tvalues>)
```

Decodes the given `variant` Tensor and returns a `RaggedTensor`. The input
could be a scalar, meaning it encodes a single `RaggedTensor` with ragged\_rank
`output_ragged_rank`. It could also have an arbitrary rank, in which case each
element is decoded into a `RaggedTensor` with ragged\_rank `input_ragged_rank`
and these are then stacked according to the input shape to output a single
`RaggedTensor` with ragged\_rank `output_ragged_rank`. Each `variant` element in
the input Tensor is decoded by retrieving from the element a 1-D `variant`
Tensor with `input_ragged_rank + 1` Tensors, corresponding to the splits and
values of the decoded `RaggedTensor`. If `input_ragged_rank` is -1, then it is
inferred as `output_ragged_rank` - `rank(encoded_ragged)`. See
`RaggedTensorToVariant` for the corresponding encoding logic.

### `raggedTensorToSparse(rtNestedSplits:rtDenseValues:)`

Converts a `RaggedTensor` into a `SparseTensor` with the same values.

``` swift
@inlinable @inline(__always) public static func raggedTensorToSparse<T: TensorFlowScalar, Tsplits: TensorFlowIndex>(rtNestedSplits: [Tensor<Tsplits>], rtDenseValues: Tensor<T>) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<T>, sparseDenseShape: Tensor<Int64>)
```

input=ragged.from\_nested\_row\_splits(rt\_dense\_values, rt\_nested\_splits)
output=SparseTensor(indices=sparse\_indices, values=sparse\_values,
dense\_shape=sparse\_dense\_shape)

### `raggedTensorToTensor(shape:_:defaultValue:rowPartitionTensors:rowPartitionTypes:)`

Create a dense tensor from a ragged tensor, possibly altering its shape.

``` swift
@inlinable @inline(__always) public static func raggedTensorToTensor<T: TensorFlowScalar, Tindex: TensorFlowIndex, Tshape: TensorFlowIndex>(shape: Tensor<Tshape>, _ values: Tensor<T>, defaultValue: Tensor<T>, rowPartitionTensors: [Tensor<Tindex>], rowPartitionTypes: [String]) -> Tensor<T>
```

The `ragged_to_dense` op creates a dense tensor from a list of row partition
tensors, a value vector, and default values. If the shape is unspecified, the
minimal shape required to contain all the elements in the ragged tensor (the
natural shape) will be used. If some dimensions are left unspecified, then the
size of the natural shape is used in that dimension.

The default\_value will be broadcast to the output shape. After that, the values
from the ragged tensor overwrite the default values. Note that the default\_value
must have less dimensions than the value.

The row partition tensors are in the order of the dimensions.
At present, the types can be:

#### Parameters

  - shape: - shape: Note that dense dimensions cannot be modified by the shape argument. Trying to
    change the size of a dense dimension will cause the op to fail.
    Examples:
    natural shape: \[4, 5, 6\]
    shape: -1
    output shape: \[4, 5, 6\]
    natural shape: \[4, 5, 6\]
    shape: \[3, -1, 2\]
    output shape: \[3, 5, 2\]
    natural shape: \[4, 5, 6\]
    shape: \[3, 7, 2\]
    output shape: \[3, 7, 2\]
  - values: - values: A 1D tensor representing the values of the ragged tensor.

### `raggedTensorToVariant(rtNestedSplits:rtDenseValues:batchedInput:)`

Encodes a `RaggedTensor` into a `variant` Tensor.

``` swift
@inlinable @inline(__always) public static func raggedTensorToVariant<Tvalues: TensorFlowScalar, Tsplits: TensorFlowIndex>(rtNestedSplits: [Tensor<Tsplits>], rtDenseValues: Tensor<Tvalues>, batchedInput: Bool) -> VariantHandle
```

Encodes the given `RaggedTensor` and returns a `variant` Tensor. If
`batched_input` is True, then input `RaggedTensor` is unbatched along the
zero-th dimension, each component `RaggedTensor` is encoded into a scalar
`variant` Tensor, and these are stacked to return a 1-D `variant` Tensor.
If `batched_input` is False, then the input `RaggedTensor` is encoded as is and
a scalar `variant` Tensor is returned. A `RaggedTensor` is encoded by first
creating a 1-D `variant` Tensor with `ragged_rank + 1` elements, containing the
splits and values Tensors of the `RaggedTensor`. Then the 1-D `variant` Tensor
is wrapped in a scalar `variant` Tensor. See `RaggedTensorFromVariant` for the
corresponding decoding logic.

### `randomCrop(image:size:seed:seed2:)`

Randomly crop `image`.

``` swift
@inlinable @inline(__always) public static func randomCrop<T: TensorFlowNumeric>(image: Tensor<T>, size: Tensor<Int64>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<T>
```

`size` is a 1-D int64 tensor with 2 elements representing the crop height and
width.  The values must be non negative.

This Op picks a random location in `image` and crops a `height` by `width`
rectangle from that location.  The random location is picked so the cropped
area will fit inside the original image.

#### Parameters

  - image: - image: 3-D of shape `[height, width, channels]`.
  - size: - size: 1-D of length 2 containing: `crop_height`, `crop_width`..

### `randomDataset(seed:seed2:outputTypes:outputShapes:)`

Creates a Dataset that returns pseudorandom numbers.

``` swift
@inlinable @inline(__always) public static func randomDataset(seed: Tensor<Int64>, seed2: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

Creates a Dataset that returns a stream of uniformly distributed
pseudorandom 64-bit signed integers.

In the TensorFlow Python API, you can instantiate this dataset via the
class `tf.data.experimental.RandomDataset`.

Instances of this dataset are also created as a result of the
`hoist_random_uniform` static optimization. Whether this optimization is
performed is determined by the `experimental_optimization.hoist_random_uniform`
option of `tf.data.Options`.

#### Parameters

  - seed: - seed: A scalar seed for the random number generator. If either seed or seed2 is set to be non-zero, the random number generator is seeded by the given seed.  Otherwise, a random seed is used.
  - seed2: - seed2: A second scalar seed to avoid seed collision.

### `randomGamma(shape:alpha:seed:seed2:)`

Outputs random values from the Gamma distribution(s) described by alpha.

``` swift
@inlinable @inline(__always) public static func randomGamma<S: TensorFlowIndex, T: FloatingPoint & TensorFlowScalar>(shape: Tensor<S>, alpha: Tensor<T>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<T>
```

This op uses the algorithm by Marsaglia et al. to acquire samples via
transformation-rejection from pairs of uniform and normal random variables.
See http://dl.acm.org/citation.cfm?id=358414

#### Parameters

  - shape: - shape: 1-D integer tensor. Shape of independent samples to draw from each distribution described by the shape parameters given in alpha.
  - alpha: - alpha: A tensor in which each scalar is a "shape" parameter describing the associated gamma distribution.

### `randomGammaGrad(alpha:sample:)`

Computes the derivative of a Gamma random sample w.r.t. `alpha`.

``` swift
@inlinable @inline(__always) public static func randomGammaGrad<T: FloatingPoint & TensorFlowScalar>(alpha: Tensor<T>, sample: Tensor<T>) -> Tensor<T>
```

### `randomPoisson(shape:rate:seed:seed2:)`

Use RandomPoissonV2 instead.

``` swift
@inlinable @inline(__always) public static func randomPoisson<S: TensorFlowIndex, Dtype: FloatingPoint & TensorFlowScalar>(shape: Tensor<S>, rate: Tensor<Dtype>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<Dtype>
```

### `randomPoissonV2(shape:rate:seed:seed2:)`

Outputs random values from the Poisson distribution(s) described by rate.

``` swift
@inlinable @inline(__always) public static func randomPoissonV2<S: TensorFlowIndex, R: TensorFlowNumeric, Dtype: TensorFlowNumeric>(shape: Tensor<S>, rate: Tensor<R>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<Dtype>
```

This op uses two algorithms, depending on rate. If rate \>= 10, then
the algorithm by Hormann is used to acquire samples via
transformation-rejection.
See http://www.sciencedirect.com/science/article/pii/0167668793909974.

Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
random variables.
See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
Programming, Volume 2. Addison Wesley

#### Parameters

  - shape: - shape: 1-D integer tensor. Shape of independent samples to draw from each distribution described by the shape parameters given in rate.
  - rate: - rate: A tensor in which each scalar is a "rate" parameter describing the associated poisson distribution.

### `randomShuffle(value:seed:seed2:)`

Randomly shuffles a tensor along its first dimension.

``` swift
@inlinable @inline(__always) public static func randomShuffle<T: TensorFlowScalar>(value: Tensor<T>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<T>
```

The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
to one and only one `output[i]`. For example, a mapping that might occur for a
3x2 tensor is:

``` 
[[1, 2],       [[5, 6],
 [3, 4],  ==>   [1, 2],
 [5, 6]]        [3, 4]]
```

#### Parameters

  - value: - value: The tensor to be shuffled.

### `randomShuffleQueueV2(componentTypes:shapes:capacity:minAfterDequeue:seed:seed2:container:sharedName:)`

A queue that randomizes the order of elements.

``` swift
@inlinable @inline(__always) public static func randomShuffleQueueV2(componentTypes: [TensorDataType], shapes: [TensorShape?], capacity: Int64 = -1, minAfterDequeue: Int64 = 0, seed: Int64 = 0, seed2: Int64 = 0, container: String, sharedName: String) -> ResourceHandle
```

### `randomStandardNormal(shape:seed:seed2:)`

Outputs random values from a normal distribution.

``` swift
@inlinable @inline(__always) public static func randomStandardNormal<Dtype: FloatingPoint & TensorFlowScalar, T: TensorFlowIndex>(shape: Tensor<T>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<Dtype>
```

The generated values will have mean 0 and standard deviation 1.

#### Parameters

  - shape: - shape: The shape of the output tensor.

### `randomUniform(shape:seed:seed2:)`

Outputs random values from a uniform distribution.

``` swift
@inlinable @inline(__always) public static func randomUniform<Dtype: FloatingPoint & TensorFlowScalar, T: TensorFlowIndex>(shape: Tensor<T>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<Dtype>
```

The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

#### Parameters

  - shape: - shape: The shape of the output tensor.

### `randomUniformInt(shape:minval:maxval:seed:seed2:)`

Outputs random integers from a uniform distribution.

``` swift
@inlinable @inline(__always) public static func randomUniformInt<Tout: TensorFlowIndex, T: TensorFlowIndex>(shape: Tensor<T>, minval: Tensor<Tout>, maxval: Tensor<Tout>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<Tout>
```

The generated values are uniform integers in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound
`maxval` is excluded.

The random integers are slightly biased unless `maxval - minval` is an exact
power of two.  The bias is small for values of `maxval - minval` significantly
smaller than the range of the output (either `2^32` or `2^64`).

#### Parameters

  - shape: - shape: The shape of the output tensor.
  - minval: - minval: 0-D.  Inclusive lower bound on the generated integers.
  - maxval: - maxval: 0-D.  Exclusive upper bound on the generated integers.

### `range(start:limit:delta:)`

Creates a sequence of numbers.

``` swift
@inlinable @inline(__always) public static func range<Tidx: TensorFlowNumeric>(start: Tensor<Tidx>, limit: Tensor<Tidx>, delta: Tensor<Tidx>) -> Tensor<Tidx>
```

This operation creates a sequence of numbers that begins at `start` and
extends by increments of `delta` up to but not including `limit`.

For example:

``` 
# 'start' is 3
# 'limit' is 18
# 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
```

#### Parameters

  - start: - start: 0-D (scalar). First entry in the sequence.
  - limit: - limit: 0-D (scalar). Upper limit of sequence, exclusive.
  - delta: - delta: 0-D (scalar). Optional. Default is 1. Number that increments `start`.

### `rangeDataset(start:stop:step:outputTypes:outputShapes:)`

Creates a dataset with a range of values. Corresponds to python's xrange.

``` swift
@inlinable @inline(__always) public static func rangeDataset(start: Tensor<Int64>, stop: Tensor<Int64>, step: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

#### Parameters

  - start: - start: corresponds to start in python's xrange().
  - stop: - stop: corresponds to stop in python's xrange().
  - step: - step: corresponds to step in python's xrange().

### `rank(_:)`

Returns the rank of a tensor.

``` swift
@inlinable @inline(__always) public static func rank<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<Int32>
```

This operation returns an integer representing the rank of `input`.

For example:

``` 
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor 't' is [2, 2, 3]
rank(t) ==> 3
```

**Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
of a tensor is the number of indices required to uniquely select each element
of the tensor. Rank is also known as "order", "degree", or "ndims."

### `readFile(filename:)`

Reads and outputs the entire contents of the input filename.

``` swift
@inlinable @inline(__always) public static func readFile(filename: StringTensor) -> StringTensor
```

### `readVariableOp(resource:)`

Reads the value of a variable.

``` swift
@inlinable @inline(__always) public static func readVariableOp<Dtype: TensorFlowScalar>(resource: ResourceHandle) -> Tensor<Dtype>
```

The tensor returned by this operation is immutable.

The value returned by this operation is guaranteed to be influenced by all the
writes on which this operation depends directly or indirectly, and to not be
influenced by any of the writes which depend directly or indirectly on this
operation.

#### Parameters

  - resource: - resource: handle to the resource in which to store the variable.

### `readerNumRecordsProducedV2(readerHandle:)`

Returns the number of records this Reader has produced.

``` swift
@inlinable @inline(__always) public static func readerNumRecordsProducedV2(readerHandle: ResourceHandle) -> Tensor<Int64>
```

This is the same as the number of ReaderRead executions that have
succeeded.

### `readerNumWorkUnitsCompletedV2(readerHandle:)`

Returns the number of work units this Reader has finished processing.

``` swift
@inlinable @inline(__always) public static func readerNumWorkUnitsCompletedV2(readerHandle: ResourceHandle) -> Tensor<Int64>
```

### `readerReadUpToV2(readerHandle:queueHandle:numRecords:)`

Returns up to `num_records` (key, value) pairs produced by a Reader.

``` swift
@inlinable @inline(__always) public static func readerReadUpToV2(readerHandle: ResourceHandle, queueHandle: ResourceHandle, numRecords: Tensor<Int64>) -> (keys: StringTensor, values: StringTensor)
```

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).
It may return less than `num_records` even before the last batch.

### `readerReadV2(readerHandle:queueHandle:)`

Returns the next record (key, value pair) produced by a Reader.

``` swift
@inlinable @inline(__always) public static func readerReadV2(readerHandle: ResourceHandle, queueHandle: ResourceHandle) -> (key: StringTensor, value: StringTensor)
```

Will dequeue from the input queue if necessary (e.g. when the
Reader needs to start reading from a new file since it has finished
with the previous file).

### `readerResetV2(readerHandle:)`

Restore a Reader to its initial clean state.

``` swift
@inlinable @inline(__always) public static func readerResetV2(readerHandle: ResourceHandle)
```

### `readerRestoreStateV2(readerHandle:state:)`

Restore a reader to a previously saved state.

``` swift
@inlinable @inline(__always) public static func readerRestoreStateV2(readerHandle: ResourceHandle, state: StringTensor)
```

Not all Readers support being restored, so this can produce an
Unimplemented error.

#### Parameters

  - state: - state: Result of a ReaderSerializeState of a Reader with type matching reader\_handle.

### `readerSerializeStateV2(readerHandle:)`

Produce a string tensor that encodes the state of a Reader.

``` swift
@inlinable @inline(__always) public static func readerSerializeStateV2(readerHandle: ResourceHandle) -> StringTensor
```

Not all Readers support being serialized, so this can produce an
Unimplemented error.

### `real(_:)`

Returns the real part of a complex number.

``` swift
@inlinable @inline(__always) public static func real<T: TensorFlowScalar, Tout: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<Tout>
```

Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the real part of each element in `input`. All elements in
`input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
part returned by this operation and *b* is the imaginary part.

For example:

``` 
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.real(input) ==> [-2.25, 3.25]
```

### `realDiv(_:_:)`

Returns x / y element-wise for real types.

``` swift
@inlinable @inline(__always) public static func realDiv<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

If `x` and `y` are reals, this will return the floating-point division.

*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `rebatchDataset(inputDataset:numReplicas:outputTypes:outputShapes:useFallback:)`

Creates a dataset that changes the batch size.

``` swift
@inlinable @inline(__always) public static func rebatchDataset(inputDataset: VariantHandle, numReplicas: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?], useFallback: Bool = true) -> VariantHandle
```

Creates a dataset that changes the batch size of the dataset to current batch
size // num\_workers.

### `reciprocal(_:)`

Computes the reciprocal of x element-wise.

``` swift
@inlinable @inline(__always) public static func reciprocal<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

I.e., \\(y = 1 / x\\).

### `reciprocalGrad(_:dy:)`

Computes the gradient for the inverse of `x` wrt its input.

``` swift
@inlinable @inline(__always) public static func reciprocalGrad<T: FloatingPoint & TensorFlowScalar>(_ y: Tensor<T>, dy: Tensor<T>) -> Tensor<T>
```

Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
is the corresponding input gradient.

### `recordInput(filePattern:fileRandomSeed:fileShuffleShiftRatio:fileBufferSize:fileParallelism:batchSize:compressionType:)`

Emits randomized records.

``` swift
@inlinable @inline(__always) public static func recordInput(filePattern: String, fileRandomSeed: Int64 = 301, fileShuffleShiftRatio: Double = 0, fileBufferSize: Int64 = 10000, fileParallelism: Int64 = 16, batchSize: Int64 = 32, compressionType: String) -> StringTensor
```

### `recv(tensorName:sendDevice:sendDeviceIncarnation:recvDevice:clientTerminated:)`

Receives the named tensor from send\_device on recv\_device.

``` swift
@inlinable @inline(__always) public static func recv<TensorType: TensorFlowScalar>(tensorName: String, sendDevice: String, sendDeviceIncarnation: Int64, recvDevice: String, clientTerminated: Bool = false) -> Tensor<TensorType>
```

### `recvTPUEmbeddingActivations(numOutputs:config:)`

An op that receives embedding activations on the TPU.

``` swift
@inlinable @inline(__always) public static func recvTPUEmbeddingActivations(numOutputs: Int64, config: String) -> [Tensor<Float>]
```

The TPU system performs the embedding lookups and aggregations specified by
the arguments to TPUEmbeddingEnqueue(Integer/Sparse/SparseTensor)Batch. The
results of these aggregations are visible to the Tensorflow Graph as the
outputs of a RecvTPUEmbeddingActivations op. This op returns a list containing
one Tensor of activations per table specified in the model. There can be at
most one RecvTPUEmbeddingActivations op in the TPU graph.

### `reduceDataset(inputDataset:initialState:otherArguments:f:outputShapes:useInterOpParallelism:)`

Reduces the input dataset to a singleton using a reduce function.

``` swift
@inlinable @inline(__always) public static func reduceDataset<FIn: TensorGroup, FOut: TensorGroup, Tstate: TensorArrayProtocol, Targuments: TensorArrayProtocol, OutputTypes: TensorGroup>(inputDataset: VariantHandle, initialState: Tstate, otherArguments: Targuments, f: (FIn) -> FOut, outputShapes: [TensorShape?], useInterOpParallelism: Bool = true) -> OutputTypes
```

### `reduceJoin(inputs:reductionIndices:keepDims:separator:)`

Joins a string Tensor across the given dimensions.

``` swift
@inlinable @inline(__always) public static func reduceJoin(inputs: StringTensor, reductionIndices: Tensor<Int32>, keepDims: Bool = false, separator: String) -> StringTensor
```

Computes the string join across dimensions in the given string Tensor of shape
`[\\(d_0, d_1, ..., d_{n-1}\\)]`.  Returns a new Tensor created by joining the input
strings with the given separator (default: empty string).  Negative indices are
counted backwards from the end, with `-1` being equivalent to `n - 1`.  If
indices are not specified, joins across all dimensions beginning from `n - 1`
through `0`.

For example:

``` python
# tensor `a` is [["a", "b"], ["c", "d"]]
tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
tf.reduce_join(a, [0, 1]) ==> "acbd"
tf.reduce_join(a, [1, 0]) ==> "abcd"
tf.reduce_join(a, []) ==> [["a", "b"], ["c", "d"]]
tf.reduce_join(a) = tf.reduce_join(a, [1, 0]) ==> "abcd"
```

#### Parameters

  - inputs: - inputs: The input to be joined.  All reduced indices must have non-zero size.

### `regexFullMatch(_:pattern:)`

Check if the input matches the regex pattern.

``` swift
@inlinable @inline(__always) public static func regexFullMatch(_ input: StringTensor, pattern: StringTensor) -> Tensor<Bool>
```

The input is a string tensor of any shape. The pattern is a scalar
string tensor which is applied to every element of the input tensor.
The boolean values (True or False) of the output tensor indicate
if the input matches the regex pattern provided.

The pattern follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)

#### Parameters

  - input: - input: A string tensor of the text to be processed.
  - pattern: - pattern: A scalar string tensor containing the regular expression to match the input.

### `regexReplace(_:pattern:rewrite:replaceGlobal:)`

Replaces matches of the `pattern` regular expression in `input` with the
replacement string provided in `rewrite`.

``` swift
@inlinable @inline(__always) public static func regexReplace(_ input: StringTensor, pattern: StringTensor, rewrite: StringTensor, replaceGlobal: Bool = true) -> StringTensor
```

It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)

#### Parameters

  - input: - input: The text to be processed.
  - pattern: - pattern: The regular expression to be matched in the `input` strings.
  - rewrite: - rewrite: The rewrite string to be substituted for the `pattern` expression where it is matched in the `input` strings.

### `relu(features:)`

Computes rectified linear: `max(features, 0)`.

``` swift
@inlinable @inline(__always) public static func relu<T: TensorFlowNumeric>(features: Tensor<T>) -> Tensor<T>
```

See: https://en.wikipedia.org/wiki/Rectifier\_(neural\_networks)
Example usage:

> > > tf.nn.relu(\[-2., 0., -0., 3.\]).numpy()
> > > array(\[ 0.,  0., -0.,  3.\], dtype=float32)

### `relu6(features:)`

Computes rectified linear 6: `min(max(features, 0), 6)`.

``` swift
@inlinable @inline(__always) public static func relu6<T: TensorFlowNumeric>(features: Tensor<T>) -> Tensor<T>
```

### `relu6Grad(gradients:features:)`

Computes rectified linear 6 gradients for a Relu6 operation.

``` swift
@inlinable @inline(__always) public static func relu6Grad<T: TensorFlowNumeric>(gradients: Tensor<T>, features: Tensor<T>) -> Tensor<T>
```

#### Parameters

  - gradients: - gradients: The backpropagated gradients to the corresponding Relu6 operation.
  - features: - features: The features passed as input to the corresponding Relu6 operation, or its output; using either one produces the same result.

### `reluGrad(gradients:features:)`

Computes rectified linear gradients for a Relu operation.

``` swift
@inlinable @inline(__always) public static func reluGrad<T: TensorFlowNumeric>(gradients: Tensor<T>, features: Tensor<T>) -> Tensor<T>
```

#### Parameters

  - gradients: - gradients: The backpropagated gradients to the corresponding Relu operation.
  - features: - features: The features passed as input to the corresponding Relu operation, OR the outputs of that operation (both work equivalently).

### `remoteCall(target:args:f:)`

Runs function `f` on a remote device indicated by `target`.

``` swift
@inlinable @inline(__always) public static func remoteCall<Tin: TensorArrayProtocol, Tout: TensorGroup, FIn: TensorGroup, FOut: TensorGroup>(target: StringTensor, args: Tin, f: (FIn) -> FOut) -> Tout
```

#### Parameters

  - target: - target: A fully specified device name where we want to run the function.
  - args: - args: A list of arguments for the function.

### `remoteFusedGraphExecute(inputs:serializedRemoteFusedGraphExecuteInfo:)`

Execute a sub graph on a remote processor.

``` swift
@inlinable @inline(__always) public static func remoteFusedGraphExecute<Tinputs: TensorArrayProtocol, Toutputs: TensorGroup>(inputs: Tinputs, serializedRemoteFusedGraphExecuteInfo: String) -> Toutputs
```

The graph specifications(such as graph itself, input tensors and output names)
are stored as a serialized protocol buffer of RemoteFusedGraphExecuteInfo
as serialized\_remote\_fused\_graph\_execute\_info.
The specifications will be passed to a dedicated registered
remote fused graph executor.  The executor will send the graph specifications
to a remote processor and execute that graph.  The execution results
will be passed to consumer nodes as outputs of this node.

#### Parameters

  - inputs: - inputs: Arbitrary number of tensors with arbitrary data types

### `repeatDataset(inputDataset:count:outputTypes:outputShapes:)`

Creates a dataset that emits the outputs of `input_dataset` `count` times.

``` swift
@inlinable @inline(__always) public static func repeatDataset(inputDataset: VariantHandle, count: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

#### Parameters

  - count: - count: A scalar representing the number of times that `input_dataset` should be repeated. A value of `-1` indicates that it should be repeated infinitely.

### `requantizationRange(_:inputMin:inputMax:)`

Computes a range that covers the actual values present in a quantized tensor.

``` swift
@inlinable @inline(__always) public static func requantizationRange<Tinput: TensorFlowScalar>(_ input: Tensor<Tinput>, inputMin: Tensor<Float>, inputMax: Tensor<Float>) -> (outputMin: Tensor<Float>, outputMax: Tensor<Float>)
```

Given a quantized tensor described by `(input, input_min, input_max)`, outputs a
range that covers the actual values present in that tensor. This op is typically
used to produce the `requested_output_min` and `requested_output_max` for
`Requantize`.

### `requantizationRangePerChannel(_:inputMin:inputMax:clipValueMax:)`

Computes requantization range per channel.

``` swift
@inlinable @inline(__always) public static func requantizationRangePerChannel<T: TensorFlowScalar>(_ input: Tensor<T>, inputMin: Tensor<Float>, inputMax: Tensor<Float>, clipValueMax: Double) -> (outputMin: Tensor<Float>, outputMax: Tensor<Float>)
```

#### Parameters

  - input: - input: The original input tensor.

### `requantize(_:inputMin:inputMax:requestedOutputMin:requestedOutputMax:)`

Converts the quantized `input` tensor into a lower-precision `output`.

``` swift
@inlinable @inline(__always) public static func requantize<Tinput: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<Tinput>, inputMin: Tensor<Float>, inputMax: Tensor<Float>, requestedOutputMin: Tensor<Float>, requestedOutputMax: Tensor<Float>) -> (output: Tensor<OutType>, outputMin: Tensor<Float>, outputMax: Tensor<Float>)
```

Converts the quantized `input` tensor into a lower-precision `output`, using the
output range specified with `requested_output_min` and `requested_output_max`.

`[input_min, input_max]` are scalar floats that specify the range for the float
interpretation of the `input` data. For example, if `input_min` is -1.0f and
`input_max` is 1.0f, and we are dealing with `quint16` quantized data, then a 0
value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

### `requantizePerChannel(_:inputMin:inputMax:requestedOutputMin:requestedOutputMax:)`

Requantizes input with min and max values known per channel.

``` swift
@inlinable @inline(__always) public static func requantizePerChannel<T: TensorFlowScalar, OutType: TensorFlowScalar>(_ input: Tensor<T>, inputMin: Tensor<Float>, inputMax: Tensor<Float>, requestedOutputMin: Tensor<Float>, requestedOutputMax: Tensor<Float>) -> (output: Tensor<OutType>, outputMin: Tensor<Float>, outputMax: Tensor<Float>)
```

#### Parameters

  - input: - input: The original input tensor.

### `requiresOlderGraphVersion()`

``` swift
@inlinable @inline(__always) public static func requiresOlderGraphVersion() -> Tensor<Int32>
```

### `reservedAttr(range:)`

``` swift
@inlinable @inline(__always) public static func reservedAttr(range: Int64)
```

### `reservedInput(_:)`

``` swift
@inlinable @inline(__always) public static func reservedInput(_ input: Tensor<Int32>)
```

### `reshape(_:shape:)`

Reshapes a tensor.

``` swift
@inlinable @inline(__always) public static func reshape<T: TensorFlowScalar, Tshape: TensorFlowIndex>(_ tensor: Tensor<T>, shape: Tensor<Tshape>) -> Tensor<T>
```

Given `tensor`, this operation returns a tensor that has the same values
as `tensor` with shape `shape`.

If one component of 1-D tensor `shape` is the special value -1, the size of that
dimension is computed so that the total size remains constant.  In particular, a
`shape` of `[-1]` flattens into 1-D.  At most one component of `shape` may be
unknown.

The `shape` must be 1-D and the operation returns a tensor with shape
`shape` filled with the values of `tensor`. In this case, the number of elements
implied by `shape` must be the same as the number of elements in `tensor`.

It is an error if `shape` is not 1-D.

For example:

``` 
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]
 
# tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                        [3, 3, 4, 4]]
 
# tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]]
# tensor 't' has shape [3, 2, 3]
# pass '[-1]' to flatten 't'
reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
 
# -1 can also be used to infer the shape
 
# -1 is inferred to be 9:
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 2:
reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 3:
reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]],
                             [[4, 4, 4],
                              [5, 5, 5],
                              [6, 6, 6]]]
 
# tensor 't' is [7]
# shape `[]` reshapes to a scalar
reshape(t, []) ==> 7
```

#### Parameters

  - shape: - shape: Defines the shape of the output tensor.

### `resizeArea(images:size:alignCorners:)`

Resize `images` to `size` using area interpolation.

``` swift
@inlinable @inline(__always) public static func resizeArea<T: TensorFlowNumeric>(images: Tensor<T>, size: Tensor<Int32>, alignCorners: Bool = false) -> Tensor<Float>
```

Input images can be of different types but output images are always float.

The range of pixel values for the output image might be slightly different
from the range for the input image because of limited numerical precision.
To guarantee an output range, for example `[0.0, 1.0]`, apply
`tf.clip_by_value` to the output.

Each output pixel is computed by first transforming the pixel's footprint into
the input tensor and then averaging the pixels that intersect the footprint. An
input pixel's contribution to the average is weighted by the fraction of its
area that intersects the footprint.  This is the same as OpenCV's INTER\_AREA.

#### Parameters

  - images: - images: 4-D with shape `[batch, height, width, channels]`.
  - size: - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new size for the images.

### `resizeBicubic(images:size:alignCorners:halfPixelCenters:)`

Resize `images` to `size` using bicubic interpolation.

``` swift
@inlinable @inline(__always) public static func resizeBicubic<T: TensorFlowNumeric>(images: Tensor<T>, size: Tensor<Int32>, alignCorners: Bool = false, halfPixelCenters: Bool = false) -> Tensor<Float>
```

Input images can be of different types but output images are always float.

#### Parameters

  - images: - images: 4-D with shape `[batch, height, width, channels]`.
  - size: - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new size for the images.

### `resizeBicubicGrad(grads:originalImage:alignCorners:halfPixelCenters:)`

Computes the gradient of bicubic interpolation.

``` swift
@inlinable @inline(__always) public static func resizeBicubicGrad<T: FloatingPoint & TensorFlowScalar>(grads: Tensor<Float>, originalImage: Tensor<T>, alignCorners: Bool = false, halfPixelCenters: Bool = false) -> Tensor<T>
```

#### Parameters

  - grads: - grads: 4-D with shape `[batch, height, width, channels]`.

### `resizeBilinear(images:size:alignCorners:halfPixelCenters:)`

Resize `images` to `size` using bilinear interpolation.

``` swift
@inlinable @inline(__always) public static func resizeBilinear<T: TensorFlowNumeric>(images: Tensor<T>, size: Tensor<Int32>, alignCorners: Bool = false, halfPixelCenters: Bool = false) -> Tensor<Float>
```

Input images can be of different types but output images are always float.

#### Parameters

  - images: - images: 4-D with shape `[batch, height, width, channels]`.
  - size: - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new size for the images.

### `resizeBilinearGrad(grads:originalImage:alignCorners:halfPixelCenters:)`

Computes the gradient of bilinear interpolation.

``` swift
@inlinable @inline(__always) public static func resizeBilinearGrad<T: FloatingPoint & TensorFlowScalar>(grads: Tensor<Float>, originalImage: Tensor<T>, alignCorners: Bool = false, halfPixelCenters: Bool = false) -> Tensor<T>
```

#### Parameters

  - grads: - grads: 4-D with shape `[batch, height, width, channels]`.

### `resizeNearestNeighbor(images:size:alignCorners:halfPixelCenters:)`

Resize `images` to `size` using nearest neighbor interpolation.

``` swift
@inlinable @inline(__always) public static func resizeNearestNeighbor<T: TensorFlowNumeric>(images: Tensor<T>, size: Tensor<Int32>, alignCorners: Bool = false, halfPixelCenters: Bool = false) -> Tensor<T>
```

#### Parameters

  - images: - images: 4-D with shape `[batch, height, width, channels]`.
  - size: - size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new size for the images.

### `resizeNearestNeighborGrad(grads:size:alignCorners:halfPixelCenters:)`

Computes the gradient of nearest neighbor interpolation.

``` swift
@inlinable @inline(__always) public static func resizeNearestNeighborGrad<T: TensorFlowNumeric>(grads: Tensor<T>, size: Tensor<Int32>, alignCorners: Bool = false, halfPixelCenters: Bool = false) -> Tensor<T>
```

#### Parameters

  - grads: - grads: 4-D with shape `[batch, height, width, channels]`.
  - size: - size: = A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The original input size.

### `resourceAccumulatorApplyGradient(handle:localStep:gradient:)`

Applies a gradient to a given accumulator.

``` swift
@inlinable @inline(__always) public static func resourceAccumulatorApplyGradient<Dtype: TensorFlowNumeric>(handle: ResourceHandle, localStep: Tensor<Int64>, gradient: Tensor<Dtype>)
```

Does not add if local\_step is lesser than the accumulator's global\_step.

#### Parameters

  - handle: - handle: The handle to a accumulator.
  - gradient: - gradient: A tensor of the gradient to be accumulated.

### `resourceAccumulatorNumAccumulated(handle:)`

Returns the number of gradients aggregated in the given accumulators.

``` swift
@inlinable @inline(__always) public static func resourceAccumulatorNumAccumulated(handle: ResourceHandle) -> Tensor<Int32>
```

#### Parameters

  - handle: - handle: The handle to an accumulator.

### `resourceAccumulatorSetGlobalStep(handle:newGlobalStep:)`

Updates the accumulator with a new value for global\_step.

``` swift
@inlinable @inline(__always) public static func resourceAccumulatorSetGlobalStep(handle: ResourceHandle, newGlobalStep: Tensor<Int64>)
```

Logs warning if the accumulator's value is already higher than
new\_global\_step.

#### Parameters

  - handle: - handle: The handle to an accumulator.

### `resourceAccumulatorTakeGradient(handle:numRequired:)`

Extracts the average gradient in the given ConditionalAccumulator.

``` swift
@inlinable @inline(__always) public static func resourceAccumulatorTakeGradient<Dtype: TensorFlowNumeric>(handle: ResourceHandle, numRequired: Tensor<Int32>) -> Tensor<Dtype>
```

The op blocks until sufficient (i.e., more than num\_required)
gradients have been accumulated.  If the accumulator has already
aggregated more than num\_required gradients, it returns the average of
the accumulated gradients.  Also automatically increments the recorded
global\_step in the accumulator by 1, and resets the aggregate to 0.

#### Parameters

  - handle: - handle: The handle to an accumulator.

### `resourceApplyAdaMax(var_:m:v:beta1Power:lr:beta1:beta2:epsilon:grad:useLocking:)`

Update '\*var' according to the AdaMax algorithm.

``` swift
@inlinable @inline(__always) public static func resourceApplyAdaMax<T: TensorFlowNumeric>(var_: ResourceHandle, m: ResourceHandle, v: ResourceHandle, beta1Power: Tensor<T>, lr: Tensor<T>, beta1: Tensor<T>, beta2: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false)
```

m\_t \<- beta1 \* m\_{t-1} + (1 - beta1) \* g
v\_t \<- max(beta2 \* v\_{t-1}, abs(g))
variable \<- variable - learning\_rate / (1 - beta1^t) \* m\_t / (v\_t + epsilon)

#### Parameters

  - var: - var: Should be from a Variable().
  - m: - m: Should be from a Variable().
  - v: - v: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - beta1: - beta1: Momentum factor. Must be a scalar.
  - beta2: - beta2: Momentum factor. Must be a scalar.
  - epsilon: - epsilon: Ridge term. Must be a scalar.
  - grad: - grad: The gradient.

### `resourceApplyAdadelta(var_:accum:accumUpdate:lr:rho:epsilon:grad:useLocking:)`

Update '\*var' according to the adadelta scheme.

``` swift
@inlinable @inline(__always) public static func resourceApplyAdadelta<T: TensorFlowNumeric>(var_: ResourceHandle, accum: ResourceHandle, accumUpdate: ResourceHandle, lr: Tensor<T>, rho: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false)
```

accum = rho() \* accum + (1 - rho()) \* grad.square();
update = (update\_accum + epsilon).sqrt() \* (accum + epsilon()).rsqrt() \* grad;
update\_accum = rho() \* update\_accum + (1 - rho()) \* update.square();
var -= update;

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - rho: - rho: Decay factor. Must be a scalar.
  - epsilon: - epsilon: Constant factor. Must be a scalar.
  - grad: - grad: The gradient.

### `resourceApplyAdagrad(var_:accum:lr:grad:useLocking:updateSlots:)`

Update '\*var' according to the adagrad scheme.

``` swift
@inlinable @inline(__always) public static func resourceApplyAdagrad<T: TensorFlowNumeric>(var_: ResourceHandle, accum: ResourceHandle, lr: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false, updateSlots: Bool = true)
```

accum += grad \* grad
var -= lr \* grad \* (1 / sqrt(accum))

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - grad: - grad: The gradient.

### `resourceApplyAdagradDA(var_:gradientAccumulator:gradientSquaredAccumulator:grad:lr:l1:l2:globalStep:useLocking:)`

Update '\*var' according to the proximal adagrad scheme.

``` swift
@inlinable @inline(__always) public static func resourceApplyAdagradDA<T: TensorFlowNumeric>(var_: ResourceHandle, gradientAccumulator: ResourceHandle, gradientSquaredAccumulator: ResourceHandle, grad: Tensor<T>, lr: Tensor<T>, l1: Tensor<T>, l2: Tensor<T>, globalStep: Tensor<Int64>, useLocking: Bool = false)
```

#### Parameters

  - var: - var: Should be from a Variable().
  - grad: - grad: The gradient.
  - lr: - lr: Scaling factor. Must be a scalar.
  - l1: - l1: L1 regularization. Must be a scalar.
  - l2: - l2: L2 regularization. Must be a scalar.

### `resourceApplyAdagradV2(var_:accum:lr:epsilon:grad:useLocking:updateSlots:)`

Update '\*var' according to the adagrad scheme.

``` swift
@inlinable @inline(__always) public static func resourceApplyAdagradV2<T: TensorFlowNumeric>(var_: ResourceHandle, accum: ResourceHandle, lr: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false, updateSlots: Bool = true)
```

accum += grad \* grad
var -= lr \* grad \* (1 / sqrt(accum))

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - epsilon: - epsilon: Constant factor. Must be a scalar.
  - grad: - grad: The gradient.

### `resourceApplyAdam(var_:m:v:beta1Power:beta2Power:lr:beta1:beta2:epsilon:grad:useLocking:useNesterov:)`

Update '\*var' according to the Adam algorithm.

``` swift
@inlinable @inline(__always) public static func resourceApplyAdam<T: TensorFlowNumeric>(var_: ResourceHandle, m: ResourceHandle, v: ResourceHandle, beta1Power: Tensor<T>, beta2Power: Tensor<T>, lr: Tensor<T>, beta1: Tensor<T>, beta2: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false, useNesterov: Bool = false)
```

$$\\text{lr}*t := \\mathrm{learning\_rate} \* \\sqrt{1 - \\beta\_2^t} / (1 - \\beta\_1^t)$$
$$m\_t := \\beta\_1 \* m*{t-1} + (1 - \\beta\_1) \* g$$
$$v\_t := \\beta\_2 \* v\_{t-1} + (1 - \\beta\_2) \* g \* g$$
$$\\text{variable} := \\text{variable} - \\text{lr}\_t \* m\_t / (\\sqrt{v\_t} + \\epsilon)$$

#### Parameters

  - var: - var: Should be from a Variable().
  - m: - m: Should be from a Variable().
  - v: - v: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - beta1: - beta1: Momentum factor. Must be a scalar.
  - beta2: - beta2: Momentum factor. Must be a scalar.
  - epsilon: - epsilon: Ridge term. Must be a scalar.
  - grad: - grad: The gradient.

### `resourceApplyAdamWithAmsgrad(var_:m:v:vhat:beta1Power:beta2Power:lr:beta1:beta2:epsilon:grad:useLocking:)`

Update '\*var' according to the Adam algorithm.

``` swift
@inlinable @inline(__always) public static func resourceApplyAdamWithAmsgrad<T: TensorFlowNumeric>(var_: ResourceHandle, m: ResourceHandle, v: ResourceHandle, vhat: ResourceHandle, beta1Power: Tensor<T>, beta2Power: Tensor<T>, lr: Tensor<T>, beta1: Tensor<T>, beta2: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false)
```

$$\\text{lr}*t := \\mathrm{learning\_rate} \* \\sqrt{1 - \\beta\_2^t} / (1 - \\beta\_1^t)$$
$$m\_t := \\beta\_1 \* m*{t-1} + (1 - \\beta\_1) \* g$$
$$v\_t := \\beta\_2 \* v\_{t-1} + (1 - \\beta\_2) \* g \* g$$
$$\\hat{v}*t := max{\\hat{v}*{t-1}, v\_t}$$
$$\\text{variable} := \\text{variable} - \\text{lr}\_t \* m\_t / (\\sqrt{\\hat{v}\_t} + \\epsilon)$$

#### Parameters

  - var: - var: Should be from a Variable().
  - m: - m: Should be from a Variable().
  - v: - v: Should be from a Variable().
  - vhat: - vhat: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - beta1: - beta1: Momentum factor. Must be a scalar.
  - beta2: - beta2: Momentum factor. Must be a scalar.
  - epsilon: - epsilon: Ridge term. Must be a scalar.
  - grad: - grad: The gradient.

### `resourceApplyAddSign(var_:m:lr:alpha:signDecay:beta:grad:useLocking:)`

Update '\*var' according to the AddSign update.

``` swift
@inlinable @inline(__always) public static func resourceApplyAddSign<T: TensorFlowNumeric>(var_: ResourceHandle, m: ResourceHandle, lr: Tensor<T>, alpha: Tensor<T>, signDecay: Tensor<T>, beta: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false)
```

m\_t \<- beta1 \* m\_{t-1} + (1 - beta1) \* g
update \<- (alpha + sign\_decay \* sign(g) \*sign(m)) \* g
variable \<- variable - lr\_t \* update

#### Parameters

  - var: - var: Should be from a Variable().
  - m: - m: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - alpha: - alpha: Must be a scalar.
  - beta: - beta: Must be a scalar.
  - grad: - grad: The gradient.

### `resourceApplyCenteredRMSProp(var_:mg:ms:mom:lr:rho:momentum:epsilon:grad:useLocking:)`

Update '\*var' according to the centered RMSProp algorithm.

``` swift
@inlinable @inline(__always) public static func resourceApplyCenteredRMSProp<T: TensorFlowNumeric>(var_: ResourceHandle, mg: ResourceHandle, ms: ResourceHandle, mom: ResourceHandle, lr: Tensor<T>, rho: Tensor<T>, momentum: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false)
```

The centered RMSProp algorithm uses an estimate of the centered second moment
(i.e., the variance) for normalization, as opposed to regular RMSProp, which
uses the (uncentered) second moment. This often helps with training, but is
slightly more expensive in terms of computation and memory.

Note that in dense implementation of this algorithm, mg, ms, and mom will
update even if the grad is zero, but in this sparse implementation, mg, ms,
and mom will not update in iterations during which the grad is zero.

mean\_square = decay \* mean\_square + (1-decay) \* gradient \*\* 2
mean\_grad = decay \* mean\_grad + (1-decay) \* gradient

Delta = learning\_rate \* gradient / sqrt(mean\_square + epsilon - mean\_grad \*\* 2)

mg \<- rho \* mg\_{t-1} + (1-rho) \* grad
ms \<- rho \* ms\_{t-1} + (1-rho) \* grad \* grad
mom \<- momentum \* mom\_{t-1} + lr \* grad / sqrt(ms - mg \* mg + epsilon)
var \<- var - mom

#### Parameters

  - var: - var: Should be from a Variable().
  - mg: - mg: Should be from a Variable().
  - ms: - ms: Should be from a Variable().
  - mom: - mom: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - rho: - rho: Decay rate. Must be a scalar.
  - epsilon: - epsilon: Ridge term. Must be a scalar.
  - grad: - grad: The gradient.

### `resourceApplyFtrl(var_:accum:linear:grad:lr:l1:l2:lrPower:useLocking:)`

Update '\*var' according to the Ftrl-proximal scheme.

``` swift
@inlinable @inline(__always) public static func resourceApplyFtrl<T: TensorFlowNumeric>(var_: ResourceHandle, accum: ResourceHandle, linear: ResourceHandle, grad: Tensor<T>, lr: Tensor<T>, l1: Tensor<T>, l2: Tensor<T>, lrPower: Tensor<T>, useLocking: Bool = false)
```

accum\_new = accum + grad \* grad
linear += grad - (accum\_new^(-lr\_power) - accum^(-lr\_power)) / lr \* var
quadratic = 1.0 / (accum\_new^(lr\_power) \* lr) + 2 \* l2
var = (sign(linear) \* l1 - linear) / quadratic if |linear| \> l1 else 0.0
accum = accum\_new

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - linear: - linear: Should be from a Variable().
  - grad: - grad: The gradient.
  - lr: - lr: Scaling factor. Must be a scalar.
  - l1: - l1: L1 regulariation. Must be a scalar.
  - l2: - l2: L2 regulariation. Must be a scalar.

### `resourceApplyFtrlV2(var_:accum:linear:grad:lr:l1:l2:l2Shrinkage:lrPower:useLocking:)`

Update '\*var' according to the Ftrl-proximal scheme.

``` swift
@inlinable @inline(__always) public static func resourceApplyFtrlV2<T: TensorFlowNumeric>(var_: ResourceHandle, accum: ResourceHandle, linear: ResourceHandle, grad: Tensor<T>, lr: Tensor<T>, l1: Tensor<T>, l2: Tensor<T>, l2Shrinkage: Tensor<T>, lrPower: Tensor<T>, useLocking: Bool = false)
```

grad\_with\_shrinkage = grad + 2 \* l2\_shrinkage \* var
accum\_new = accum + grad\_with\_shrinkage \* grad\_with\_shrinkage
linear += grad\_with\_shrinkage +
(accum\_new^(-lr\_power) - accum^(-lr\_power)) / lr \* var
quadratic = 1.0 / (accum\_new^(lr\_power) \* lr) + 2 \* l2
var = (sign(linear) \* l1 - linear) / quadratic if |linear| \> l1 else 0.0
accum = accum\_new

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - linear: - linear: Should be from a Variable().
  - grad: - grad: The gradient.
  - lr: - lr: Scaling factor. Must be a scalar.
  - l1: - l1: L1 regulariation. Must be a scalar.
  - l2: - l2: L2 shrinkage regulariation. Must be a scalar.

### `resourceApplyGradientDescent(var_:alpha:delta:useLocking:)`

Update '\*var' by subtracting 'alpha' \* 'delta' from it.

``` swift
@inlinable @inline(__always) public static func resourceApplyGradientDescent<T: TensorFlowNumeric>(var_: ResourceHandle, alpha: Tensor<T>, delta: Tensor<T>, useLocking: Bool = false)
```

#### Parameters

  - var: - var: Should be from a Variable().
  - alpha: - alpha: Scaling factor. Must be a scalar.
  - delta: - delta: The change.

### `resourceApplyKerasMomentum(var_:accum:lr:grad:momentum:useLocking:useNesterov:)`

Update '\*var' according to the momentum scheme.

``` swift
@inlinable @inline(__always) public static func resourceApplyKerasMomentum<T: TensorFlowNumeric>(var_: ResourceHandle, accum: ResourceHandle, lr: Tensor<T>, grad: Tensor<T>, momentum: Tensor<T>, useLocking: Bool = false, useNesterov: Bool = false)
```

Set use\_nesterov = True if you want to use Nesterov momentum.

accum = accum \* momentum - lr \* grad
var += accum

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - grad: - grad: The gradient.
  - momentum: - momentum: Momentum. Must be a scalar.

### `resourceApplyMomentum(var_:accum:lr:grad:momentum:useLocking:useNesterov:)`

Update '\*var' according to the momentum scheme. Set use\_nesterov = True if you

``` swift
@inlinable @inline(__always) public static func resourceApplyMomentum<T: TensorFlowNumeric>(var_: ResourceHandle, accum: ResourceHandle, lr: Tensor<T>, grad: Tensor<T>, momentum: Tensor<T>, useLocking: Bool = false, useNesterov: Bool = false)
```

want to use Nesterov momentum.

accum = accum \* momentum + grad
var -= lr \* accum

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - grad: - grad: The gradient.
  - momentum: - momentum: Momentum. Must be a scalar.

### `resourceApplyPowerSign(var_:m:lr:logbase:signDecay:beta:grad:useLocking:)`

Update '\*var' according to the AddSign update.

``` swift
@inlinable @inline(__always) public static func resourceApplyPowerSign<T: TensorFlowNumeric>(var_: ResourceHandle, m: ResourceHandle, lr: Tensor<T>, logbase: Tensor<T>, signDecay: Tensor<T>, beta: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false)
```

m\_t \<- beta1 \* m\_{t-1} + (1 - beta1) \* g
update \<- exp(logbase \* sign\_decay \* sign(g) \* sign(m\_t)) \* g
variable \<- variable - lr\_t \* update

#### Parameters

  - var: - var: Should be from a Variable().
  - m: - m: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - logbase: - logbase: Must be a scalar.
  - beta: - beta: Must be a scalar.
  - grad: - grad: The gradient.

### `resourceApplyProximalAdagrad(var_:accum:lr:l1:l2:grad:useLocking:)`

Update '\*var' and '\*accum' according to FOBOS with Adagrad learning rate.

``` swift
@inlinable @inline(__always) public static func resourceApplyProximalAdagrad<T: TensorFlowNumeric>(var_: ResourceHandle, accum: ResourceHandle, lr: Tensor<T>, l1: Tensor<T>, l2: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false)
```

accum += grad \* grad
prox\_v = var - lr \* grad \* (1 / sqrt(accum))
var = sign(prox\_v)/(1+lr*l2) \* max{|prox\_v|-lr*l1,0}

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - l1: - l1: L1 regularization. Must be a scalar.
  - l2: - l2: L2 regularization. Must be a scalar.
  - grad: - grad: The gradient.

### `resourceApplyProximalGradientDescent(var_:alpha:l1:l2:delta:useLocking:)`

Update '\*var' as FOBOS algorithm with fixed learning rate.

``` swift
@inlinable @inline(__always) public static func resourceApplyProximalGradientDescent<T: TensorFlowNumeric>(var_: ResourceHandle, alpha: Tensor<T>, l1: Tensor<T>, l2: Tensor<T>, delta: Tensor<T>, useLocking: Bool = false)
```

prox\_v = var - alpha \* delta
var = sign(prox\_v)/(1+alpha*l2) \* max{|prox\_v|-alpha*l1,0}

#### Parameters

  - var: - var: Should be from a Variable().
  - alpha: - alpha: Scaling factor. Must be a scalar.
  - l1: - l1: L1 regularization. Must be a scalar.
  - l2: - l2: L2 regularization. Must be a scalar.
  - delta: - delta: The change.

### `resourceApplyRMSProp(var_:ms:mom:lr:rho:momentum:epsilon:grad:useLocking:)`

Update '\*var' according to the RMSProp algorithm.

``` swift
@inlinable @inline(__always) public static func resourceApplyRMSProp<T: TensorFlowNumeric>(var_: ResourceHandle, ms: ResourceHandle, mom: ResourceHandle, lr: Tensor<T>, rho: Tensor<T>, momentum: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, useLocking: Bool = false)
```

Note that in dense implementation of this algorithm, ms and mom will
update even if the grad is zero, but in this sparse implementation, ms
and mom will not update in iterations during which the grad is zero.

mean\_square = decay \* mean\_square + (1-decay) \* gradient \*\* 2
Delta = learning\_rate \* gradient / sqrt(mean\_square + epsilon)

ms \<- rho \* ms\_{t-1} + (1-rho) \* grad \* grad
mom \<- momentum \* mom\_{t-1} + lr \* grad / sqrt(ms + epsilon)
var \<- var - mom

#### Parameters

  - var: - var: Should be from a Variable().
  - ms: - ms: Should be from a Variable().
  - mom: - mom: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - rho: - rho: Decay rate. Must be a scalar.
  - epsilon: - epsilon: Ridge term. Must be a scalar.
  - grad: - grad: The gradient.

### `resourceConditionalAccumulator(dtype:shape:container:sharedName:reductionType:)`

A conditional accumulator for aggregating gradients.

``` swift
@inlinable @inline(__always) public static func resourceConditionalAccumulator(dtype: TensorDataType, shape: TensorShape?, container: String, sharedName: String, reductionType: ReductionType = .mean) -> ResourceHandle
```

The accumulator accepts gradients marked with local\_step greater or
equal to the most recent global\_step known to the accumulator. The
average can be extracted from the accumulator, provided sufficient
gradients have been accumulated. Extracting the average automatically
resets the aggregate to 0, and increments the global\_step recorded by
the accumulator.
This is a resource version of ConditionalAccumulator that will work in TF2.0
with tf.cond version 2.

### `resourceCountUpTo(resource:limit:)`

Increments variable pointed to by 'resource' until it reaches 'limit'.

``` swift
@inlinable @inline(__always) public static func resourceCountUpTo<T: TensorFlowIndex>(resource: ResourceHandle, limit: Int64) -> Tensor<T>
```

#### Parameters

  - resource: - resource: Should be from a scalar `Variable` node.

### `resourceCreateOp(resource:)`

``` swift
@inlinable @inline(__always) public static func resourceCreateOp(resource: ResourceHandle)
```

### `resourceGather(resource:indices:batchDims:validateIndices:)`

Gather slices from the variable pointed to by `resource` according to `indices`.

``` swift
@inlinable @inline(__always) public static func resourceGather<Dtype: TensorFlowScalar, Tindices: TensorFlowIndex>(resource: ResourceHandle, indices: Tensor<Tindices>, batchDims: Int64 = 0, validateIndices: Bool = true) -> Tensor<Dtype>
```

`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

``` python
    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]
 
    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]
 
    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
```

### `resourceGatherNd(resource:indices:)`

``` swift
@inlinable @inline(__always) public static func resourceGatherNd<Dtype: TensorFlowScalar, Tindices: TensorFlowIndex>(resource: ResourceHandle, indices: Tensor<Tindices>) -> Tensor<Dtype>
```

### `resourceInitializedOp(resource:)`

``` swift
@inlinable @inline(__always) public static func resourceInitializedOp(resource: ResourceHandle) -> Tensor<Bool>
```

### `resourceScatterAdd(resource:indices:updates:)`

Adds sparse updates to the variable referenced by `resource`.

``` swift
@inlinable @inline(__always) public static func resourceScatterAdd<Dtype: TensorFlowNumeric, Tindices: TensorFlowIndex>(resource: ResourceHandle, indices: Tensor<Tindices>, updates: Tensor<Dtype>)
```

This operation computes

``` 
# Scalar indices
ref[indices, ...] += updates[...]

# Vector indices (for each i)
ref[indices[i], ...] += updates[i, ...]

# High rank indices (for each i, ..., j)
ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]
```

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>

#### Parameters

  - resource: - resource: Should be from a `Variable` node.
  - indices: - indices: A tensor of indices into the first dimension of `ref`.
  - updates: - updates: A tensor of updated values to add to `ref`.

### `resourceScatterDiv(resource:indices:updates:)`

Divides sparse updates into the variable referenced by `resource`.

``` swift
@inlinable @inline(__always) public static func resourceScatterDiv<Dtype: TensorFlowNumeric, Tindices: TensorFlowIndex>(resource: ResourceHandle, indices: Tensor<Tindices>, updates: Tensor<Dtype>)
```

This operation computes

``` 
# Scalar indices
ref[indices, ...] /= updates[...]

# Vector indices (for each i)
ref[indices[i], ...] /= updates[i, ...]

# High rank indices (for each i, ..., j)
ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]
```

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions multiply.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>

#### Parameters

  - resource: - resource: Should be from a `Variable` node.
  - indices: - indices: A tensor of indices into the first dimension of `ref`.
  - updates: - updates: A tensor of updated values to add to `ref`.

### `resourceScatterMax(resource:indices:updates:)`

Reduces sparse updates into the variable referenced by `resource` using the `max` operation.

``` swift
@inlinable @inline(__always) public static func resourceScatterMax<Dtype: TensorFlowNumeric, Tindices: TensorFlowIndex>(resource: ResourceHandle, indices: Tensor<Tindices>, updates: Tensor<Dtype>)
```

This operation computes

``` 
# Scalar indices
ref[indices, ...] = max(ref[indices, ...], updates[...])

# Vector indices (for each i)
ref[indices[i], ...] = max(ref[indices[i], ...], updates[i, ...])

# High rank indices (for each i, ..., j)
ref[indices[i, ..., j], ...] = max(ref[indices[i, ..., j], ...], updates[i, ..., j, ...])
```

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions are combined.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>

#### Parameters

  - resource: - resource: Should be from a `Variable` node.
  - indices: - indices: A tensor of indices into the first dimension of `ref`.
  - updates: - updates: A tensor of updated values to add to `ref`.

### `resourceScatterMin(resource:indices:updates:)`

Reduces sparse updates into the variable referenced by `resource` using the `min` operation.

``` swift
@inlinable @inline(__always) public static func resourceScatterMin<Dtype: TensorFlowNumeric, Tindices: TensorFlowIndex>(resource: ResourceHandle, indices: Tensor<Tindices>, updates: Tensor<Dtype>)
```

This operation computes

``` 
# Scalar indices
ref[indices, ...] = min(ref[indices, ...], updates[...])

# Vector indices (for each i)
ref[indices[i], ...] = min(ref[indices[i], ...], updates[i, ...])

# High rank indices (for each i, ..., j)
ref[indices[i, ..., j], ...] = min(ref[indices[i, ..., j], ...], updates[i, ..., j, ...])
```

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions are combined.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>

#### Parameters

  - resource: - resource: Should be from a `Variable` node.
  - indices: - indices: A tensor of indices into the first dimension of `ref`.
  - updates: - updates: A tensor of updated values to add to `ref`.

### `resourceScatterMul(resource:indices:updates:)`

Multiplies sparse updates into the variable referenced by `resource`.

``` swift
@inlinable @inline(__always) public static func resourceScatterMul<Dtype: TensorFlowNumeric, Tindices: TensorFlowIndex>(resource: ResourceHandle, indices: Tensor<Tindices>, updates: Tensor<Dtype>)
```

This operation computes

``` 
# Scalar indices
ref[indices, ...] *= updates[...]

# Vector indices (for each i)
ref[indices[i], ...] *= updates[i, ...]

# High rank indices (for each i, ..., j)
ref[indices[i, ..., j], ...] *= updates[i, ..., j, ...]
```

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions multiply.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>

#### Parameters

  - resource: - resource: Should be from a `Variable` node.
  - indices: - indices: A tensor of indices into the first dimension of `ref`.
  - updates: - updates: A tensor of updated values to add to `ref`.

### `resourceScatterNdAdd(ref:indices:updates:useLocking:)`

Applies sparse addition to individual values or slices in a Variable.

``` swift
@inlinable @inline(__always) public static func resourceScatterNdAdd<T: TensorFlowScalar, Tindices: TensorFlowIndex>(ref: ResourceHandle, indices: Tensor<Tindices>, updates: Tensor<T>, useLocking: Bool = true)
```

`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

``` 
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]
```

For example, say we want to add 4 scattered elements to a rank-1 tensor to
8 elements. In Python, that addition would look like this:

``` python
ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8], use_resource=True)
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
add = tf.scatter_nd_add(ref, indices, updates)
with tf.Session() as sess:
  print sess.run(add)
```

The resulting update to ref would look like this:

``` 
[1, 13, 3, 14, 14, 6, 7, 20]
```

See `tf.scatter_nd` for more details about how to make updates to
slices.

#### Parameters

  - ref: - ref: A resource handle. Must be from a VarHandleOp.
  - indices: - indices: A Tensor. Must be one of the following types: int32, int64. A tensor of indices into ref.
  - updates: - updates: A Tensor. Must have the same type as ref. A tensor of values to add to ref.

### `resourceScatterNdSub(ref:indices:updates:useLocking:)`

Applies sparse subtraction to individual values or slices in a Variable.

``` swift
@inlinable @inline(__always) public static func resourceScatterNdSub<T: TensorFlowScalar, Tindices: TensorFlowIndex>(ref: ResourceHandle, indices: Tensor<Tindices>, updates: Tensor<T>, useLocking: Bool = true)
```

`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

``` 
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]
```

For example, say we want to subtract 4 scattered elements from a rank-1 tensor
with 8 elements. In Python, that subtraction would look like this:

``` python
ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8], use_resource=True)
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
sub = tf.scatter_nd_sub(ref, indices, updates)
with tf.Session() as sess:
  print sess.run(sub)
```

The resulting update to ref would look like this:

``` 
[1, -9, 3, -6, -4, 6, 7, -4]
```

See `tf.scatter_nd` for more details about how to make updates to
slices.

#### Parameters

  - ref: - ref: A resource handle. Must be from a VarHandleOp.
  - indices: - indices: A Tensor. Must be one of the following types: int32, int64. A tensor of indices into ref.
  - updates: - updates: A Tensor. Must have the same type as ref. A tensor of values to add to ref.

### `resourceScatterNdUpdate(ref:indices:updates:useLocking:)`

Applies sparse `updates` to individual values or slices within a given

``` swift
@inlinable @inline(__always) public static func resourceScatterNdUpdate<T: TensorFlowScalar, Tindices: TensorFlowIndex>(ref: ResourceHandle, indices: Tensor<Tindices>, updates: Tensor<T>, useLocking: Bool = true)
```

variable according to `indices`.

`ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `ref`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `ref`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

``` 
[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
```

For example, say we want to update 4 scattered elements to a rank-1 tensor to
8 elements. In Python, that update would look like this:

``` python
    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    indices = tf.constant([[4], [3], [1] ,[7]])
    updates = tf.constant([9, 10, 11, 12])
    update = tf.scatter_nd_update(ref, indices, updates)
    with tf.Session() as sess:
      print sess.run(update)
```

The resulting update to ref would look like this:

``` 
[1, 11, 3, 10, 9, 6, 7, 12]
```

See `tf.scatter_nd` for more details about how to make updates to
slices.

#### Parameters

  - ref: - ref: A resource handle. Must be from a VarHandleOp.
  - indices: - indices: A Tensor. Must be one of the following types: int32, int64. A tensor of indices into ref.
  - updates: - updates: A Tensor. Must have the same type as ref. A tensor of updated values to add to ref.

### `resourceScatterSub(resource:indices:updates:)`

Subtracts sparse updates from the variable referenced by `resource`.

``` swift
@inlinable @inline(__always) public static func resourceScatterSub<Dtype: TensorFlowNumeric, Tindices: TensorFlowIndex>(resource: ResourceHandle, indices: Tensor<Tindices>, updates: Tensor<Dtype>)
```

This operation computes

``` 
# Scalar indices
ref[indices, ...] -= updates[...]

# Vector indices (for each i)
ref[indices[i], ...] -= updates[i, ...]

# High rank indices (for each i, ..., j)
ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
```

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions add.

Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src='https://www.tensorflow.org/images/ScatterAdd.png' alt>
</div>

#### Parameters

  - resource: - resource: Should be from a `Variable` node.
  - indices: - indices: A tensor of indices into the first dimension of `ref`.
  - updates: - updates: A tensor of updated values to add to `ref`.

### `resourceScatterUpdate(resource:indices:updates:)`

Assigns sparse updates to the variable referenced by `resource`.

``` swift
@inlinable @inline(__always) public static func resourceScatterUpdate<Dtype: TensorFlowScalar, Tindices: TensorFlowIndex>(resource: ResourceHandle, indices: Tensor<Tindices>, updates: Tensor<Dtype>)
```

This operation computes

``` 
# Scalar indices
ref[indices, ...] = updates[...]

# Vector indices (for each i)
ref[indices[i], ...] = updates[i, ...]

# High rank indices (for each i, ..., j)
ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
```

#### Parameters

  - resource: - resource: Should be from a `Variable` node.
  - indices: - indices: A tensor of indices into the first dimension of `ref`.
  - updates: - updates: A tensor of updated values to add to `ref`.

### `resourceSparseApplyAdadelta(var_:accum:accumUpdate:lr:rho:epsilon:grad:indices:useLocking:)`

var: Should be from a Variable().

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyAdadelta<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, accum: ResourceHandle, accumUpdate: ResourceHandle, lr: Tensor<T>, rho: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, indices: Tensor<Tindices>, useLocking: Bool = false)
```

#### Parameters

  - accum: - accum: Should be from a Variable().
  - lr: - lr: Learning rate. Must be a scalar.
  - rho: - rho: Decay factor. Must be a scalar.
  - epsilon: - epsilon: Constant factor. Must be a scalar.
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var and accum.

### `resourceSparseApplyAdagrad(var_:accum:lr:grad:indices:useLocking:updateSlots:)`

Update relevant entries in '\*var' and '\*accum' according to the adagrad scheme.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyAdagrad<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, accum: ResourceHandle, lr: Tensor<T>, grad: Tensor<T>, indices: Tensor<Tindices>, useLocking: Bool = false, updateSlots: Bool = true)
```

That is for rows we have grad for, we update var and accum as follows:
accum += grad \* grad
var -= lr \* grad \* (1 / sqrt(accum))

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Learning rate. Must be a scalar.
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var and accum.

### `resourceSparseApplyAdagradDA(var_:gradientAccumulator:gradientSquaredAccumulator:grad:indices:lr:l1:l2:globalStep:useLocking:)`

Update entries in '\*var' and '\*accum' according to the proximal adagrad scheme.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyAdagradDA<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, gradientAccumulator: ResourceHandle, gradientSquaredAccumulator: ResourceHandle, grad: Tensor<T>, indices: Tensor<Tindices>, lr: Tensor<T>, l1: Tensor<T>, l2: Tensor<T>, globalStep: Tensor<Int64>, useLocking: Bool = false)
```

#### Parameters

  - var: - var: Should be from a Variable().
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var and accum.
  - lr: - lr: Learning rate. Must be a scalar.
  - l1: - l1: L1 regularization. Must be a scalar.
  - l2: - l2: L2 regularization. Must be a scalar.

### `resourceSparseApplyAdagradV2(var_:accum:lr:epsilon:grad:indices:useLocking:updateSlots:)`

Update relevant entries in '\*var' and '\*accum' according to the adagrad scheme.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyAdagradV2<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, accum: ResourceHandle, lr: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, indices: Tensor<Tindices>, useLocking: Bool = false, updateSlots: Bool = true)
```

That is for rows we have grad for, we update var and accum as follows:
accum += grad \* grad
var -= lr \* grad \* (1 / sqrt(accum))

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Learning rate. Must be a scalar.
  - epsilon: - epsilon: Constant factor. Must be a scalar.
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var and accum.

### `resourceSparseApplyCenteredRMSProp(var_:mg:ms:mom:lr:rho:momentum:epsilon:grad:indices:useLocking:)`

Update '\*var' according to the centered RMSProp algorithm.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyCenteredRMSProp<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, mg: ResourceHandle, ms: ResourceHandle, mom: ResourceHandle, lr: Tensor<T>, rho: Tensor<T>, momentum: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, indices: Tensor<Tindices>, useLocking: Bool = false)
```

The centered RMSProp algorithm uses an estimate of the centered second moment
(i.e., the variance) for normalization, as opposed to regular RMSProp, which
uses the (uncentered) second moment. This often helps with training, but is
slightly more expensive in terms of computation and memory.

Note that in dense implementation of this algorithm, mg, ms, and mom will
update even if the grad is zero, but in this sparse implementation, mg, ms,
and mom will not update in iterations during which the grad is zero.

mean\_square = decay \* mean\_square + (1-decay) \* gradient \*\* 2
mean\_grad = decay \* mean\_grad + (1-decay) \* gradient
Delta = learning\_rate \* gradient / sqrt(mean\_square + epsilon - mean\_grad \*\* 2)

ms \<- rho \* ms\_{t-1} + (1-rho) \* grad \* grad
mom \<- momentum \* mom\_{t-1} + lr \* grad / sqrt(ms + epsilon)
var \<- var - mom

#### Parameters

  - var: - var: Should be from a Variable().
  - mg: - mg: Should be from a Variable().
  - ms: - ms: Should be from a Variable().
  - mom: - mom: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - rho: - rho: Decay rate. Must be a scalar.
  - epsilon: - epsilon: Ridge term. Must be a scalar.
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var, ms and mom.

### `resourceSparseApplyFtrl(var_:accum:linear:grad:indices:lr:l1:l2:lrPower:useLocking:)`

Update relevant entries in '\*var' according to the Ftrl-proximal scheme.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyFtrl<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, accum: ResourceHandle, linear: ResourceHandle, grad: Tensor<T>, indices: Tensor<Tindices>, lr: Tensor<T>, l1: Tensor<T>, l2: Tensor<T>, lrPower: Tensor<T>, useLocking: Bool = false)
```

That is for rows we have grad for, we update var, accum and linear as follows:
accum\_new = accum + grad \* grad
linear += grad - (accum\_new^(-lr\_power) - accum^(-lr\_power)) / lr \* var
quadratic = 1.0 / (accum\_new^(lr\_power) \* lr) + 2 \* l2
var = (sign(linear) \* l1 - linear) / quadratic if |linear| \> l1 else 0.0
accum = accum\_new

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - linear: - linear: Should be from a Variable().
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var and accum.
  - lr: - lr: Scaling factor. Must be a scalar.
  - l1: - l1: L1 regularization. Must be a scalar.
  - l2: - l2: L2 regularization. Must be a scalar.

### `resourceSparseApplyFtrlV2(var_:accum:linear:grad:indices:lr:l1:l2:l2Shrinkage:lrPower:useLocking:)`

Update relevant entries in '\*var' according to the Ftrl-proximal scheme.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyFtrlV2<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, accum: ResourceHandle, linear: ResourceHandle, grad: Tensor<T>, indices: Tensor<Tindices>, lr: Tensor<T>, l1: Tensor<T>, l2: Tensor<T>, l2Shrinkage: Tensor<T>, lrPower: Tensor<T>, useLocking: Bool = false)
```

That is for rows we have grad for, we update var, accum and linear as follows:
grad\_with\_shrinkage = grad + 2 \* l2\_shrinkage \* var
accum\_new = accum + grad\_with\_shrinkage \* grad\_with\_shrinkage
linear += grad\_with\_shrinkage +
(accum\_new^(-lr\_power) - accum^(-lr\_power)) / lr \* var
quadratic = 1.0 / (accum\_new^(lr\_power) \* lr) + 2 \* l2
var = (sign(linear) \* l1 - linear) / quadratic if |linear| \> l1 else 0.0
accum = accum\_new

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - linear: - linear: Should be from a Variable().
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var and accum.
  - lr: - lr: Scaling factor. Must be a scalar.
  - l1: - l1: L1 regularization. Must be a scalar.
  - l2: - l2: L2 shrinkage regulariation. Must be a scalar.

### `resourceSparseApplyKerasMomentum(var_:accum:lr:grad:indices:momentum:useLocking:useNesterov:)`

Update relevant entries in '\*var' and '\*accum' according to the momentum scheme.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyKerasMomentum<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, accum: ResourceHandle, lr: Tensor<T>, grad: Tensor<T>, indices: Tensor<Tindices>, momentum: Tensor<T>, useLocking: Bool = false, useNesterov: Bool = false)
```

Set use\_nesterov = True if you want to use Nesterov momentum.

That is for rows we have grad for, we update var and accum as follows:

accum = accum \* momentum - lr \* grad
var += accum

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Learning rate. Must be a scalar.
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var and accum.
  - momentum: - momentum: Momentum. Must be a scalar.

### `resourceSparseApplyMomentum(var_:accum:lr:grad:indices:momentum:useLocking:useNesterov:)`

Update relevant entries in '\*var' and '\*accum' according to the momentum scheme.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyMomentum<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, accum: ResourceHandle, lr: Tensor<T>, grad: Tensor<T>, indices: Tensor<Tindices>, momentum: Tensor<T>, useLocking: Bool = false, useNesterov: Bool = false)
```

Set use\_nesterov = True if you want to use Nesterov momentum.

That is for rows we have grad for, we update var and accum as follows:

accum = accum \* momentum + grad
var -= lr \* accum

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Learning rate. Must be a scalar.
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var and accum.
  - momentum: - momentum: Momentum. Must be a scalar.

### `resourceSparseApplyProximalAdagrad(var_:accum:lr:l1:l2:grad:indices:useLocking:)`

Sparse update entries in '\*var' and '\*accum' according to FOBOS algorithm.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyProximalAdagrad<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, accum: ResourceHandle, lr: Tensor<T>, l1: Tensor<T>, l2: Tensor<T>, grad: Tensor<T>, indices: Tensor<Tindices>, useLocking: Bool = false)
```

That is for rows we have grad for, we update var and accum as follows:
accum += grad \* grad
prox\_v = var
prox\_v -= lr \* grad \* (1 / sqrt(accum))
var = sign(prox\_v)/(1+lr*l2) \* max{|prox\_v|-lr*l1,0}

#### Parameters

  - var: - var: Should be from a Variable().
  - accum: - accum: Should be from a Variable().
  - lr: - lr: Learning rate. Must be a scalar.
  - l1: - l1: L1 regularization. Must be a scalar.
  - l2: - l2: L2 regularization. Must be a scalar.
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var and accum.

### `resourceSparseApplyProximalGradientDescent(var_:alpha:l1:l2:grad:indices:useLocking:)`

Sparse update '\*var' as FOBOS algorithm with fixed learning rate.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyProximalGradientDescent<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, alpha: Tensor<T>, l1: Tensor<T>, l2: Tensor<T>, grad: Tensor<T>, indices: Tensor<Tindices>, useLocking: Bool = false)
```

That is for rows we have grad for, we update var as follows:
prox\_v = var - alpha \* grad
var = sign(prox\_v)/(1+alpha*l2) \* max{|prox\_v|-alpha*l1,0}

#### Parameters

  - var: - var: Should be from a Variable().
  - alpha: - alpha: Scaling factor. Must be a scalar.
  - l1: - l1: L1 regularization. Must be a scalar.
  - l2: - l2: L2 regularization. Must be a scalar.
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var and accum.

### `resourceSparseApplyRMSProp(var_:ms:mom:lr:rho:momentum:epsilon:grad:indices:useLocking:)`

Update '\*var' according to the RMSProp algorithm.

``` swift
@inlinable @inline(__always) public static func resourceSparseApplyRMSProp<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(var_: ResourceHandle, ms: ResourceHandle, mom: ResourceHandle, lr: Tensor<T>, rho: Tensor<T>, momentum: Tensor<T>, epsilon: Tensor<T>, grad: Tensor<T>, indices: Tensor<Tindices>, useLocking: Bool = false)
```

Note that in dense implementation of this algorithm, ms and mom will
update even if the grad is zero, but in this sparse implementation, ms
and mom will not update in iterations during which the grad is zero.

mean\_square = decay \* mean\_square + (1-decay) \* gradient \*\* 2
Delta = learning\_rate \* gradient / sqrt(mean\_square + epsilon)

ms \<- rho \* ms\_{t-1} + (1-rho) \* grad \* grad
mom \<- momentum \* mom\_{t-1} + lr \* grad / sqrt(ms + epsilon)
var \<- var - mom

#### Parameters

  - var: - var: Should be from a Variable().
  - ms: - ms: Should be from a Variable().
  - mom: - mom: Should be from a Variable().
  - lr: - lr: Scaling factor. Must be a scalar.
  - rho: - rho: Decay rate. Must be a scalar.
  - epsilon: - epsilon: Ridge term. Must be a scalar.
  - grad: - grad: The gradient.
  - indices: - indices: A vector of indices into the first dimension of var, ms and mom.

### `resourceStridedSliceAssign(ref:begin:end:strides:value:beginMask:endMask:ellipsisMask:newAxisMask:shrinkAxisMask:)`

Assign `value` to the sliced l-value reference of `ref`.

``` swift
@inlinable @inline(__always) public static func resourceStridedSliceAssign<T: TensorFlowScalar, Index: TensorFlowIndex>(ref: ResourceHandle, begin: Tensor<Index>, end: Tensor<Index>, strides: Tensor<Index>, value: Tensor<T>, beginMask: Int64 = 0, endMask: Int64 = 0, ellipsisMask: Int64 = 0, newAxisMask: Int64 = 0, shrinkAxisMask: Int64 = 0)
```

The values of `value` are assigned to the positions in the variable
`ref` that are selected by the slice parameters. The slice parameters
`begin, `end`, `strides`, etc. work exactly as in `StridedSlice\`.

NOTE this op currently does not support broadcasting and so `value`'s
shape must be exactly the shape produced by the slice of `ref`.

### `resourceUsingOp(resource:)`

``` swift
@inlinable @inline(__always) public static func resourceUsingOp(resource: ResourceHandle)
```

### `restore(filePattern:tensorName:preferredShard:)`

Restores a tensor from checkpoint files.

``` swift
@inlinable @inline(__always) public static func restore<Dt: TensorFlowScalar>(filePattern: StringTensor, tensorName: StringTensor, preferredShard: Int64 = -1) -> Tensor<Dt>
```

Reads a tensor stored in one or several files. If there are several files (for
instance because a tensor was saved as slices), `file_pattern` may contain
wildcard symbols (`*` and `?`) in the filename portion only, not in the
directory portion.

If a `file_pattern` matches several files, `preferred_shard` can be used to hint
in which file the requested tensor is likely to be found. This op will first
open the file at index `preferred_shard` in the list of matching files and try
to restore tensors from that file.  Only if some tensors or tensor slices are
not found in that first file, then the Op opens all the files. Setting
`preferred_shard` to match the value passed as the `shard` input
of a matching `Save` Op may speed up Restore.  This attribute only affects
performance, not correctness.  The default value -1 means files are processed in
order.

See also `RestoreSlice`.

### `restoreSlice(filePattern:tensorName:shapeAndSlice:preferredShard:)`

Restores a tensor from checkpoint files.

``` swift
@inlinable @inline(__always) public static func restoreSlice<Dt: TensorFlowScalar>(filePattern: StringTensor, tensorName: StringTensor, shapeAndSlice: StringTensor, preferredShard: Int64 = -1) -> Tensor<Dt>
```

This is like `Restore` except that restored tensor can be listed as filling
only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
larger tensor and the slice that the restored tensor covers.

The `shape_and_slice` input has the same format as the
elements of the `shapes_and_slices` input of the `SaveSlices` op.

### `restoreV2(prefix:tensorNames:shapeAndSlices:)`

Restores tensors from a V2 checkpoint.

``` swift
@inlinable @inline(__always) public static func restoreV2<Dtypes: TensorGroup>(prefix: StringTensor, tensorNames: StringTensor, shapeAndSlices: StringTensor) -> Dtypes
```

For backward compatibility with the V1 format, this Op currently allows
restoring from a V1 checkpoint as well:

By default, restores the named tensors in full.  If the caller wishes to restore
specific slices of stored tensors, "shape\_and\_slices" should be non-empty
strings and correspondingly well-formed.

Callers must ensure all the named tensors are indeed stored in the checkpoint.

#### Parameters

  - prefix: - prefix: Must have a single element.  The prefix of a V2 checkpoint.

### `restrict(_:)`

``` swift
@inlinable @inline(__always) public static func restrict<T: TensorFlowScalar>(_ a: Tensor<T>) -> Tensor<T>
```

### `restrict(_:)`

``` swift
@inlinable @inline(__always) public static func restrict(_ a: StringTensor) -> StringTensor
```

### `retrieveTPUEmbeddingADAMParameters(tableId:tableName:numShards:shardId:config:)`

Retrieve ADAM embedding parameters.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingADAMParameters(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (parameters: Tensor<Float>, momenta: Tensor<Float>, velocities: Tensor<Float>)
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingADAMParametersGradAccumDebug(tableId:tableName:numShards:shardId:config:)`

Retrieve ADAM embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingADAMParametersGradAccumDebug(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (
    parameters: Tensor<Float>, momenta: Tensor<Float>, velocities: Tensor<Float>,
    gradientAccumulators: Tensor<Float>
  )
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingAdadeltaParameters(tableId:tableName:numShards:shardId:config:)`

Retrieve Adadelta embedding parameters.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingAdadeltaParameters(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, updates: Tensor<Float>)
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingAdadeltaParametersGradAccumDebug(tableId:tableName:numShards:shardId:config:)`

Retrieve Adadelta embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingAdadeltaParametersGradAccumDebug(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (
    parameters: Tensor<Float>, accumulators: Tensor<Float>, updates: Tensor<Float>,
    gradientAccumulators: Tensor<Float>
  )
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingAdagradParameters(tableId:tableName:numShards:shardId:config:)`

Retrieve Adagrad embedding parameters.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingAdagradParameters(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>)
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingAdagradParametersGradAccumDebug(tableId:tableName:numShards:shardId:config:)`

Retrieve Adagrad embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingAdagradParametersGradAccumDebug(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (
    parameters: Tensor<Float>, accumulators: Tensor<Float>, gradientAccumulators: Tensor<Float>
  )
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingCenteredRMSPropParameters(tableId:tableName:numShards:shardId:config:)`

Retrieve centered RMSProp embedding parameters.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingCenteredRMSPropParameters(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>, mg: Tensor<Float>)
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingFTRLParameters(tableId:tableName:numShards:shardId:config:)`

Retrieve FTRL embedding parameters.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingFTRLParameters(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>, linears: Tensor<Float>)
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingFTRLParametersGradAccumDebug(tableId:tableName:numShards:shardId:config:)`

Retrieve FTRL embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingFTRLParametersGradAccumDebug(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (
    parameters: Tensor<Float>, accumulators: Tensor<Float>, linears: Tensor<Float>,
    gradientAccumulators: Tensor<Float>
  )
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingMDLAdagradLightParameters(tableId:tableName:numShards:shardId:config:)`

Retrieve MDL Adagrad Light embedding parameters.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingMDLAdagradLightParameters(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (
    parameters: Tensor<Float>, accumulators: Tensor<Float>, weights: Tensor<Float>,
    benefits: Tensor<Float>
  )
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingMomentumParameters(tableId:tableName:numShards:shardId:config:)`

Retrieve Momentum embedding parameters.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingMomentumParameters(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (parameters: Tensor<Float>, momenta: Tensor<Float>)
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingMomentumParametersGradAccumDebug(tableId:tableName:numShards:shardId:config:)`

Retrieve Momentum embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingMomentumParametersGradAccumDebug(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (parameters: Tensor<Float>, momenta: Tensor<Float>, gradientAccumulators: Tensor<Float>)
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingProximalAdagradParameters(tableId:tableName:numShards:shardId:config:)`

Retrieve proximal Adagrad embedding parameters.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingProximalAdagradParameters(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (parameters: Tensor<Float>, accumulators: Tensor<Float>)
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug(tableId:tableName:numShards:shardId:config:)`

Retrieve proximal Adagrad embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingProximalAdagradParametersGradAccumDebug(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (
    parameters: Tensor<Float>, accumulators: Tensor<Float>, gradientAccumulators: Tensor<Float>
  )
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingRMSPropParameters(tableId:tableName:numShards:shardId:config:)`

Retrieve RMSProp embedding parameters.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingRMSPropParameters(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>)
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingRMSPropParametersGradAccumDebug(tableId:tableName:numShards:shardId:config:)`

Retrieve RMSProp embedding parameters with debug support.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingRMSPropParametersGradAccumDebug(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> (
    parameters: Tensor<Float>, ms: Tensor<Float>, mom: Tensor<Float>,
    gradientAccumulators: Tensor<Float>
  )
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `retrieveTPUEmbeddingStochasticGradientDescentParameters(tableId:tableName:numShards:shardId:config:)`

Retrieve SGD embedding parameters.

``` swift
@inlinable @inline(__always) public static func retrieveTPUEmbeddingStochasticGradientDescentParameters(tableId: Int64 = -1, tableName: String, numShards: Int64, shardId: Int64, config: String) -> Tensor<Float>
```

An op that retrieves optimization parameters from embedding to host
memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
the correct embedding table configuration. For example, this op is
used to retrieve updated parameters before saving a checkpoint.

### `reverse(_:dims:)`

Reverses specific dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func reverse<T: TensorFlowScalar>(_ tensor: Tensor<T>, dims: Tensor<Bool>) -> Tensor<T>
```

Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
of `tensor`, this operation reverses each dimension i of `tensor` where
`dims[i]` is `True`.

`tensor` can have up to 8 dimensions. The number of dimensions
of `tensor` must equal the number of elements in `dims`. In other words:

`rank(tensor) = size(dims)`

For example:

``` 
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]
 
# 'dims' is [False, False, False, True]
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]
 
# 'dims' is [False, True, False, False]
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]
 
# 'dims' is [False, False, True, False]
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

#### Parameters

  - tensor: - tensor: Up to 8-D.
  - dims: - dims: 1-D. The dimensions to reverse.

### `reverse(_:dims:)`

Reverses specific dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func reverse(_ tensor: StringTensor, dims: Tensor<Bool>) -> StringTensor
```

Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
of `tensor`, this operation reverses each dimension i of `tensor` where
`dims[i]` is `True`.

`tensor` can have up to 8 dimensions. The number of dimensions
of `tensor` must equal the number of elements in `dims`. In other words:

`rank(tensor) = size(dims)`

For example:

``` 
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]
 
# 'dims' is [False, False, False, True]
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]
 
# 'dims' is [False, True, False, False]
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]
 
# 'dims' is [False, False, True, False]
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

#### Parameters

  - tensor: - tensor: Up to 8-D.
  - dims: - dims: 1-D. The dimensions to reverse.

### `reverseSequence(_:seqLengths:seqDim:batchDim:)`

Reverses variable length slices.

``` swift
@inlinable @inline(__always) public static func reverseSequence<T: TensorFlowScalar, Tlen: TensorFlowIndex>(_ input: Tensor<T>, seqLengths: Tensor<Tlen>, seqDim: Int64, batchDim: Int64 = 0) -> Tensor<T>
```

This op first slices `input` along the dimension `batch_dim`, and for each
slice `i`, reverses the first `seq_lengths[i]` elements along
the dimension `seq_dim`.

The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

The output slice `i` along dimension `batch_dim` is then given by input
slice `i`, with the first `seq_lengths[i]` slices along dimension
`seq_dim` reversed.

For example:

``` 
# Given this:
batch_dim = 0
seq_dim = 1
input.dims = (4, 8, ...)
seq_lengths = [7, 2, 3, 5]
 
# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
 
# while entries past seq_lens are copied through:
output[0, 7:, :, ...] = input[0, 7:, :, ...]
output[1, 2:, :, ...] = input[1, 2:, :, ...]
output[2, 3:, :, ...] = input[2, 3:, :, ...]
output[3, 2:, :, ...] = input[3, 2:, :, ...]
```

In contrast, if:

``` 
# Given this:
batch_dim = 2
seq_dim = 0
input.dims = (8, ?, 4, ...)
seq_lengths = [7, 2, 3, 5]
 
# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]
 
# while entries past seq_lens are copied through:
output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
```

#### Parameters

  - input: - input: The input to reverse.

### `reverseV2(_:axis:)`

Reverses specific dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func reverseV2<Tidx: TensorFlowIndex, T: TensorFlowScalar>(_ tensor: Tensor<T>, axis: Tensor<Tidx>) -> Tensor<T>
```

NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
`tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.

Given a `tensor`, and a `int32` tensor `axis` representing the set of
dimensions of `tensor` to reverse. This operation reverses each dimension
`i` for which there exists `j` s.t. `axis[j] == i`.

`tensor` can have up to 8 dimensions. The number of dimensions specified
in `axis` may be 0 or more entries. If an index is specified more than
once, a InvalidArgument error is raised.

For example:

``` 
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]
 
# 'dims' is [3] or 'dims' is [-1]
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]
 
# 'dims' is '[1]' (or 'dims' is '[-3]')
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]
 
# 'dims' is '[2]' (or 'dims' is '[-2]')
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

#### Parameters

  - tensor: - tensor: Up to 8-D.
  - axis: - axis: 1-D. The indices of the dimensions to reverse. Must be in the range `[-rank(tensor), rank(tensor))`.

### `reverseV2(_:axis:)`

Reverses specific dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func reverseV2<Tidx: TensorFlowIndex>(_ tensor: StringTensor, axis: Tensor<Tidx>) -> StringTensor
```

NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
`tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.

Given a `tensor`, and a `int32` tensor `axis` representing the set of
dimensions of `tensor` to reverse. This operation reverses each dimension
`i` for which there exists `j` s.t. `axis[j] == i`.

`tensor` can have up to 8 dimensions. The number of dimensions specified
in `axis` may be 0 or more entries. If an index is specified more than
once, a InvalidArgument error is raised.

For example:

``` 
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]
 
# 'dims' is [3] or 'dims' is [-1]
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]
 
# 'dims' is '[1]' (or 'dims' is '[-3]')
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]
 
# 'dims' is '[2]' (or 'dims' is '[-2]')
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

#### Parameters

  - tensor: - tensor: Up to 8-D.
  - axis: - axis: 1-D. The indices of the dimensions to reverse. Must be in the range `[-rank(tensor), rank(tensor))`.

### `rightShift(_:_:)`

Elementwise computes the bitwise right-shift of `x` and `y`.

``` swift
@inlinable @inline(__always) public static func rightShift<T: TensorFlowInteger>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

Performs a logical shift for unsigned integer types, and an arithmetic shift
for signed integer types.

If `y` is negative, or greater than or equal to than the width of `x` in bits
the result is implementation defined.

Example:

``` python
import tensorflow as tf
from tensorflow.python.ops import bitwise_ops
import numpy as np
dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64]
 
for dtype in dtype_list:
  lhs = tf.constant([-1, -5, -3, -14], dtype=dtype)
  rhs = tf.constant([5, 0, 7, 11], dtype=dtype)
 
  right_shift_result = bitwise_ops.right_shift(lhs, rhs)
 
  print(right_shift_result)
 
# This will print:
# tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int8)
# tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int16)
# tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int32)
# tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int64)
 
lhs = np.array([-2, 64, 101, 32], dtype=np.int8)
rhs = np.array([-1, -5, -3, -14], dtype=np.int8)
bitwise_ops.right_shift(lhs, rhs)
# <tf.Tensor: shape=(4,), dtype=int8, numpy=array([ -2,  64, 101,  32], dtype=int8)>
```

### `rint(_:)`

Returns element-wise integer closest to x.

``` swift
@inlinable @inline(__always) public static func rint<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

If the result is midway between two representable values,
the even representable is chosen.
For example:

``` 
rint(-1.5) ==> -2.0
rint(0.5000001) ==> 1.0
rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
```

### `rngSkip(resource:algorithm:delta:)`

Advance the counter of a counter-based RNG.

``` swift
@inlinable @inline(__always) public static func rngSkip(resource: ResourceHandle, algorithm: Tensor<Int64>, delta: Tensor<Int64>)
```

The state of the RNG after
`rng_skip(n)` will be the same as that after `stateful_uniform([n])`
(or any other distribution). The actual increment added to the
counter is an unspecified implementation detail.

#### Parameters

  - resource: - resource: The handle of the resource variable that stores the state of the RNG.
  - algorithm: - algorithm: The RNG algorithm.
  - delta: - delta: The amount of advancement.

### `roll(_:shift:axis:)`

Rolls the elements of a tensor along an axis.

``` swift
@inlinable @inline(__always) public static func roll<T: TensorFlowScalar, Tshift: TensorFlowIndex, Taxis: TensorFlowIndex>(_ input: Tensor<T>, shift: Tensor<Tshift>, axis: Tensor<Taxis>) -> Tensor<T>
```

The elements are shifted positively (towards larger indices) by the offset of
`shift` along the dimension of `axis`. Negative `shift` values will shift
elements in the opposite direction. Elements that roll passed the last position
will wrap around to the first and vice versa. Multiple shifts along multiple
axes may be specified.

For example:

``` 
# 't' is [0, 1, 2, 3, 4]
roll(t, shift=2, axis=0) ==> [3, 4, 0, 1, 2]
 
# shifting along multiple dimensions
# 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
roll(t, shift=[1, -2], axis=[0, 1]) ==> [[7, 8, 9, 5, 6], [2, 3, 4, 0, 1]]
 
# shifting along the same axis multiple times
# 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
roll(t, shift=[2, -3], axis=[1, 1]) ==> [[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]
```

#### Parameters

  - shift: - shift: Dimension must be 0-D or 1-D. `shift[i]` specifies the number of places by which elements are shifted positively (towards larger indices) along the dimension specified by `axis[i]`. Negative shifts will roll the elements in the opposite direction.
  - axis: - axis: Dimension must be 0-D or 1-D. `axis[i]` specifies the dimension that the shift `shift[i]` should occur. If the same axis is referenced more than once, the total shift for that axis will be the sum of all the shifts that belong to that axis.

### `round(_:)`

Rounds the values of a tensor to the nearest integer, element-wise.

``` swift
@inlinable @inline(__always) public static func round<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

Rounds half to even.  Also known as bankers rounding. If you want to round
according to the current system rounding mode use std::cint.

### `rpc(address:method:request:protocol_:failFast:timeoutInMs:)`

Perform batches of RPC requests.

``` swift
@inlinable @inline(__always) public static func rpc(address: StringTensor, method: StringTensor, request: StringTensor, protocol_: String, failFast: Bool = true, timeoutInMs: Int64 = 0) -> StringTensor
```

This op asynchronously performs either a single RPC request, or a batch
of requests.  RPC requests are defined by three main parameters:

For example, if you have an RPC service running on port localhost:2345,
and its interface is configured with the following proto declaration:

``` 
service MyService {
  rpc MyMethod(MyRequestProto) returns (MyResponseProto) {
  }
};
```

then call this op with arguments:

``` 
address = "localhost:2345"
method = "MyService/MyMethod"
```

The `request` tensor is a string tensor representing serialized `MyRequestProto`
strings; and the output string tensor `response` will have the same shape
and contain (upon successful completion) corresponding serialized
`MyResponseProto` strings.

For example, to send a single, empty, `MyRequestProto`, call
this op with `request = ""`.  To send 5 **parallel** empty requests,
call this op with `request = ["", "", "", "", ""]`.

More generally, one can create a batch of `MyRequestProto` serialized protos
from regular batched tensors using the `encode_proto` op, and convert
the response `MyResponseProto` serialized protos to batched tensors
using the `decode_proto` op.

**NOTE** Working with serialized proto strings is faster than instantiating
actual proto objects in memory, so no performance degradation is expected
compared to writing custom kernels for this workflow.

If the connection fails or the remote worker returns an error
status, the op reraises this exception locally.

See the `TryRpc` op if you prefer to handle RPC failures manually in the graph.

#### Parameters

  - address: - address: `0-D` or `1-D`.  The address (i.e. host\_name:port) of the RPC server. If this tensor has more than 1 element, then multiple parallel rpc requests are sent.  This argument broadcasts with `method` and `request`.
  - method: - method: `0-D` or `1-D`.  The method address on the RPC server. If this tensor has more than 1 element, then multiple parallel rpc requests are sent.  This argument broadcasts with `address` and `request`.
  - request: - request: `0-D` or `1-D`.  Serialized proto strings: the rpc request argument. If this tensor has more than 1 element, then multiple parallel rpc requests are sent.  This argument broadcasts with `address` and `method`.

### `rsqrt(_:)`

Computes reciprocal of square root of x element-wise.

``` swift
@inlinable @inline(__always) public static func rsqrt<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

I.e., \\(y = 1 / \\sqrt{x}\\).

### `rsqrtGrad(_:dy:)`

Computes the gradient for the rsqrt of `x` wrt its input.

``` swift
@inlinable @inline(__always) public static func rsqrtGrad<T: FloatingPoint & TensorFlowScalar>(_ y: Tensor<T>, dy: Tensor<T>) -> Tensor<T>
```

Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
is the corresponding input gradient.

### `sampleDistortedBoundingBox(imageSize:boundingBoxes:seed:seed2:minObjectCovered:aspectRatioRange:areaRange:maxAttempts:useImageIfNoBoundingBoxes:)`

Generate a single randomly distorted bounding box for an image.

``` swift
@inlinable @inline(__always) public static func sampleDistortedBoundingBox<T: TensorFlowInteger>(imageSize: Tensor<T>, boundingBoxes: Tensor<Float>, seed: Int64 = 0, seed2: Int64 = 0, minObjectCovered: Double = 0.1, aspectRatioRange: [Double] = [0.75, 1.33], areaRange: [Double] = [0.05, 1], maxAttempts: Int64 = 100, useImageIfNoBoundingBoxes: Bool = false) -> (begin: Tensor<T>, size: Tensor<T>, bboxes: Tensor<Float>)
```

Bounding box annotations are often supplied in addition to ground-truth labels
in image recognition or object localization tasks. A common technique for
training such a system is to randomly distort an image while preserving
its content, i.e. *data augmentation*. This Op outputs a randomly distorted
localization of an object, i.e. bounding box, given an `image_size`,
`bounding_boxes` and a series of constraints.

The output of this Op is a single bounding box that may be used to crop the
original image. The output is returned as 3 tensors: `begin`, `size` and
`bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
what the bounding box looks like.

Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example,

``` python
    # Generate a single distorted bounding box.
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bounding_boxes)
 
    # Draw the bounding box in an image summary.
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox_for_draw)
    tf.summary.image('images_with_box', image_with_box)
 
    # Employ the bounding box to distort the image.
    distorted_image = tf.slice(image, begin, size)
```

Note that if no bounding box information is available, setting
`use_image_if_no_bounding_boxes = true` will assume there is a single implicit
bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
false and no bounding boxes are supplied, an error is raised.

### `sampleDistortedBoundingBoxV2(imageSize:boundingBoxes:minObjectCovered:seed:seed2:aspectRatioRange:areaRange:maxAttempts:useImageIfNoBoundingBoxes:)`

Generate a single randomly distorted bounding box for an image.

``` swift
@inlinable @inline(__always) public static func sampleDistortedBoundingBoxV2<T: TensorFlowInteger>(imageSize: Tensor<T>, boundingBoxes: Tensor<Float>, minObjectCovered: Tensor<Float>, seed: Int64 = 0, seed2: Int64 = 0, aspectRatioRange: [Double] = [0.75, 1.33], areaRange: [Double] = [0.05, 1], maxAttempts: Int64 = 100, useImageIfNoBoundingBoxes: Bool = false) -> (begin: Tensor<T>, size: Tensor<T>, bboxes: Tensor<Float>)
```

Bounding box annotations are often supplied in addition to ground-truth labels
in image recognition or object localization tasks. A common technique for
training such a system is to randomly distort an image while preserving
its content, i.e. *data augmentation*. This Op outputs a randomly distorted
localization of an object, i.e. bounding box, given an `image_size`,
`bounding_boxes` and a series of constraints.

The output of this Op is a single bounding box that may be used to crop the
original image. The output is returned as 3 tensors: `begin`, `size` and
`bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
image. The latter may be supplied to `tf.image.draw_bounding_boxes` to visualize
what the bounding box looks like.

Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
height of the underlying image.

For example,

``` python
    # Generate a single distorted bounding box.
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bounding_boxes)
 
    # Draw the bounding box in an image summary.
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox_for_draw)
    tf.summary.image('images_with_box', image_with_box)
 
    # Employ the bounding box to distort the image.
    distorted_image = tf.slice(image, begin, size)
```

Note that if no bounding box information is available, setting
`use_image_if_no_bounding_boxes = true` will assume there is a single implicit
bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
false and no bounding boxes are supplied, an error is raised.

### `samplingDataset(inputDataset:rate:seed:seed2:outputTypes:outputShapes:)`

Creates a dataset that takes a Bernoulli sample of the contents of another dataset.

``` swift
@inlinable @inline(__always) public static func samplingDataset(inputDataset: VariantHandle, rate: Tensor<Float>, seed: Tensor<Int64>, seed2: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

There is no transformation in the `tf.data` Python API for creating this dataset.
Instead, it is created as a result of the `filter_with_random_uniform_fusion`
static optimization. Whether this optimization is performed is determined by the
`experimental_optimization.filter_with_random_uniform_fusion` option of
`tf.data.Options`.

#### Parameters

  - rate: - rate: A scalar representing the sample rate. Each element of `input_dataset` is retained with this probability, independent of all other elements.
  - seed: - seed: A scalar representing seed of random number generator.
  - seed2: - seed2: A scalar representing seed2 of random number generator.

### `save(filename:tensorNames:data:)`

Saves the input tensors to disk.

``` swift
@inlinable @inline(__always) public static func save<T: TensorArrayProtocol>(filename: StringTensor, tensorNames: StringTensor, data: T)
```

The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
is written to `filename` with name `tensor_names[i]`.

See also `SaveSlices`.

#### Parameters

  - filename: - filename: Must have a single element. The name of the file to which we write the tensor.
  - data: - data: `N` tensors to save.

### `saveSlices(filename:tensorNames:shapesAndSlices:data:)`

Saves input tensors slices to disk.

``` swift
@inlinable @inline(__always) public static func saveSlices<T: TensorArrayProtocol>(filename: StringTensor, tensorNames: StringTensor, shapesAndSlices: StringTensor, data: T)
```

This is like `Save` except that tensors can be listed in the saved file as being
a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
larger tensor and the slice that this tensor covers. `shapes_and_slices` must
have as many elements as `tensor_names`.

Elements of the `shapes_and_slices` input must either be:

`slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
where each `sliceI` is either:

See also `Save`.

#### Parameters

  - filename: - filename: Must have a single element. The name of the file to which we write the tensor.
  - data: - data: `N` tensors to save.

### `saveV2(prefix:tensorNames:shapeAndSlices:tensors:)`

Saves tensors in V2 checkpoint format.

``` swift
@inlinable @inline(__always) public static func saveV2<Dtypes: TensorArrayProtocol>(prefix: StringTensor, tensorNames: StringTensor, shapeAndSlices: StringTensor, tensors: Dtypes)
```

By default, saves the named tensors in full.  If the caller wishes to save
specific slices of full tensors, "shape\_and\_slices" should be non-empty strings
and correspondingly well-formed.

#### Parameters

  - prefix: - prefix: Must have a single element. The prefix of the V2 checkpoint to which we write the tensors.
  - tensors: - tensors: `N` tensors to save.

### `scalarSummary(tags:_:)`

Outputs a `Summary` protocol buffer with scalar values.

``` swift
@inlinable @inline(__always) public static func scalarSummary<T: TensorFlowNumeric>(tags: StringTensor, _ values: Tensor<T>) -> StringTensor
```

The input `tags` and `values` must have the same shape.  The generated summary
has a summary value for each tag-value pair in `tags` and `values`.

#### Parameters

  - tags: - tags: Tags for the summary.
  - values: - values: Same shape as \`tags.  Values for the summary.

### `scaleAndTranslate(images:size:scale:translation:kernelType:antialias:)`

``` swift
@inlinable @inline(__always) public static func scaleAndTranslate<T: TensorFlowNumeric>(images: Tensor<T>, size: Tensor<Int32>, scale: Tensor<Float>, translation: Tensor<Float>, kernelType: String = "lanczos3", antialias: Bool = true) -> Tensor<Float>
```

### `scaleAndTranslateGrad(grads:originalImage:scale:translation:kernelType:antialias:)`

``` swift
@inlinable @inline(__always) public static func scaleAndTranslateGrad<T: FloatingPoint & TensorFlowScalar>(grads: Tensor<T>, originalImage: Tensor<T>, scale: Tensor<Float>, translation: Tensor<Float>, kernelType: String = "lanczos3", antialias: Bool = true) -> Tensor<T>
```

### `scanDataset(inputDataset:initialState:otherArguments:f:outputTypes:outputShapes:preserveCardinality:useDefaultDevice:)`

Creates a dataset successively reduces `f` over the elements of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func scanDataset<FIn: TensorGroup, FOut: TensorGroup, Tstate: TensorArrayProtocol, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, initialState: Tstate, otherArguments: Targuments, f: (FIn) -> FOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?], preserveCardinality: Bool = false, useDefaultDevice: Bool = true) -> VariantHandle
```

### `scatterNd(indices:updates:shape:)`

Scatter `updates` into a new tensor according to `indices`.

``` swift
@inlinable @inline(__always) public static func scatterNd<T: TensorFlowScalar, Tindices: TensorFlowIndex>(indices: Tensor<Tindices>, updates: Tensor<T>, shape: Tensor<Tindices>) -> Tensor<T>
```

Creates a new tensor by applying sparse `updates` to individual values or
slices within a tensor (initially zero for numeric, empty for string) of
the given `shape` according to indices.  This operator is the inverse of the
`tf.gather_nd` operator which extracts values or slices from a given tensor.

This operation is similar to tensor\_scatter\_add, except that the tensor is
zero-initialized. Calling `tf.scatter_nd(indices, values, shape)` is identical
to `tensor_scatter_add(tf.zeros(shape, values.dtype), indices, values)`

If `indices` contains duplicates, then their updates are accumulated (summed).

**WARNING**: The order in which updates are applied is nondeterministic, so the
output will be nondeterministic if `indices` contains duplicates -- because
of some numerical approximation issues, numbers summed in different order
may yield different results.

`indices` is an integer tensor containing indices into a new tensor of shape
`shape`.  The last dimension of `indices` can be at most the rank of `shape`:

``` 
indices.shape[-1] <= shape.rank
```

The last dimension of `indices` corresponds to indices into elements
(if `indices.shape[-1] = shape.rank`) or slices
(if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
`shape`.  `updates` is a tensor with shape

``` 
indices.shape[:-1] + shape[indices.shape[-1]:]
```

The simplest form of scatter is to insert individual elements in a tensor by
index. For example, say we want to insert 4 scattered elements in a rank-1
tensor with 8 elements.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
</div>

In Python, this scatter operation would look like this:

``` python
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
```

The resulting tensor would look like this:

``` 
[0, 11, 0, 10, 9, 0, 0, 12]
```

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
</div>

In Python, this scatter operation would look like this:

``` python
    indices = tf.constant([[0], [2]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
```

The resulting tensor would look like this:

``` 
[[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
 [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
```

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, the index is ignored.

#### Parameters

  - indices: - indices: Index tensor.
  - updates: - updates: Updates to scatter into output.
  - shape: - shape: 1-D. The shape of the resulting tensor.

### `scatterNdNonAliasingAdd(_:indices:updates:)`

Applies sparse addition to `input` using individual values or slices

``` swift
@inlinable @inline(__always) public static func scatterNdNonAliasingAdd<T: TensorFlowScalar, Tindices: TensorFlowIndex>(_ input: Tensor<T>, indices: Tensor<Tindices>, updates: Tensor<T>) -> Tensor<T>
```

from `updates` according to indices `indices`.  The updates are non-aliasing:
`input` is only modified in-place if no other operations will use it.
Otherwise, a copy of `input` is made.  This operation has a gradient with
respect to both `input` and `updates`.

`input` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

`indices` must be integer tensor, containing indices into `input`.
It must be shape \\(\[d\_0, ..., d\_{Q-2}, K\]\\) where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or `(P-K)`-dimensional slices
(if `K < P`) along the `K`th dimension of `input`.

`updates` is `Tensor` of rank `Q-1+P-K` with shape:

$$\[d\_0, ..., d\_{Q-2}, input.shape\[K\], ..., input.shape\[P-1\]\].$$

For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
elements. In Python, that addition would look like this:

``` 
input = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
output = tf.scatter_nd_non_aliasing_add(input, indices, updates)
with tf.Session() as sess:
  print(sess.run(output))
```

The resulting value `output` would look like this:

``` 
[1, 13, 3, 14, 14, 6, 7, 20]
```

See `tf.scatter_nd` for more details about how to make updates to slices.

#### Parameters

  - input: - input: A Tensor.
  - indices: - indices: A Tensor. Must be one of the following types: `int32`, `int64`. A tensor of indices into `input`.
  - updates: - updates: A Tensor. Must have the same type as ref. A tensor of updated values to add to `input`.

### `sdcaFprint(_:)`

Computes fingerprints of the input strings.

``` swift
@inlinable @inline(__always) public static func sdcaFprint(_ input: StringTensor) -> Tensor<Int64>
```

#### Parameters

  - input: - input: vector of strings to compute fingerprints on.

### `sdcaOptimizer(sparseExampleIndices:sparseFeatureIndices:sparseFeatureValues:denseFeatures:exampleWeights:exampleLabels:sparseIndices:sparseWeights:denseWeights:exampleStateData:lossType:adaptative:l1:l2:numLossPartitions:numInnerIterations:)`

Distributed version of Stochastic Dual Coordinate Ascent (SDCA) optimizer for

``` swift
@inlinable @inline(__always) public static func sdcaOptimizer(sparseExampleIndices: [Tensor<Int64>], sparseFeatureIndices: [Tensor<Int64>], sparseFeatureValues: [Tensor<Float>], denseFeatures: [Tensor<Float>], exampleWeights: Tensor<Float>, exampleLabels: Tensor<Float>, sparseIndices: [Tensor<Int64>], sparseWeights: [Tensor<Float>], denseWeights: [Tensor<Float>], exampleStateData: Tensor<Float>, lossType: LossType, adaptative: Bool = false, l1: Double, l2: Double, numLossPartitions: Int64, numInnerIterations: Int64) -> (
    outExampleStateData: Tensor<Float>, outDeltaSparseWeights: [Tensor<Float>],
    outDeltaDenseWeights: [Tensor<Float>]
  )
```

linear models with L1 + L2 regularization. As global optimization objective is
strongly-convex, the optimizer optimizes the dual objective at each step. The
optimizer applies each update one example at a time. Examples are sampled
uniformly, and the optimizer is learning rate free and enjoys linear convergence
rate.

[Proximal Stochastic Dual Coordinate Ascent](http://arxiv.org/pdf/1211.2717v1.pdf).<br>
Shai Shalev-Shwartz, Tong Zhang. 2012

$$Loss Objective = \\sum f\_{i} (wx\_{i}) + (l2 / 2) \* |w|^2 + l1 \* |w|$$

[Adding vs. Averaging in Distributed Primal-Dual Optimization](http://arxiv.org/abs/1502.03508).<br>
Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan,
Peter Richtarik, Martin Takac. 2015

[Stochastic Dual Coordinate Ascent with Adaptive Probabilities](https://arxiv.org/abs/1502.08053).<br>
Dominik Csiba, Zheng Qu, Peter Richtarik. 2015

### `sdcaOptimizerV2(sparseExampleIndices:sparseFeatureIndices:sparseFeatureValues:denseFeatures:exampleWeights:exampleLabels:sparseIndices:sparseWeights:denseWeights:exampleStateData:lossType:adaptive:l1:l2:numLossPartitions:numInnerIterations:)`

Distributed version of Stochastic Dual Coordinate Ascent (SDCA) optimizer for

``` swift
@inlinable @inline(__always) public static func sdcaOptimizerV2(sparseExampleIndices: [Tensor<Int64>], sparseFeatureIndices: [Tensor<Int64>], sparseFeatureValues: [Tensor<Float>], denseFeatures: [Tensor<Float>], exampleWeights: Tensor<Float>, exampleLabels: Tensor<Float>, sparseIndices: [Tensor<Int64>], sparseWeights: [Tensor<Float>], denseWeights: [Tensor<Float>], exampleStateData: Tensor<Float>, lossType: LossType, adaptive: Bool = false, l1: Double, l2: Double, numLossPartitions: Int64, numInnerIterations: Int64) -> (
    outExampleStateData: Tensor<Float>, outDeltaSparseWeights: [Tensor<Float>],
    outDeltaDenseWeights: [Tensor<Float>]
  )
```

linear models with L1 + L2 regularization. As global optimization objective is
strongly-convex, the optimizer optimizes the dual objective at each step. The
optimizer applies each update one example at a time. Examples are sampled
uniformly, and the optimizer is learning rate free and enjoys linear convergence
rate.

[Proximal Stochastic Dual Coordinate Ascent](http://arxiv.org/pdf/1211.2717v1.pdf).<br>
Shai Shalev-Shwartz, Tong Zhang. 2012

$$Loss Objective = \\sum f\_{i} (wx\_{i}) + (l2 / 2) \* |w|^2 + l1 \* |w|$$

[Adding vs. Averaging in Distributed Primal-Dual Optimization](http://arxiv.org/abs/1502.03508).<br>
Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan,
Peter Richtarik, Martin Takac. 2015

[Stochastic Dual Coordinate Ascent with Adaptive Probabilities](https://arxiv.org/abs/1502.08053).<br>
Dominik Csiba, Zheng Qu, Peter Richtarik. 2015

### `segmentMax(data:segmentIds:)`

Computes the maximum along segments of a tensor.

``` swift
@inlinable @inline(__always) public static func segmentMax<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(data: Tensor<T>, segmentIds: Tensor<Tindices>) -> Tensor<T>
```

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output\_i = \\max\_j(data\_j)\\) where `max` is over `j` such
that `segment_ids[j] == i`.

If the max is empty for a given segment ID `i`, `output[i] = 0`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentMax.png" alt>
</div>

For example:

``` 
c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_max(c, tf.constant([0, 0, 1]))
# ==> [[4, 3, 3, 4],
#      [5, 6, 7, 8]]
```

### `segmentMean(data:segmentIds:)`

Computes the mean along segments of a tensor.

``` swift
@inlinable @inline(__always) public static func segmentMean<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(data: Tensor<T>, segmentIds: Tensor<Tindices>) -> Tensor<T>
```

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output\_i = \\frac{\\sum\_j data\_j}{N}\\) where `mean` is
over `j` such that `segment_ids[j] == i` and `N` is the total number of
values summed.

If the mean is empty for a given segment ID `i`, `output[i] = 0`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentMean.png" alt>
</div>

For example:

``` 
c = tf.constant([[1.0,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_mean(c, tf.constant([0, 0, 1]))
# ==> [[2.5, 2.5, 2.5, 2.5],
#      [5, 6, 7, 8]]
```

### `segmentMin(data:segmentIds:)`

Computes the minimum along segments of a tensor.

``` swift
@inlinable @inline(__always) public static func segmentMin<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(data: Tensor<T>, segmentIds: Tensor<Tindices>) -> Tensor<T>
```

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output\_i = \\min\_j(data\_j)\\) where `min` is over `j` such
that `segment_ids[j] == i`.

If the min is empty for a given segment ID `i`, `output[i] = 0`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
</div>

For example:

``` 
c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_min(c, tf.constant([0, 0, 1]))
# ==> [[1, 2, 2, 1],
#      [5, 6, 7, 8]]
```

### `segmentProd(data:segmentIds:)`

Computes the product along segments of a tensor.

``` swift
@inlinable @inline(__always) public static func segmentProd<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(data: Tensor<T>, segmentIds: Tensor<Tindices>) -> Tensor<T>
```

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output\_i = \\prod\_j data\_j\\) where the product is over `j` such
that `segment_ids[j] == i`.

If the product is empty for a given segment ID `i`, `output[i] = 1`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentProd.png" alt>
</div>

For example:

``` 
c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_prod(c, tf.constant([0, 0, 1]))
# ==> [[4, 6, 6, 4],
#      [5, 6, 7, 8]]
```

### `segmentSum(data:segmentIds:)`

Computes the sum along segments of a tensor.

``` swift
@inlinable @inline(__always) public static func segmentSum<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(data: Tensor<T>, segmentIds: Tensor<Tindices>) -> Tensor<T>
```

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output\_i = \\sum\_j data\_j\\) where sum is over `j` such
that `segment_ids[j] == i`.

If the sum is empty for a given segment ID `i`, `output[i] = 0`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/SegmentSum.png" alt>
</div>

For example:

``` 
c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
tf.segment_sum(c, tf.constant([0, 0, 1]))
# ==> [[5, 5, 5, 5],
#      [5, 6, 7, 8]]
```

### `select(condition:t:e:)`

Selects elements from `x` or `y`, depending on `condition`.

``` swift
@inlinable @inline(__always) public static func select<T: TensorFlowScalar>(condition: Tensor<Bool>, t: Tensor<T>, e: Tensor<T>) -> Tensor<T>
```

The `x`, and `y` tensors must all have the same shape, and the
output will also have that shape.

The `condition` tensor must be a scalar if `x` and `y` are scalars.
If `x` and `y` are vectors or higher rank, then `condition` must be either a
scalar, a vector with size matching the first dimension of `x`, or must have
the same shape as `x`.

The `condition` tensor acts as a mask that chooses, based on the value at each
element, whether the corresponding element / row in the output should be
taken from `x` (if true) or `y` (if false).

If `condition` is a vector and `x` and `y` are higher rank matrices, then
it chooses which row (outer dimension) to copy from `x` and `y`.
If `condition` has the same shape as `x` and `y`, then it chooses which
element to copy from `x` and `y`.

For example:

``` python
# 'condition' tensor is [[True,  False]
#                        [False, True]]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e)  # => [[1, 6], [7, 4]]
 
 
# 'condition' tensor is [True, False]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e) ==> [[1, 2],
                             [7, 8]]
 
```

#### Parameters

  - t: - t: = A `Tensor` which may have the same shape as `condition`. If `condition` is rank 1, `x` may have higher rank, but its first dimension must match the size of `condition`.
  - e: - e: = A `Tensor` with the same type and shape as `x`.

### `selectV2(condition:t:e:)`

``` swift
@inlinable @inline(__always) public static func selectV2<T: TensorFlowScalar>(condition: Tensor<Bool>, t: Tensor<T>, e: Tensor<T>) -> Tensor<T>
```

### `selfAdjointEig(_:)`

Computes the Eigen Decomposition of a batch of square self-adjoint matrices.

``` swift
@inlinable @inline(__always) public static func selfAdjointEig<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix
SelfAdjointEig.

The result is a \[..., M+1, M\] matrix with \[..., 0,:\] containing the
eigenvalues, and subsequent \[...,1:, :\] containing the eigenvectors. The eigenvalues
are sorted in non-decreasing order.

#### Parameters

  - input: - input: Shape is `[..., M, M]`.

### `selfAdjointEigV2(_:computeV:)`

Computes the eigen decomposition of one or more square self-adjoint matrices.

``` swift
@inlinable @inline(__always) public static func selfAdjointEigV2<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, computeV: Bool = true) -> (e: Tensor<T>, v: Tensor<T>)
```

Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
`input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`. The eigenvalues
are sorted in non-decreasing order.

``` python
# a is a tensor.
# e is a tensor of eigenvalues.
# v is a tensor of eigenvectors.
e, v = self_adjoint_eig(a)
e = self_adjoint_eig(a, compute_v=False)
```

#### Parameters

  - input: - input: `Tensor` input of shape `[N, N]`.

### `selu(features:)`

Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`

``` swift
@inlinable @inline(__always) public static func selu<T: FloatingPoint & TensorFlowScalar>(features: Tensor<T>) -> Tensor<T>
```

if \< 0, `scale * features` otherwise.

To be used together with
`initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN')`.
For correct dropout, use `tf.contrib.nn.alpha_dropout`.

See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

### `seluGrad(gradients:outputs:)`

Computes gradients for the scaled exponential linear (Selu) operation.

``` swift
@inlinable @inline(__always) public static func seluGrad<T: FloatingPoint & TensorFlowScalar>(gradients: Tensor<T>, outputs: Tensor<T>) -> Tensor<T>
```

#### Parameters

  - gradients: - gradients: The backpropagated gradients to the corresponding Selu operation.
  - outputs: - outputs: The outputs of the corresponding Selu operation.

### `send(_:tensorName:sendDevice:sendDeviceIncarnation:recvDevice:clientTerminated:)`

Sends the named tensor from send\_device to recv\_device.

``` swift
@inlinable @inline(__always) public static func send<T: TensorFlowScalar>(_ tensor: Tensor<T>, tensorName: String, sendDevice: String, sendDeviceIncarnation: Int64, recvDevice: String, clientTerminated: Bool = false)
```

#### Parameters

  - tensor: - tensor: The tensor to send.

### `sendTPUEmbeddingGradients(inputs:learningRates:config:)`

Performs gradient updates of embedding tables.

``` swift
@inlinable @inline(__always) public static func sendTPUEmbeddingGradients(inputs: [Tensor<Float>], learningRates: [Tensor<Float>], config: String)
```

#### Parameters

  - inputs: - inputs: A TensorList of gradients with which to update embedding tables. This argument has the same length and shapes as the return value of RecvTPUEmbeddingActivations, but contains gradients of the model's loss with respect to the embedding activations. The embedding tables are updated from these gradients via the optimizer specified in the TPU embedding configuration given to tpu.initialize\_system.

### `serializeIterator(resourceHandle:)`

Converts the given `resource_handle` representing an iterator to a variant tensor.

``` swift
@inlinable @inline(__always) public static func serializeIterator(resourceHandle: ResourceHandle) -> VariantHandle
```

### `serializeManySparse(sparseIndices:sparseValues:sparseShape:)`

Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` `Tensor` object.

``` swift
@inlinable @inline(__always) public static func serializeManySparse<T: TensorFlowScalar, OutType: TensorFlowScalar>(sparseIndices: Tensor<Int64>, sparseValues: Tensor<T>, sparseShape: Tensor<Int64>) -> Tensor<OutType>
```

The `SparseTensor` must have rank `R` greater than 1, and the first dimension
is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The serialized
`SparseTensor` objects going into each row of `serialized_sparse` will have
rank `R-1`.

The minibatch size `N` is extracted from `sparse_shape[0]`.

### `serializeManySparse(sparseIndices:sparseValues:sparseShape:)`

Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` `Tensor` object.

``` swift
@inlinable @inline(__always) public static func serializeManySparse<T: TensorFlowScalar>(sparseIndices: Tensor<Int64>, sparseValues: Tensor<T>, sparseShape: Tensor<Int64>) -> StringTensor
```

The `SparseTensor` must have rank `R` greater than 1, and the first dimension
is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The serialized
`SparseTensor` objects going into each row of `serialized_sparse` will have
rank `R-1`.

The minibatch size `N` is extracted from `sparse_shape[0]`.

### `serializeSparse(sparseIndices:sparseValues:sparseShape:)`

Serialize a `SparseTensor` into a `[3]` `Tensor` object.

``` swift
@inlinable @inline(__always) public static func serializeSparse<T: TensorFlowScalar, OutType: TensorFlowScalar>(sparseIndices: Tensor<Int64>, sparseValues: Tensor<T>, sparseShape: Tensor<Int64>) -> Tensor<OutType>
```

### `serializeSparse(sparseIndices:sparseValues:sparseShape:)`

Serialize a `SparseTensor` into a `[3]` `Tensor` object.

``` swift
@inlinable @inline(__always) public static func serializeSparse<T: TensorFlowScalar>(sparseIndices: Tensor<Int64>, sparseValues: Tensor<T>, sparseShape: Tensor<Int64>) -> StringTensor
```

### `serializeTensor(_:)`

Transforms a Tensor into a serialized TensorProto proto.

``` swift
@inlinable @inline(__always) public static func serializeTensor<T: TensorFlowScalar>(_ tensor: Tensor<T>) -> StringTensor
```

#### Parameters

  - tensor: - tensor: A Tensor of type `T`.

### `setSize(setIndices:setValues:setShape:validateIndices:)`

Number of unique elements along last dimension of input `set`.

``` swift
@inlinable @inline(__always) public static func setSize<T: TensorFlowInteger>(setIndices: Tensor<Int64>, setValues: Tensor<T>, setShape: Tensor<Int64>, validateIndices: Bool = true) -> Tensor<Int32>
```

Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
and `set_shape`. The last dimension contains values in a set, duplicates are
allowed but ignored.

If `validate_indices` is `True`, this op validates the order and range of `set`
indices.

### `setSize(setIndices:setValues:setShape:validateIndices:)`

Number of unique elements along last dimension of input `set`.

``` swift
@inlinable @inline(__always) public static func setSize(setIndices: Tensor<Int64>, setValues: StringTensor, setShape: Tensor<Int64>, validateIndices: Bool = true) -> Tensor<Int32>
```

Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
and `set_shape`. The last dimension contains values in a set, duplicates are
allowed but ignored.

If `validate_indices` is `True`, this op validates the order and range of `set`
indices.

### `setStatsAggregatorDataset(inputDataset:statsAggregator:tag:counterPrefix:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func setStatsAggregatorDataset(inputDataset: VariantHandle, statsAggregator: ResourceHandle, tag: StringTensor, counterPrefix: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `shape(_:)`

Returns the shape of a tensor.

``` swift
@inlinable @inline(__always) public static func shape<T: TensorFlowScalar, OutType: TensorFlowIndex>(_ input: Tensor<T>) -> Tensor<OutType>
```

This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

``` 
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```

### `shapeN(_:)`

Returns shape of tensors.

``` swift
@inlinable @inline(__always) public static func shapeN<T: TensorFlowScalar, OutType: TensorFlowIndex>(_ input: [Tensor<T>]) -> [Tensor<OutType>]
```

This operation returns N 1-D integer tensors representing shape of `input[i]s`.

### `shardDataset(inputDataset:numShards:index:requireNonEmpty:outputTypes:outputShapes:)`

Creates a `Dataset` that includes only 1/`num_shards` of this dataset.

``` swift
@inlinable @inline(__always) public static func shardDataset(inputDataset: VariantHandle, numShards: Tensor<Int64>, index: Tensor<Int64>, requireNonEmpty: Bool = false, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

#### Parameters

  - index: - index: An integer representing the current worker index.

### `shardedFilename(basename:shard:numShards:)`

Generate a sharded filename. The filename is printf formatted as

``` swift
@inlinable @inline(__always) public static func shardedFilename(basename: StringTensor, shard: Tensor<Int32>, numShards: Tensor<Int32>) -> StringTensor
```

%s-%05d-of-%05d, basename, shard, num\_shards.

### `shardedFilespec(basename:numShards:)`

Generate a glob pattern matching all sharded file names.

``` swift
@inlinable @inline(__always) public static func shardedFilespec(basename: StringTensor, numShards: Tensor<Int32>) -> StringTensor
```

### `shuffleAndRepeatDataset(inputDataset:bufferSize:seed:seed2:count:outputTypes:outputShapes:)`

Creates a dataset that shuffles and repeats elements from `input_dataset`

``` swift
@inlinable @inline(__always) public static func shuffleAndRepeatDataset(inputDataset: VariantHandle, bufferSize: Tensor<Int64>, seed: Tensor<Int64>, seed2: Tensor<Int64>, count: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

pseudorandomly.

#### Parameters

  - seed: - seed: A scalar seed for the random number generator. If either `seed` or `seed2` is set to be non-zero, the random number generator is seeded by the given seed.  Otherwise, a random seed is used.
  - seed2: - seed2: A second scalar seed to avoid seed collision.
  - count: - count: A scalar representing the number of times the underlying dataset should be repeated. The default is `-1`, which results in infinite repetition.

### `shuffleDataset(inputDataset:bufferSize:seed:seed2:reshuffleEachIteration:outputTypes:outputShapes:)`

Creates a dataset that shuffles elements from `input_dataset` pseudorandomly.

``` swift
@inlinable @inline(__always) public static func shuffleDataset(inputDataset: VariantHandle, bufferSize: Tensor<Int64>, seed: Tensor<Int64>, seed2: Tensor<Int64>, reshuffleEachIteration: Bool = true, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

#### Parameters

  - seed: - seed: A scalar seed for the random number generator. If either `seed` or `seed2` is set to be non-zero, the random number generator is seeded by the given seed.  Otherwise, a random seed is used.
  - seed2: - seed2: A second scalar seed to avoid seed collision.

### `shuffleDatasetV2(inputDataset:bufferSize:seedGenerator:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func shuffleDatasetV2(inputDataset: VariantHandle, bufferSize: Tensor<Int64>, seedGenerator: ResourceHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `shutdownDistributedTPU()`

Shuts down a running distributed TPU system.

``` swift
@inlinable @inline(__always) public static func shutdownDistributedTPU()
```

The op returns an error if no system is running.

### `sigmoid(_:)`

Computes sigmoid of `x` element-wise.

``` swift
@inlinable @inline(__always) public static func sigmoid<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Specifically, `y = 1 / (1 + exp(-x))`.

### `sigmoidGrad(_:dy:)`

Computes the gradient of the sigmoid of `x` wrt its input.

``` swift
@inlinable @inline(__always) public static func sigmoidGrad<T: FloatingPoint & TensorFlowScalar>(_ y: Tensor<T>, dy: Tensor<T>) -> Tensor<T>
```

Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
`dy` is the corresponding input gradient.

### `sign(_:)`

Returns an element-wise indication of the sign of a number.

``` swift
@inlinable @inline(__always) public static func sign<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

`y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

Example usage:

> > > tf.math.sign(\[0., 2., -3.\])
> > > \<tf.Tensor: shape=(3,), dtype=float32, numpy=array(\[ 0.,  1., -1.\], dtype=float32)\>

### `simple(_:)`

``` swift
@inlinable @inline(__always) public static func simple(_ a: Tensor<Int32>) -> Tensor<Float>
```

### `simpleStruct(nA:)`

``` swift
@inlinable @inline(__always) public static func simpleStruct(nA: Int64) -> [Tensor<Int32>]
```

### `sin(_:)`

Computes sine of x element-wise.

``` swift
@inlinable @inline(__always) public static func sin<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Given an input tensor, this function computes sine of every
element in the tensor. Input range is `(-inf, inf)` and
output range is `[-1,1]`.

``` python
x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10, float("inf")])
tf.math.sin(x) ==> [nan -0.4121185 -0.47942555 0.84147096 0.9320391 -0.87329733 -0.54402107 nan]
```

### `sinh(_:)`

Computes hyperbolic sine of x element-wise.

``` swift
@inlinable @inline(__always) public static func sinh<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Given an input tensor, this function computes hyperbolic sine of every
element in the tensor. Input range is `[-inf,inf]` and output range
is `[-inf,inf]`.

``` python
x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
tf.math.sinh(x) ==> [-inf -4.0515420e+03 -5.2109528e-01 1.1752012e+00 1.5094614e+00 3.6268604e+00 1.1013232e+04 inf]
```

### `size(_:)`

Returns the size of a tensor.

``` swift
@inlinable @inline(__always) public static func size<T: TensorFlowScalar, OutType: TensorFlowIndex>(_ input: Tensor<T>) -> Tensor<OutType>
```

This operation returns an integer representing the number of elements in
`input`.

For example:

``` 
# 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
size(t) ==> 12
```

### `skipDataset(inputDataset:count:outputTypes:outputShapes:)`

Creates a dataset that skips `count` elements from the `input_dataset`.

``` swift
@inlinable @inline(__always) public static func skipDataset(inputDataset: VariantHandle, count: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

#### Parameters

  - count: - count: A scalar representing the number of elements from the `input_dataset` that should be skipped.  If count is -1, skips everything.

### `skipgram(filename:batchSize:windowSize:minCount:subsample:)`

Parses a text file and creates a batch of examples.

``` swift
@inlinable @inline(__always) public static func skipgram(filename: String, batchSize: Int64, windowSize: Int64 = 5, minCount: Int64 = 5, subsample: Double = 0.001) -> (
    vocabWord: StringTensor, vocabFreq: Tensor<Int32>, wordsPerEpoch: Tensor<Int64>,
    currentEpoch: Tensor<Int32>, totalWordsProcessed: Tensor<Int64>, examples: Tensor<Int32>,
    labels: Tensor<Int32>
  )
```

### `sleepDataset(inputDataset:sleepMicroseconds:outputTypes:outputShapes:)`

``` swift
@inlinable @inline(__always) public static func sleepDataset(inputDataset: VariantHandle, sleepMicroseconds: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `slice(_:begin:size:)`

Return a slice from 'input'.

``` swift
@inlinable @inline(__always) public static func slice<T: TensorFlowScalar, Index: TensorFlowIndex>(_ input: Tensor<T>, begin: Tensor<Index>, size: Tensor<Index>) -> Tensor<T>
```

The output tensor is a tensor with dimensions described by 'size'
whose values are extracted from 'input' starting at the offsets in
'begin'.

*Requirements*:
0 \<= begin\[i\] \<= begin\[i\] + size\[i\] \<= Di  for i in \[0, n)

#### Parameters

  - begin: - begin: begin\[i\] specifies the offset into the 'i'th dimension of 'input' to slice from.
  - size: - size: size\[i\] specifies the number of elements of the 'i'th dimension of 'input' to slice. If size\[i\] is -1, all remaining elements in dimension i are included in the slice (i.e. this is equivalent to setting size\[i\] = input.dim\_size(i) - begin\[i\]).

### `slidingWindowDataset(inputDataset:windowSize:windowShift:windowStride:outputTypes:outputShapes:)`

Creates a dataset that passes a sliding window over `input_dataset`.

``` swift
@inlinable @inline(__always) public static func slidingWindowDataset(inputDataset: VariantHandle, windowSize: Tensor<Int64>, windowShift: Tensor<Int64>, windowStride: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `snapshot(_:)`

Returns a copy of the input tensor.

``` swift
@inlinable @inline(__always) public static func snapshot<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

### `snapshotDataset(inputDataset:path:outputTypes:outputShapes:compression:readerPathPrefix:writerPathPrefix:shardSizeBytes:pendingSnapshotExpirySeconds:numReaderThreads:readerBufferSize:numWriterThreads:writerBufferSize:shuffleOnRead:seed:seed2:)`

Creates a dataset that will write to / read from a snapshot.

``` swift
@inlinable @inline(__always) public static func snapshotDataset(inputDataset: VariantHandle, path: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?], compression: String, readerPathPrefix: String, writerPathPrefix: String, shardSizeBytes: Int64 = 10_737_418_240, pendingSnapshotExpirySeconds: Int64 = 86400, numReaderThreads: Int64 = 1, readerBufferSize: Int64 = 1, numWriterThreads: Int64 = 1, writerBufferSize: Int64 = 1, shuffleOnRead: Bool = false, seed: Int64 = 0, seed2: Int64 = 0) -> VariantHandle
```

This dataset attempts to determine whether a valid snapshot exists at the
`snapshot_path`, and reads from the snapshot in lieu of using `input_dataset`.
If not, it will run the preprocessing pipeline as usual, and write out a
snapshot of the data processed for future use.

#### Parameters

  - path: - path: The path we should write snapshots to / read snapshots from.

### `softmax(logits:)`

Computes softmax activations.

``` swift
@inlinable @inline(__always) public static func softmax<T: FloatingPoint & TensorFlowScalar>(logits: Tensor<T>) -> Tensor<T>
```

For each batch `i` and class `j` we have

``` 
$$softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))$$
```

#### Parameters

  - logits: - logits: 2-D with shape `[batch_size, num_classes]`.

### `softmaxCrossEntropyWithLogits(features:labels:)`

Computes softmax cross entropy cost and gradients to backpropagate.

``` swift
@inlinable @inline(__always) public static func softmaxCrossEntropyWithLogits<T: FloatingPoint & TensorFlowScalar>(features: Tensor<T>, labels: Tensor<T>) -> (loss: Tensor<T>, backprop: Tensor<T>)
```

Inputs are the logits, not probabilities.

#### Parameters

  - features: - features: batch\_size x num\_classes matrix
  - labels: - labels: batch\_size x num\_classes matrix The caller must ensure that each batch of labels represents a valid probability distribution.

### `softplus(features:)`

Computes softplus: `log(exp(features) + 1)`.

``` swift
@inlinable @inline(__always) public static func softplus<T: FloatingPoint & TensorFlowScalar>(features: Tensor<T>) -> Tensor<T>
```

### `softplusGrad(gradients:features:)`

Computes softplus gradients for a softplus operation.

``` swift
@inlinable @inline(__always) public static func softplusGrad<T: FloatingPoint & TensorFlowScalar>(gradients: Tensor<T>, features: Tensor<T>) -> Tensor<T>
```

#### Parameters

  - gradients: - gradients: The backpropagated gradients to the corresponding softplus operation.
  - features: - features: The features passed as input to the corresponding softplus operation.

### `softsign(features:)`

Computes softsign: `features / (abs(features) + 1)`.

``` swift
@inlinable @inline(__always) public static func softsign<T: FloatingPoint & TensorFlowScalar>(features: Tensor<T>) -> Tensor<T>
```

### `softsignGrad(gradients:features:)`

Computes softsign gradients for a softsign operation.

``` swift
@inlinable @inline(__always) public static func softsignGrad<T: FloatingPoint & TensorFlowScalar>(gradients: Tensor<T>, features: Tensor<T>) -> Tensor<T>
```

#### Parameters

  - gradients: - gradients: The backpropagated gradients to the corresponding softsign operation.
  - features: - features: The features passed as input to the corresponding softsign operation.

### `spaceToBatch(_:paddings:blockSize:)`

SpaceToBatch for 4-D tensors of type T.

``` swift
@inlinable @inline(__always) public static func spaceToBatch<T: TensorFlowScalar, Tpaddings: TensorFlowIndex>(_ input: Tensor<T>, paddings: Tensor<Tpaddings>, blockSize: Int64) -> Tensor<T>
```

This is a legacy version of the more general SpaceToBatchND.

Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size.

#### Parameters

  - input: - input: 4-D with shape `[batch, height, width, depth]`.
  - paddings: - paddings: \`\`\`
    paddings = \[\[pad\_top, pad\_bottom\], \[pad\_left, pad\_right\]\]
    The effective spatial dimensions of the zero-padded input tensor will be:
    ``` 
    height_pad = pad_top + height + pad_bottom
    width_pad = pad_left + width + pad_right
    ```
    ``` 
    
    The attr `block_size` must be greater than one. It indicates the block size.
    
    ```
      - Non-overlapping blocks of size `block_size x block size` in the height and
        width dimensions are rearranged into the batch dimension at each location.
      - The batch of the output tensor is `batch * block_size * block_size`.
      - Both height\_pad and width\_pad must be divisible by block\_size.
    ``` 
    
    The shape of the output will be:
    
    ```
    \[batch*block\_size*block\_size, height\_pad/block\_size, width\_pad/block\_size,
    depth\]
    ``` 
    
    Some examples:
    
    (1) For the following input of shape `[1, 2, 2, 1]` and block\_size of 2:
    
    ```
    x = \[\[\[\[1\], \[2\]\], \[\[3\], \[4\]\]\]\]
    ``` 
    
    The output tensor has shape `[4, 1, 1, 1]` and value:
    
    ```
    \[\[\[\[1\]\]\], \[\[\[2\]\]\], \[\[\[3\]\]\], \[\[\[4\]\]\]\]
    ``` 
    
    (2) For the following input of shape `[1, 2, 2, 3]` and block\_size of 2:
    
    ```
    x = \[\[\[\[1, 2, 3\], \[4, 5, 6\]\],
    \[\[7, 8, 9\], \[10, 11, 12\]\]\]\]
    ``` 
    
    The output tensor has shape `[4, 1, 1, 3]` and value:
    
    ```
    \[\[\[\[1, 2, 3\]\]\], \[\[\[4, 5, 6\]\]\], \[\[\[7, 8, 9\]\]\], \[\[\[10, 11, 12\]\]\]\]
    ``` 
    
    (3) For the following input of shape `[1, 4, 4, 1]` and block\_size of 2:
    
    ```
    x = \[\[\[\[1\],   \[2\],  \[3\],  \[4\]\],
    \[\[5\],   \[6\],  \[7\],  \[8\]\],
    \[\[9\],  \[10\], \[11\],  \[12\]\],
    \[\[13\], \[14\], \[15\],  \[16\]\]\]\]
    ``` 
    
    The output tensor has shape `[4, 2, 2, 1]` and value:
    
    ```
    x = \[\[\[\[1\], \[3\]\], \[\[9\], \[11\]\]\],
    \[\[\[2\], \[4\]\], \[\[10\], \[12\]\]\],
    \[\[\[5\], \[7\]\], \[\[13\], \[15\]\]\],
    \[\[\[6\], \[8\]\], \[\[14\], \[16\]\]\]\]
    ``` 
    
    (4) For the following input of shape `[2, 2, 4, 1]` and block\_size of 2:
    
    ```
    x = \[\[\[\[1\],   \[2\],  \[3\],  \[4\]\],
    \[\[5\],   \[6\],  \[7\],  \[8\]\]\],
    \[\[\[9\],  \[10\], \[11\],  \[12\]\],
    \[\[13\], \[14\], \[15\],  \[16\]\]\]\]
    ``` 
    
    The output tensor has shape `[8, 1, 2, 1]` and value:
    
    ```
    x = \[\[\[\[1\], \[3\]\]\], \[\[\[9\], \[11\]\]\], \[\[\[2\], \[4\]\]\], \[\[\[10\], \[12\]\]\],
    \[\[\[5\], \[7\]\]\], \[\[\[13\], \[15\]\]\], \[\[\[6\], \[8\]\]\], \[\[\[14\], \[16\]\]\]\]
    ``` 
    
    Among others, this operation is useful for reducing atrous convolution into
    regular convolution.
    ```

### `spaceToBatchND(_:blockShape:paddings:)`

SpaceToBatch for N-D tensors of type T.

``` swift
@inlinable @inline(__always) public static func spaceToBatchND<T: TensorFlowScalar, TblockShape: TensorFlowIndex, Tpaddings: TensorFlowIndex>(_ input: Tensor<T>, blockShape: Tensor<TblockShape>, paddings: Tensor<Tpaddings>) -> Tensor<T>
```

This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
grid of blocks of shape `block_shape`, and interleaves these blocks with the
"batch" dimension (0) such that in the output, the spatial dimensions
`[1, ..., M]` correspond to the position within the grid, and the batch
dimension combines both the position within a spatial block and the original
batch position.  Prior to division into blocks, the spatial dimensions of the
input are optionally zero padded according to `paddings`.  See below for a
precise description.

#### Parameters

  - input: - input: N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`, where spatial\_shape has `M` dimensions.
  - paddings: - paddings: This operation is equivalent to the following steps:
    1.  Zero-pad the start and end of dimensions `[1, ..., M]` of the
        input according to `paddings` to produce `padded` of shape `padded_shape`.
    
    2.  Reshape `padded` to `reshaped_padded` of shape:
        
        \[batch\] +
        \[padded\_shape\[1\] / block\_shape\[0\],
        block\_shape\[0\],
        ...,
        padded\_shape\[M\] / block\_shape\[M-1\],
        block\_shape\[M-1\]\] +
        remaining\_shape
    
    3.  Permute dimensions of `reshaped_padded` to produce
        `permuted_reshaped_padded` of shape:
        
        block\_shape +
        \[batch\] +
        \[padded\_shape\[1\] / block\_shape\[0\],
        ...,
        padded\_shape\[M\] / block\_shape\[M-1\]\] +
        remaining\_shape
    
    4.  Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
        dimension, producing an output tensor of shape:
        
        \[batch \* prod(block\_shape)\] +
        \[padded\_shape\[1\] / block\_shape\[0\],
        ...,
        padded\_shape\[M\] / block\_shape\[M-1\]\] +
        remaining\_shape
    Some examples:
    (1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
    `paddings = [[0, 0], [0, 0]]`:
    ``` 
    x = [[[[1], [2]], [[3], [4]]]]
    ```
    The output tensor has shape `[4, 1, 1, 1]` and value:
    ``` 
    [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    ```
    (2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
    `paddings = [[0, 0], [0, 0]]`:
    ``` 
    x = [[[[1, 2, 3], [4, 5, 6]],
          [[7, 8, 9], [10, 11, 12]]]]
    ```
    The output tensor has shape `[4, 1, 1, 3]` and value:
    ``` 
    [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
    ```
    (3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
    `paddings = [[0, 0], [0, 0]]`:
    ``` 
    x = [[[[1],   [2],  [3],  [4]],
          [[5],   [6],  [7],  [8]],
          [[9],  [10], [11],  [12]],
          [[13], [14], [15],  [16]]]]
    ```
    The output tensor has shape `[4, 2, 2, 1]` and value:
    ``` 
    x = [[[[1], [3]], [[9], [11]]],
         [[[2], [4]], [[10], [12]]],
         [[[5], [7]], [[13], [15]]],
         [[[6], [8]], [[14], [16]]]]
    ```
    (4) For the following input of shape `[2, 2, 4, 1]`, block\_shape = `[2, 2]`, and
    paddings = `[[0, 0], [2, 0]]`:
    ``` 
    x = [[[[1],   [2],  [3],  [4]],
          [[5],   [6],  [7],  [8]]],
         [[[9],  [10], [11],  [12]],
          [[13], [14], [15],  [16]]]]
    ```
    The output tensor has shape `[8, 1, 3, 1]` and value:
    ``` 
    x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
         [[[0], [2], [4]]], [[[0], [10], [12]]],
         [[[0], [5], [7]]], [[[0], [13], [15]]],
         [[[0], [6], [8]]], [[[0], [14], [16]]]]
    ```
    Among others, this operation is useful for reducing atrous convolution into
    regular convolution.

### `spaceToDepth(_:blockSize:dataFormat:)`

SpaceToDepth for tensors of type T.

``` swift
@inlinable @inline(__always) public static func spaceToDepth<T: TensorFlowScalar>(_ input: Tensor<T>, blockSize: Int64, dataFormat: DataFormat5 = .nhwc) -> Tensor<T>
```

Rearranges blocks of spatial data, into depth. More specifically,
this op outputs a copy of the input tensor where values from the `height`
and `width` dimensions are moved to the `depth` dimension.
The attr `block_size` indicates the input block size.

The `data_format` attr specifies the layout of the input and output tensors
with the following options:
"NHWC": `[ batch, height, width, channels ]`
"NCHW": `[ batch, channels, height, width ]`
"NCHW\_VECT\_C":
`qint8 [ batch, channels / 4, height, width, 4 ]`

It is useful to consider the operation as transforming a 6-D Tensor.
e.g. for data\_format = NHWC,
Each element in the input tensor can be specified via 6 coordinates,
ordered by decreasing memory layout significance as:
n,oY,bY,oX,bX,iC  (where n=batch index, oX, oY means X or Y coordinates
within the output image, bX, bY means coordinates
within the input block, iC means input channels).
The output would be a transpose to the following layout:
n,oY,oX,bY,bX,iC

This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.

For example, given an input of shape `[1, 2, 2, 1]`, data\_format = "NHWC" and
block\_size = 2:

``` 
x = [[[[1], [2]],
      [[3], [4]]]]
```

This operation will output a tensor of shape `[1, 1, 1, 4]`:

``` 
[[[[1, 2, 3, 4]]]]
```

Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
the corresponding output will have a single element (i.e. width and height are
both 1) and will have a depth of 4 channels (1 \* block\_size \* block\_size).
The output element shape is `[1, 1, 4]`.

For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.

``` 
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

This operation, for block\_size of 2, will return the following tensor of shape
`[1, 1, 1, 12]`

``` 
[[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```

Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:

``` 
x = [[[[1],   [2],  [5],  [6]],
      [[3],   [4],  [7],  [8]],
      [[9],  [10], [13],  [14]],
      [[11], [12], [15],  [16]]]]
```

the operator will return the following tensor of shape `[1 2 2 4]`:

``` 
x = [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```

### `sparseAdd(aIndices:aValues:aShape:bIndices:bValues:bShape:thresh:)`

Adds two `SparseTensor` objects to produce another `SparseTensor`.

``` swift
@inlinable @inline(__always) public static func sparseAdd<T: TensorFlowNumeric, Treal: TensorFlowNumeric>(aIndices: Tensor<Int64>, aValues: Tensor<T>, aShape: Tensor<Int64>, bIndices: Tensor<Int64>, bValues: Tensor<T>, bShape: Tensor<Int64>, thresh: Tensor<Treal>) -> (sumIndices: Tensor<Int64>, sumValues: Tensor<T>, sumShape: Tensor<Int64>)
```

The input `SparseTensor` objects' indices are assumed ordered in standard
lexicographic order.  If this is not the case, before this step run
`SparseReorder` to restore index ordering.

By default, if two values sum to zero at some index, the output `SparseTensor`
would still include that particular location in its index, storing a zero in the
corresponding value slot.  To override this, callers can specify `thresh`,
indicating that if the sum has a magnitude strictly smaller than `thresh`, its
corresponding value and index would then not be included.  In particular,
`thresh == 0` (default) means everything is kept and actual thresholding happens
only for a positive value.

In the following shapes, `nnz` is the count after taking `thresh` into account.

#### Parameters

  - thresh: - thresh: 0-D.  The magnitude threshold that determines if an output value/index pair takes space.

### `sparseAddGrad(backpropValGrad:aIndices:bIndices:sumIndices:)`

The gradient operator for the SparseAdd op.

``` swift
@inlinable @inline(__always) public static func sparseAddGrad<T: TensorFlowNumeric>(backpropValGrad: Tensor<T>, aIndices: Tensor<Int64>, bIndices: Tensor<Int64>, sumIndices: Tensor<Int64>) -> (aValGrad: Tensor<T>, bValGrad: Tensor<T>)
```

The SparseAdd op calculates A + B, where A, B, and the sum are all represented
as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
values of A and B.

### `sparseConcat(indices:_:shapes:concatDim:)`

Concatenates a list of `SparseTensor` along the specified dimension.

``` swift
@inlinable @inline(__always) public static func sparseConcat<T: TensorFlowScalar>(indices: [Tensor<Int64>], _ values: [Tensor<T>], shapes: [Tensor<Int64>], concatDim: Int64) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>)
```

Concatenation is with respect to the dense versions of these sparse tensors.
It is assumed that each input is a `SparseTensor` whose elements are ordered
along increasing dimension number.

All inputs' shapes must match, except for the concat dimension.  The
`indices`, `values`, and `shapes` lists must have the same length.

The output shape is identical to the inputs', except along the concat
dimension, where it is the sum of the inputs' sizes along that dimension.

The output elements will be resorted to preserve the sort order along
increasing dimension number.

This op runs in `O(M log M)` time, where `M` is the total number of non-empty
values across all inputs. This is due to the need for an internal sort in
order to concatenate efficiently across an arbitrary dimension.

For example, if `concat_dim = 1` and the inputs are

``` 
sp_inputs[0]: shape = [2, 3]
[0, 2]: "a"
[1, 0]: "b"
[1, 1]: "c"

sp_inputs[1]: shape = [2, 4]
[0, 1]: "d"
[0, 2]: "e"
```

then the output will be

``` 
shape = [2, 7]
[0, 2]: "a"
[0, 4]: "d"
[0, 5]: "e"
[1, 0]: "b"
[1, 1]: "c"
```

Graphically this is equivalent to doing

``` 
[    a] concat [  d e  ] = [    a   d e  ]
[b c  ]        [       ]   [b c          ]
```

#### Parameters

  - indices: - indices: 2-D.  Indices of each input `SparseTensor`.
  - values: - values: 1-D.  Non-empty values of each `SparseTensor`.
  - shapes: - shapes: 1-D.  Shapes of each `SparseTensor`.

### `sparseCross(indices:_:shapes:denseInputs:hashedOutput:numBuckets:hashKey:internalType:)`

Generates sparse cross from a list of sparse and dense tensors.

``` swift
@inlinable @inline(__always) public static func sparseCross<SparseTypes: TensorArrayProtocol, DenseTypes: TensorArrayProtocol, OutType: TensorFlowIndex>(indices: [Tensor<Int64>], _ values: SparseTypes, shapes: [Tensor<Int64>], denseInputs: DenseTypes, hashedOutput: Bool, numBuckets: Int64, hashKey: Int64, internalType: TensorDataType) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<OutType>, outputShape: Tensor<Int64>)
```

The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
representing features of one feature column. It outputs a 2D `SparseTensor` with
the batchwise crosses of these features.

For example, if the inputs are

``` 
inputs[0]: SparseTensor with shape = [2, 2]
[0, 0]: "a"
[1, 0]: "b"
[1, 1]: "c"

inputs[1]: SparseTensor with shape = [2, 1]
[0, 0]: "d"
[1, 0]: "e"

inputs[2]: Tensor [["f"], ["g"]]
```

then the output will be

``` 
shape = [2, 2]
[0, 0]: "a_X_d_X_f"
[1, 0]: "b_X_e_X_g"
[1, 1]: "c_X_e_X_g"
```

if hashed\_output=true then the output will be

``` 
shape = [2, 2]
[0, 0]: FingerprintCat64(
            Fingerprint64("f"), FingerprintCat64(
                Fingerprint64("d"), Fingerprint64("a")))
[1, 0]: FingerprintCat64(
            Fingerprint64("g"), FingerprintCat64(
                Fingerprint64("e"), Fingerprint64("b")))
[1, 1]: FingerprintCat64(
            Fingerprint64("g"), FingerprintCat64(
                Fingerprint64("e"), Fingerprint64("c")))
```

#### Parameters

  - indices: - indices: 2-D.  Indices of each input `SparseTensor`.
  - values: - values: 1-D.   values of each `SparseTensor`.
  - shapes: - shapes: 1-D.   Shapes of each `SparseTensor`.

### `sparseCross(indices:_:shapes:denseInputs:hashedOutput:numBuckets:hashKey:internalType:)`

Generates sparse cross from a list of sparse and dense tensors.

``` swift
@inlinable @inline(__always) public static func sparseCross<SparseTypes: TensorArrayProtocol, DenseTypes: TensorArrayProtocol>(indices: [Tensor<Int64>], _ values: SparseTypes, shapes: [Tensor<Int64>], denseInputs: DenseTypes, hashedOutput: Bool, numBuckets: Int64, hashKey: Int64, internalType: TensorDataType) -> (outputIndices: Tensor<Int64>, outputValues: StringTensor, outputShape: Tensor<Int64>)
```

The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
representing features of one feature column. It outputs a 2D `SparseTensor` with
the batchwise crosses of these features.

For example, if the inputs are

``` 
inputs[0]: SparseTensor with shape = [2, 2]
[0, 0]: "a"
[1, 0]: "b"
[1, 1]: "c"

inputs[1]: SparseTensor with shape = [2, 1]
[0, 0]: "d"
[1, 0]: "e"

inputs[2]: Tensor [["f"], ["g"]]
```

then the output will be

``` 
shape = [2, 2]
[0, 0]: "a_X_d_X_f"
[1, 0]: "b_X_e_X_g"
[1, 1]: "c_X_e_X_g"
```

if hashed\_output=true then the output will be

``` 
shape = [2, 2]
[0, 0]: FingerprintCat64(
            Fingerprint64("f"), FingerprintCat64(
                Fingerprint64("d"), Fingerprint64("a")))
[1, 0]: FingerprintCat64(
            Fingerprint64("g"), FingerprintCat64(
                Fingerprint64("e"), Fingerprint64("b")))
[1, 1]: FingerprintCat64(
            Fingerprint64("g"), FingerprintCat64(
                Fingerprint64("e"), Fingerprint64("c")))
```

#### Parameters

  - indices: - indices: 2-D.  Indices of each input `SparseTensor`.
  - values: - values: 1-D.   values of each `SparseTensor`.
  - shapes: - shapes: 1-D.   Shapes of each `SparseTensor`.

### `sparseDenseCwiseAdd(spIndices:spValues:spShape:dense:)`

Adds up a SparseTensor and a dense Tensor, using these special rules:

``` swift
@inlinable @inline(__always) public static func sparseDenseCwiseAdd<T: TensorFlowNumeric>(spIndices: Tensor<Int64>, spValues: Tensor<T>, spShape: Tensor<Int64>, dense: Tensor<T>) -> Tensor<T>
```

(1) Broadcasts the dense side to have the same shape as the sparse side, if
eligible;
(2) Then, only the dense values pointed to by the indices of the SparseTensor
participate in the cwise addition.

By these rules, the result is a logical SparseTensor with exactly the same
indices and shape, but possibly with different non-zero values.  The output of
this Op is the resultant non-zero values.

#### Parameters

  - dense: - dense: `R`-D.  The dense Tensor operand.

### `sparseDenseCwiseDiv(spIndices:spValues:spShape:dense:)`

Component-wise divides a SparseTensor by a dense Tensor.

``` swift
@inlinable @inline(__always) public static func sparseDenseCwiseDiv<T: TensorFlowNumeric>(spIndices: Tensor<Int64>, spValues: Tensor<T>, spShape: Tensor<Int64>, dense: Tensor<T>) -> Tensor<T>
```

*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction.

#### Parameters

  - dense: - dense: `R`-D.  The dense Tensor operand.

### `sparseDenseCwiseMul(spIndices:spValues:spShape:dense:)`

Component-wise multiplies a SparseTensor by a dense Tensor.

``` swift
@inlinable @inline(__always) public static func sparseDenseCwiseMul<T: TensorFlowNumeric>(spIndices: Tensor<Int64>, spValues: Tensor<T>, spShape: Tensor<Int64>, dense: Tensor<T>) -> Tensor<T>
```

The output locations corresponding to the implicitly zero elements in the sparse
tensor will be zero (i.e., will not take up storage space), regardless of the
contents of the dense tensor (even if it's +/-INF and that INF\*0 == NaN).

*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction.

#### Parameters

  - dense: - dense: `R`-D.  The dense Tensor operand.

### `sparseFillEmptyRows(indices:_:denseShape:defaultValue:)`

Fills empty rows in the input 2-D `SparseTensor` with a default value.

``` swift
@inlinable @inline(__always) public static func sparseFillEmptyRows<T: TensorFlowScalar>(indices: Tensor<Int64>, _ values: Tensor<T>, denseShape: Tensor<Int64>, defaultValue: Tensor<T>) -> (
    outputIndices: Tensor<Int64>, outputValues: Tensor<T>, emptyRowIndicator: Tensor<Bool>,
    reverseIndexMap: Tensor<Int64>
  )
```

The input `SparseTensor` is represented via the tuple of inputs
(`indices`, `values`, `dense_shape`).  The output `SparseTensor` has the
same `dense_shape` but with indices `output_indices` and values
`output_values`.

This op inserts a single entry for every row that doesn't have any values.
The index is created as `[row, 0, ..., 0]` and the inserted value
is `default_value`.

For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:

``` 
[0, 1]: a
[0, 3]: b
[2, 0]: c
[3, 1]: d
```

Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:

``` 
[0, 1]: a
[0, 3]: b
[1, 0]: default_value
[2, 0]: c
[3, 1]: d
[4, 0]: default_value
```

The output `SparseTensor` will be in row-major order and will have the
same shape as the input.

This op also returns an indicator vector shaped `[dense_shape[0]]` such that

``` 
empty_row_indicator[i] = True iff row i was an empty row.
```

And a reverse index map vector shaped `[indices.shape[0]]` that is used during
backpropagation,

``` 
reverse_index_map[j] = out_j s.t. indices[j, :] == output_indices[out_j, :]
```

#### Parameters

  - indices: - indices: 2-D. the indices of the sparse tensor.
  - values: - values: 1-D. the values of the sparse tensor.

### `sparseFillEmptyRowsGrad(reverseIndexMap:gradValues:)`

The gradient of SparseFillEmptyRows.

``` swift
@inlinable @inline(__always) public static func sparseFillEmptyRowsGrad<T: TensorFlowScalar>(reverseIndexMap: Tensor<Int64>, gradValues: Tensor<T>) -> (dValues: Tensor<T>, dDefaultValue: Tensor<T>)
```

Takes vectors reverse\_index\_map, shaped `[N]`, and grad\_values,
shaped `[N_full]`, where `N_full >= N` and copies data into either
`d_values` or `d_default_value`.  Here `d_values` is shaped `[N]` and
`d_default_value` is a scalar.

d\_values\[j\] = grad\_values\[reverse\_index\_map\[j\]\]
d\_default\_value = sum\_{k : 0 .. N\_full - 1} (
grad\_values\[k\] \* 1{k not in reverse\_index\_map})

### `sparseMatMul(_:_:transposeA:transposeB:aIsSparse:bIsSparse:)`

Multiply matrix "a" by matrix "b".

``` swift
@inlinable @inline(__always) public static func sparseMatMul<Ta: FloatingPoint & TensorFlowScalar, Tb: FloatingPoint & TensorFlowScalar>(_ a: Tensor<Ta>, _ b: Tensor<Tb>, transposeA: Bool = false, transposeB: Bool = false, aIsSparse: Bool = false, bIsSparse: Bool = false) -> Tensor<Float>
```

The inputs must be two-dimensional matrices and the inner dimension of "a" must
match the outer dimension of "b". Both "a" and "b" must be `Tensor`s not
`SparseTensor`s.  This op is optimized for the case where at least one of "a" or
"b" is sparse, in the sense that they have a large proportion of zero values.
The breakeven for using this versus a dense matrix multiply on one platform was
30% zero values in the sparse matrix.

The gradient computation of this operation will only take advantage of sparsity
in the input gradient when that gradient comes from a Relu.

### `sparseMatrixAdd(_:_:alpha:beta:)`

Sparse addition of two CSR matrices, C = alpha \* A + beta \* B.

``` swift
@inlinable @inline(__always) public static func sparseMatrixAdd<T: FloatingPoint & TensorFlowScalar>(_ a: VariantHandle, _ b: VariantHandle, alpha: Tensor<T>, beta: Tensor<T>) -> VariantHandle
```

The gradients of SparseMatrixAdd outputs with respect to alpha and beta are not
currently defined (TensorFlow will return zeros for these entries).

#### Parameters

  - a: - a: A CSRSparseMatrix.
  - b: - b: A CSRSparseMatrix.
  - alpha: - alpha: A constant scalar.
  - beta: - beta: A constant scalar.

### `sparseMatrixMatMul(_:_:transposeA:transposeB:adjointA:adjointB:transposeOutput:conjugateOutput:)`

Matrix-multiplies a sparse matrix with a dense matrix.

``` swift
@inlinable @inline(__always) public static func sparseMatrixMatMul<T: TensorFlowScalar>(_ a: VariantHandle, _ b: Tensor<T>, transposeA: Bool = false, transposeB: Bool = false, adjointA: Bool = false, adjointB: Bool = false, transposeOutput: Bool = false, conjugateOutput: Bool = false) -> Tensor<T>
```

Returns a dense matrix.
For inputs A and B, where A is CSR and B is dense; this op returns a dense C;

If transpose\_output is false, returns:

``` 
  C = A . B
```

If transpose\_output is `true`, returns:

``` 
  C = transpose(A . B) = transpose(B) . transpose(A)
```

where the transposition is performed along the two innermost (matrix)
dimensions.

If conjugate\_output is `true`, returns:

``` 
  C = conjugate(A . B) = conjugate(A) . conjugate(B)
```

If both conjugate\_output and transpose\_output are `true`, returns:

``` 
  C = conjugate(transpose(A . B)) = conjugate(transpose(B)) .
                                    conjugate(transpose(A))
```

#### Parameters

  - a: - a: A CSRSparseMatrix.
  - b: - b: A dense tensor.

### `sparseMatrixMul(_:_:)`

Element-wise multiplication of a sparse matrix with a dense tensor.

``` swift
@inlinable @inline(__always) public static func sparseMatrixMul<T: TensorFlowScalar>(_ a: VariantHandle, _ b: Tensor<T>) -> VariantHandle
```

Returns a sparse matrix.

The dense tensor `b` may be either a scalar; otherwise `a` must be a rank-3
`SparseMatrix`; in this case `b` must be shaped `[batch_size, 1, 1]` and the
multiply operation broadcasts.

**NOTE** even if `b` is zero, the sparsity structure of the output does not
change.

#### Parameters

  - a: - a: A CSRSparseMatrix.
  - b: - b: A dense tensor.

### `sparseMatrixNNZ(sparseMatrix:)`

Returns the number of nonzeroes of `sparse_matrix`.

``` swift
@inlinable @inline(__always) public static func sparseMatrixNNZ(sparseMatrix: VariantHandle) -> Tensor<Int32>
```

### `sparseMatrixOrderingAMD(_:)`

Computes the Approximate Minimum Degree (AMD) ordering of `input`.

``` swift
@inlinable @inline(__always) public static func sparseMatrixOrderingAMD(_ input: VariantHandle) -> Tensor<Int32>
```

Computes the Approximate Minimum Degree (AMD) ordering for a sparse matrix.

The returned permutation may be used to permute the rows and columns of the
given sparse matrix. This typically results in permuted sparse matrix's sparse
Cholesky (or other decompositions) in having fewer zero fill-in compared to
decomposition of the original matrix.

The input sparse matrix may have rank 2 or rank 3. The output Tensor,
representing would then have rank 1 or 2 respectively, with the same batch
shape as the input.

Each component of the input sparse matrix must represent a square symmetric
matrix; only the lower triangular part of the matrix is read. The values of the
sparse matrix does not affect the returned permutation, only the sparsity
pattern of the sparse matrix is used. Hence, a single AMD ordering may be
reused for the Cholesky decompositions of sparse matrices with the same sparsity
pattern but with possibly different values.

Each batch component of the output permutation represents a permutation of `N`
elements, where the input sparse matrix components each have `N` rows. That is,
the component contains each of the integers `{0, .. N-1}` exactly once. The
`i`th element represents the row index that the `i`th row maps to.

Usage example:

``` python
    from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
 
    a_indices = np.array([[0, 0], [1, 1], [2, 1], [2, 2], [3, 3]])
    a_values = np.array([1.0, 2.0, 1.0, 3.0, 4.0], np.float32)
    a_dense_shape = [4, 4]
 
    with tf.Session() as sess:
      # Define (COO format) SparseTensor over Numpy array.
      a_st = tf.SparseTensor(a_indices, a_values, a_dense_shape)
 
      # Convert SparseTensors to CSR SparseMatrix.
      a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
          a_st.indices, a_st.values, a_st.dense_shape)
 
      # Obtain the AMD Ordering for the CSR SparseMatrix.
      ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(sparse_matrix)
 
      ordering_amd_value = sess.run(ordering_amd)
```

`ordering_amd_value` stores the AMD ordering: `[1 2 3 0]`.

input: A `CSRSparseMatrix`.

#### Parameters

  - input: - input: A `CSRSparseMatrix`.

### `sparseMatrixSoftmax(logits:type:)`

Calculates the softmax of a CSRSparseMatrix.

``` swift
@inlinable @inline(__always) public static func sparseMatrixSoftmax(logits: VariantHandle, type: TensorDataType) -> VariantHandle
```

Calculate the softmax of the innermost dimensions of a SparseMatrix.

Missing values are treated as `-inf` (i.e., logits of zero probability); and
the output has the same sparsity structure as the input (though missing values
in the output may now be treated as having probability zero).

#### Parameters

  - logits: - logits: A CSRSparseMatrix.

### `sparseMatrixSoftmaxGrad(softmax:gradSoftmax:type:)`

Calculates the gradient of the SparseMatrixSoftmax op.

``` swift
@inlinable @inline(__always) public static func sparseMatrixSoftmaxGrad(softmax: VariantHandle, gradSoftmax: VariantHandle, type: TensorDataType) -> VariantHandle
```

#### Parameters

  - softmax: - softmax: A CSRSparseMatrix.

### `sparseMatrixSparseCholesky(_:permutation:type:)`

Computes the sparse Cholesky decomposition of `input`.

``` swift
@inlinable @inline(__always) public static func sparseMatrixSparseCholesky(_ input: VariantHandle, permutation: Tensor<Int32>, type: TensorDataType) -> VariantHandle
```

Computes the Sparse Cholesky decomposition of a sparse matrix, with the given
fill-in reducing permutation.

The input sparse matrix and the fill-in reducing permutation `permutation` must
have compatible shapes. If the sparse matrix has rank 3; with the batch
dimension `B`, then the `permutation` must be of rank 2; with the same batch
dimension `B`. There is no support for broadcasting.

Furthermore, each component vector of `permutation` must be of length `N`,
containing each of the integers {0, 1, ..., N - 1} exactly once, where `N` is
the number of rows of each component of the sparse matrix.

Each component of the input sparse matrix must represent a symmetric positive
definite (SPD) matrix; although only the lower triangular part of the matrix is
read. If any individual component is not SPD, then an InvalidArgument error is
thrown.

The returned sparse matrix has the same dense shape as the input sparse matrix.
For each component `A` of the input sparse matrix, the corresponding output
sparse matrix represents `L`, the lower triangular Cholesky factor satisfying
the following identity:

``` 
  A = L * Lt
```

where Lt denotes the transpose of L (or its conjugate transpose, if `type` is
`complex64` or `complex128`).

The `type` parameter denotes the type of the matrix elements. The supported
types are: `float32`, `float64`, `complex64` and `complex128`.

Usage example:

``` python
    from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
 
    a_indices = np.array([[0, 0], [1, 1], [2, 1], [2, 2], [3, 3]])
    a_values = np.array([1.0, 2.0, 1.0, 3.0, 4.0], np.float32)
    a_dense_shape = [4, 4]
 
    with tf.Session() as sess:
      # Define (COO format) SparseTensor over Numpy array.
      a_st = tf.SparseTensor(a_indices, a_values, a_dense_shape)
 
      # Convert SparseTensors to CSR SparseMatrix.
      a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
          a_st.indices, a_st.values, a_st.dense_shape)
 
      # Obtain the Sparse Cholesky factor using AMD Ordering for reducing zero
      # fill-in (number of structural non-zeros in the sparse Cholesky factor).
      ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(sparse_matrix)
      cholesky_sparse_matrices = (
          sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(
              sparse_matrix, ordering_amd, type=tf.float32))
 
      # Convert the CSRSparseMatrix Cholesky factor to a dense Tensor
      dense_cholesky = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          cholesky_sparse_matrices, tf.float32)
 
      # Evaluate the dense Tensor value.
      dense_cholesky_value = sess.run(dense_cholesky)
```

`dense_cholesky_value` stores the dense Cholesky factor:

``` 
    [[  1.  0.    0.    0.]
     [  0.  1.41  0.    0.]
     [  0.  0.70  1.58  0.]
     [  0.  0.    0.    2.]]
```

input: A `CSRSparseMatrix`.
permutation: A `Tensor`.
type: The type of `input`.

#### Parameters

  - input: - input: A `CSRSparseMatrix`.
  - permutation: - permutation: A fill-in reducing permutation matrix.

### `sparseMatrixSparseMatMul(_:_:type:transposeA:transposeB:adjointA:adjointB:)`

Sparse-matrix-multiplies two CSR matrices `a` and `b`.

``` swift
@inlinable @inline(__always) public static func sparseMatrixSparseMatMul(_ a: VariantHandle, _ b: VariantHandle, type: TensorDataType, transposeA: Bool = false, transposeB: Bool = false, adjointA: Bool = false, adjointB: Bool = false) -> VariantHandle
```

Performs a matrix multiplication of a sparse matrix `a` with a sparse matrix
`b`; returns a sparse matrix `a * b`, unless either `a` or `b` is transposed or
adjointed.

Each matrix may be transposed or adjointed (conjugated and transposed)
according to the Boolean parameters `transpose_a`, `adjoint_a`, `transpose_b`
and `adjoint_b`. At most one of `transpose_a` or `adjoint_a` may be True.
Similarly, at most one of `transpose_b` or `adjoint_b` may be True.

The inputs must have compatible shapes. That is, the inner dimension of `a`
must be equal to the outer dimension of `b`. This requirement is adjusted
according to whether either `a` or `b` is transposed or adjointed.

The `type` parameter denotes the type of the matrix elements. Both `a` and `b`
must have the same type. The supported types are: `float32`, `float64`,
`complex64` and `complex128`.

Both `a` and `b` must have the same rank. Broadcasting is not supported. If they
have rank 3, each batch of 2D CSRSparseMatrices within `a` and `b` must have the
same dense shape.

The sparse matrix product may have numeric (non-structural) zeros.
TODO(anudhyan): Consider adding a boolean attribute to control whether to prune
zeros.

Usage example:

``` python
    from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
 
    a_indices = np.array([[0, 0], [2, 3], [2, 4], [3, 0]])
    a_values = np.array([1.0, 5.0, -1.0, -2.0], np.float32)
    a_dense_shape = [4, 5]
 
    b_indices = np.array([[0, 0], [3, 0], [3, 1]])
    b_values = np.array([2.0, 7.0, 8.0], np.float32)
    b_dense_shape = [5, 3]
 
    with tf.Session() as sess:
      # Define (COO format) Sparse Tensors over Numpy arrays
      a_st = tf.SparseTensor(a_indices, a_values, a_dense_shape)
      b_st = tf.SparseTensor(b_indices, b_values, b_dense_shape)
 
      # Convert SparseTensors to CSR SparseMatrix
      a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
          a_st.indices, a_st.values, a_st.dense_shape)
      b_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
          b_st.indices, b_st.values, b_st.dense_shape)
 
      # Compute the CSR SparseMatrix matrix multiplication
      c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
          a=a_sm, b=b_sm, type=tf.float32)
 
      # Convert the CSR SparseMatrix product to a dense Tensor
      c_sm_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
          c_sm, tf.float32)
      # Evaluate the dense Tensor value
      c_sm_dense_value = sess.run(c_sm_dense)
```

`c_sm_dense_value` stores the dense matrix product:

``` 
    [[  2.   0.   0.]
     [  0.   0.   0.]
     [ 35.  40.   0.]
     [ -4.   0.   0.]]
```

a: A `CSRSparseMatrix`.
b: A `CSRSparseMatrix` with the same type and rank as `a`.
type: The type of both `a` and `b`.
transpose\_a: If True, `a` transposed before multiplication.
transpose\_b: If True, `b` transposed before multiplication.
adjoint\_a: If True, `a` adjointed before multiplication.
adjoint\_b: If True, `b` adjointed before multiplication.

#### Parameters

  - a: - a: A CSRSparseMatrix.
  - b: - b: A CSRSparseMatrix.

### `sparseMatrixTranspose(_:conjugate:type:)`

Transposes the inner (matrix) dimensions of a CSRSparseMatrix.

``` swift
@inlinable @inline(__always) public static func sparseMatrixTranspose(_ input: VariantHandle, conjugate: Bool = false, type: TensorDataType) -> VariantHandle
```

Transposes the inner (matrix) dimensions of a SparseMatrix and optionally
conjugates its values.

#### Parameters

  - input: - input: A CSRSparseMatrix.

### `sparseMatrixZeros(denseShape:type:)`

Creates an all-zeros CSRSparseMatrix with shape `dense_shape`.

``` swift
@inlinable @inline(__always) public static func sparseMatrixZeros(denseShape: Tensor<Int64>, type: TensorDataType) -> VariantHandle
```

### `sparseReduceMax(inputIndices:inputValues:inputShape:reductionAxes:keepDims:)`

Computes the max of elements across dimensions of a SparseTensor.

``` swift
@inlinable @inline(__always) public static func sparseReduceMax<T: TensorFlowNumeric>(inputIndices: Tensor<Int64>, inputValues: Tensor<T>, inputShape: Tensor<Int64>, reductionAxes: Tensor<Int32>, keepDims: Bool = false) -> Tensor<T>
```

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
instead of a sparse one.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.

### `sparseReduceMaxSparse(inputIndices:inputValues:inputShape:reductionAxes:keepDims:)`

Computes the max of elements across dimensions of a SparseTensor.

``` swift
@inlinable @inline(__always) public static func sparseReduceMaxSparse<T: TensorFlowNumeric>(inputIndices: Tensor<Int64>, inputValues: Tensor<T>, inputShape: Tensor<Int64>, reductionAxes: Tensor<Int32>, keepDims: Bool = false) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>)
```

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_max()`.  In contrast to SparseReduceMax, this Op returns a
SparseTensor.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.

### `sparseReduceSum(inputIndices:inputValues:inputShape:reductionAxes:keepDims:)`

Computes the sum of elements across dimensions of a SparseTensor.

``` swift
@inlinable @inline(__always) public static func sparseReduceSum<T: TensorFlowNumeric>(inputIndices: Tensor<Int64>, inputValues: Tensor<T>, inputShape: Tensor<Int64>, reductionAxes: Tensor<Int32>, keepDims: Bool = false) -> Tensor<T>
```

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
instead of a sparse one.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.

### `sparseReduceSumSparse(inputIndices:inputValues:inputShape:reductionAxes:keepDims:)`

Computes the sum of elements across dimensions of a SparseTensor.

``` swift
@inlinable @inline(__always) public static func sparseReduceSumSparse<T: TensorFlowNumeric>(inputIndices: Tensor<Int64>, inputValues: Tensor<T>, inputShape: Tensor<Int64>, reductionAxes: Tensor<Int32>, keepDims: Bool = false) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>)
```

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
SparseTensor.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.

### `sparseReorder(inputIndices:inputValues:inputShape:)`

Reorders a SparseTensor into the canonical, row-major ordering.

``` swift
@inlinable @inline(__always) public static func sparseReorder<T: TensorFlowScalar>(inputIndices: Tensor<Int64>, inputValues: Tensor<T>, inputShape: Tensor<Int64>) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>)
```

Note that by convention, all sparse ops preserve the canonical ordering along
increasing dimension number. The only time ordering can be violated is during
manual manipulation of the indices and values vectors to add entries.

Reordering does not affect the shape of the SparseTensor.

If the tensor has rank `R` and `N` non-empty values, `input_indices` has
shape `[N, R]`, input\_values has length `N`, and input\_shape has length `R`.

### `sparseReshape(inputIndices:inputShape:newShape:)`

Reshapes a SparseTensor to represent values in a new dense shape.

``` swift
@inlinable @inline(__always) public static func sparseReshape(inputIndices: Tensor<Int64>, inputShape: Tensor<Int64>, newShape: Tensor<Int64>) -> (outputIndices: Tensor<Int64>, outputShape: Tensor<Int64>)
```

This operation has the same semantics as reshape on the represented dense
tensor.  The `input_indices` are recomputed based on the requested `new_shape`.

If one component of `new_shape` is the special value -1, the size of that
dimension is computed so that the total dense size remains constant.  At
most one component of `new_shape` can be -1.  The number of dense elements
implied by `new_shape` must be the same as the number of dense elements
originally implied by `input_shape`.

Reshaping does not affect the order of values in the SparseTensor.

If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
has length `R_out`, then `input_indices` has shape `[N, R_in]`,
`input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
`output_shape` has length `R_out`.

### `sparseSegmentMean(data:indices:segmentIds:)`

Computes the mean along sparse segments of a tensor.

``` swift
@inlinable @inline(__always) public static func sparseSegmentMean<T: FloatingPoint & TensorFlowScalar, Tidx: TensorFlowIndex>(data: Tensor<T>, indices: Tensor<Tidx>, segmentIds: Tensor<Int32>) -> Tensor<T>
```

See `tf.sparse.segment_sum` for usage examples.

Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`.

#### Parameters

  - indices: - indices: A 1-D tensor. Has same rank as `segment_ids`.

### `sparseSegmentMeanGrad(grad:indices:segmentIds:outputDim0:)`

Computes gradients for SparseSegmentMean.

``` swift
@inlinable @inline(__always) public static func sparseSegmentMeanGrad<T: FloatingPoint & TensorFlowScalar, Tidx: TensorFlowIndex>(grad: Tensor<T>, indices: Tensor<Tidx>, segmentIds: Tensor<Int32>, outputDim0: Tensor<Int32>) -> Tensor<T>
```

Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output\_dim0.

#### Parameters

  - grad: - grad: gradient propagated to the SparseSegmentMean op.
  - indices: - indices: indices passed to the corresponding SparseSegmentMean op.

### `sparseSegmentMeanWithNumSegments(data:indices:segmentIds:numSegments:)`

Computes the mean along sparse segments of a tensor.

``` swift
@inlinable @inline(__always) public static func sparseSegmentMeanWithNumSegments<T: FloatingPoint & TensorFlowScalar, Tidx: TensorFlowIndex, Tnumsegments: TensorFlowIndex>(data: Tensor<T>, indices: Tensor<Tidx>, segmentIds: Tensor<Int32>, numSegments: Tensor<Tnumsegments>) -> Tensor<T>
```

Like `SparseSegmentMean`, but allows missing ids in `segment_ids`. If an id is
misisng, the `output` tensor at that position will be zeroed.

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

#### Parameters

  - indices: - indices: A 1-D tensor. Has same rank as `segment_ids`.

### `sparseSegmentSqrtN(data:indices:segmentIds:)`

Computes the sum along sparse segments of a tensor divided by the sqrt of N.

``` swift
@inlinable @inline(__always) public static func sparseSegmentSqrtN<T: FloatingPoint & TensorFlowScalar, Tidx: TensorFlowIndex>(data: Tensor<T>, indices: Tensor<Tidx>, segmentIds: Tensor<Int32>) -> Tensor<T>
```

N is the size of the segment being reduced.

See `tf.sparse.segment_sum` for usage examples.

#### Parameters

  - indices: - indices: A 1-D tensor. Has same rank as `segment_ids`.

### `sparseSegmentSqrtNGrad(grad:indices:segmentIds:outputDim0:)`

Computes gradients for SparseSegmentSqrtN.

``` swift
@inlinable @inline(__always) public static func sparseSegmentSqrtNGrad<T: FloatingPoint & TensorFlowScalar, Tidx: TensorFlowIndex>(grad: Tensor<T>, indices: Tensor<Tidx>, segmentIds: Tensor<Int32>, outputDim0: Tensor<Int32>) -> Tensor<T>
```

Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output\_dim0.

#### Parameters

  - grad: - grad: gradient propagated to the SparseSegmentSqrtN op.
  - indices: - indices: indices passed to the corresponding SparseSegmentSqrtN op.

### `sparseSegmentSqrtNWithNumSegments(data:indices:segmentIds:numSegments:)`

Computes the sum along sparse segments of a tensor divided by the sqrt of N.

``` swift
@inlinable @inline(__always) public static func sparseSegmentSqrtNWithNumSegments<T: FloatingPoint & TensorFlowScalar, Tidx: TensorFlowIndex, Tnumsegments: TensorFlowIndex>(data: Tensor<T>, indices: Tensor<Tidx>, segmentIds: Tensor<Int32>, numSegments: Tensor<Tnumsegments>) -> Tensor<T>
```

N is the size of the segment being reduced.

Like `SparseSegmentSqrtN`, but allows missing ids in `segment_ids`. If an id is
misisng, the `output` tensor at that position will be zeroed.

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

#### Parameters

  - indices: - indices: A 1-D tensor. Has same rank as `segment_ids`.

### `sparseSegmentSum(data:indices:segmentIds:)`

Computes the sum along sparse segments of a tensor.

``` swift
@inlinable @inline(__always) public static func sparseSegmentSum<T: TensorFlowNumeric, Tidx: TensorFlowIndex>(data: Tensor<T>, indices: Tensor<Tidx>, segmentIds: Tensor<Int32>) -> Tensor<T>
```

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension 0, specified by `indices`.

For example:

``` python
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
 
# Select two rows, one segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
# => [[0 0 0 0]]
 
# Select two rows, two segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
# => [[ 1  2  3  4]
#     [-1 -2 -3 -4]]
 
# Select all rows, two segments.
tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
# => [[0 0 0 0]
#     [5 6 7 8]]
 
# Which is equivalent to:
tf.segment_sum(c, tf.constant([0, 0, 1]))
```

#### Parameters

  - indices: - indices: A 1-D tensor. Has same rank as `segment_ids`.

### `sparseSegmentSumWithNumSegments(data:indices:segmentIds:numSegments:)`

Computes the sum along sparse segments of a tensor.

``` swift
@inlinable @inline(__always) public static func sparseSegmentSumWithNumSegments<T: TensorFlowNumeric, Tidx: TensorFlowIndex, Tnumsegments: TensorFlowIndex>(data: Tensor<T>, indices: Tensor<Tidx>, segmentIds: Tensor<Int32>, numSegments: Tensor<Tnumsegments>) -> Tensor<T>
```

Like `SparseSegmentSum`, but allows missing ids in `segment_ids`. If an id is
misisng, the `output` tensor at that position will be zeroed.

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/sparse#Segmentation)
for an explanation of segments.

For example:

``` python
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
 
tf.sparse_segment_sum_with_num_segments(
    c, tf.constant([0, 1]), tf.constant([0, 0]), num_segments=3)
# => [[0 0 0 0]
#     [0 0 0 0]
#     [0 0 0 0]]
 
tf.sparse_segment_sum_with_num_segments(c,
                                        tf.constant([0, 1]),
                                        tf.constant([0, 2],
                                        num_segments=4))
# => [[ 1  2  3  4]
#     [ 0  0  0  0]
#     [-1 -2 -3 -4]
#     [ 0  0  0  0]]
```

#### Parameters

  - indices: - indices: A 1-D tensor. Has same rank as `segment_ids`.

### `sparseSlice(indices:_:shape:start:size:)`

Slice a `SparseTensor` based on the `start` and `size`.

``` swift
@inlinable @inline(__always) public static func sparseSlice<T: TensorFlowScalar>(indices: Tensor<Int64>, _ values: Tensor<T>, shape: Tensor<Int64>, start: Tensor<Int64>, size: Tensor<Int64>) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>, outputShape: Tensor<Int64>)
```

For example, if the input is

``` 
input_tensor = shape = [2, 7]
[    a   d e  ]
[b c          ]
```

Graphically the output tensors are:

``` 
sparse_slice([0, 0], [2, 4]) = shape = [2, 4]
[    a  ]
[b c    ]

sparse_slice([0, 4], [2, 3]) = shape = [2, 3]
[ d e  ]
[      ]
```

#### Parameters

  - indices: - indices: 2-D tensor represents the indices of the sparse tensor.
  - values: - values: 1-D tensor represents the values of the sparse tensor.
  - shape: - shape: 1-D. tensor represents the shape of the sparse tensor.
  - start: - start: 1-D. tensor represents the start of the slice.
  - size: - size: 1-D. tensor represents the size of the slice. output indices: A list of 1-D tensors represents the indices of the output sparse tensors.

### `sparseSliceGrad(backpropValGrad:inputIndices:inputStart:outputIndices:)`

The gradient operator for the SparseSlice op.

``` swift
@inlinable @inline(__always) public static func sparseSliceGrad<T: TensorFlowNumeric>(backpropValGrad: Tensor<T>, inputIndices: Tensor<Int64>, inputStart: Tensor<Int64>, outputIndices: Tensor<Int64>) -> Tensor<T>
```

This op takes in the upstream gradient w.r.t. non-empty values of
the sliced `SparseTensor`, and outputs the gradients w.r.t.
the non-empty values of input `SparseTensor`.

### `sparseSoftmax(spIndices:spValues:spShape:)`

Applies softmax to a batched N-D `SparseTensor`.

``` swift
@inlinable @inline(__always) public static func sparseSoftmax<T: FloatingPoint & TensorFlowScalar>(spIndices: Tensor<Int64>, spValues: Tensor<T>, spShape: Tensor<Int64>) -> Tensor<T>
```

The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
(where `N >= 2`), and with indices sorted in the canonical lexicographic order.

This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
logical submatrix with shape `[B, C]`, but with the catch that *the implicitly
zero elements do not participate*.  Specifically, the algorithm is equivalent
to the following:

(1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
with shape `[B, C]`, along the size-C dimension;
(2) Masks out the original implicitly-zero locations;
(3) Renormalizes the remaining elements.

Hence, the `SparseTensor` result has exactly the same non-zero indices and
shape.

### `sparseSoftmaxCrossEntropyWithLogits(features:labels:)`

Computes softmax cross entropy cost and gradients to backpropagate.

``` swift
@inlinable @inline(__always) public static func sparseSoftmaxCrossEntropyWithLogits<T: FloatingPoint & TensorFlowScalar, Tlabels: TensorFlowIndex>(features: Tensor<T>, labels: Tensor<Tlabels>) -> (loss: Tensor<T>, backprop: Tensor<T>)
```

Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
a matrix of label probabilities, but rather a single label per row
of features.  This label is considered to have probability 1.0 for the
given row.

Inputs are the logits, not probabilities.

#### Parameters

  - features: - features: batch\_size x num\_classes matrix
  - labels: - labels: batch\_size vector with values in \[0, num\_classes). This is the label for the given minibatch entry.

### `sparseSparseMaximum(aIndices:aValues:aShape:bIndices:bValues:bShape:)`

Returns the element-wise max of two SparseTensors.

``` swift
@inlinable @inline(__always) public static func sparseSparseMaximum<T: TensorFlowNumeric>(aIndices: Tensor<Int64>, aValues: Tensor<T>, aShape: Tensor<Int64>, bIndices: Tensor<Int64>, bValues: Tensor<T>, bShape: Tensor<Int64>) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>)
```

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

### `sparseSparseMinimum(aIndices:aValues:aShape:bIndices:bValues:bShape:)`

Returns the element-wise min of two SparseTensors.

``` swift
@inlinable @inline(__always) public static func sparseSparseMinimum<T: TensorFlowNumeric>(aIndices: Tensor<Int64>, aValues: Tensor<T>, aShape: Tensor<Int64>, bIndices: Tensor<Int64>, bValues: Tensor<T>, bShape: Tensor<Int64>) -> (outputIndices: Tensor<Int64>, outputValues: Tensor<T>)
```

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

### `sparseSplit(splitDim:indices:_:shape:numSplit:)`

Split a `SparseTensor` into `num_split` tensors along one dimension.

``` swift
@inlinable @inline(__always) public static func sparseSplit<T: TensorFlowScalar>(splitDim: Tensor<Int64>, indices: Tensor<Int64>, _ values: Tensor<T>, shape: Tensor<Int64>, numSplit: Int64) -> (outputIndices: [Tensor<Int64>], outputValues: [Tensor<T>], outputShape: [Tensor<Int64>])
```

If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
`[0 : shape[split_dim] % num_split]` gets one extra dimension.
For example, if `split_dim = 1` and `num_split = 2` and the input is

``` 
input_tensor = shape = [2, 7]
[    a   d e  ]
[b c          ]
```

Graphically the output tensors are:

``` 
output_tensor[0] = shape = [2, 4]
[    a  ]
[b c    ]

output_tensor[1] = shape = [2, 3]
[ d e  ]
[      ]
```

#### Parameters

  - indices: - indices: 2-D tensor represents the indices of the sparse tensor.
  - values: - values: 1-D tensor represents the values of the sparse tensor.
  - shape: - shape: 1-D. tensor represents the shape of the sparse tensor. output indices: A list of 1-D tensors represents the indices of the output sparse tensors.

### `sparseTensorDenseAdd(aIndices:aValues:aShape:_:)`

Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.

``` swift
@inlinable @inline(__always) public static func sparseTensorDenseAdd<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(aIndices: Tensor<Tindices>, aValues: Tensor<T>, aShape: Tensor<Tindices>, _ b: Tensor<T>) -> Tensor<T>
```

This Op does not require `a_indices` be sorted in standard lexicographic order.

#### Parameters

  - b: - b: `ndims`-D Tensor.  With shape `a_shape`.

### `sparseTensorDenseMatMul(aIndices:aValues:aShape:_:adjointA:adjointB:)`

Multiply SparseTensor (of rank 2) "A" by dense matrix "B".

``` swift
@inlinable @inline(__always) public static func sparseTensorDenseMatMul<T: TensorFlowScalar, Tindices: TensorFlowIndex>(aIndices: Tensor<Tindices>, aValues: Tensor<T>, aShape: Tensor<Int64>, _ b: Tensor<T>, adjointA: Bool = false, adjointB: Bool = false) -> Tensor<T>
```

No validity checking is performed on the indices of A.  However, the following
input format is recommended for optimal behavior:

if adjoint\_a == false:
A should be sorted in lexicographically increasing order.  Use SparseReorder
if you're not sure.
if adjoint\_a == true:
A should be sorted in order of increasing dimension 1 (i.e., "column major"
order instead of "row major" order).

#### Parameters

  - b: - b: 2-D.  A dense Matrix.

### `sparseTensorSliceDataset(indices:_:denseShape:)`

Creates a dataset that splits a SparseTensor into elements row-wise.

``` swift
@inlinable @inline(__always) public static func sparseTensorSliceDataset<Tvalues: TensorFlowScalar>(indices: Tensor<Int64>, _ values: Tensor<Tvalues>, denseShape: Tensor<Int64>) -> VariantHandle
```

### `sparseTensorToCSRSparseMatrix(indices:_:denseShape:)`

Converts a SparseTensor to a (possibly batched) CSRSparseMatrix.

``` swift
@inlinable @inline(__always) public static func sparseTensorToCSRSparseMatrix<T: FloatingPoint & TensorFlowScalar>(indices: Tensor<Int64>, _ values: Tensor<T>, denseShape: Tensor<Int64>) -> VariantHandle
```

#### Parameters

  - indices: - indices: SparseTensor indices.
  - values: - values: SparseTensor values.

### `sparseToDense(sparseIndices:outputShape:sparseValues:defaultValue:validateIndices:)`

Converts a sparse representation into a dense tensor.

``` swift
@inlinable @inline(__always) public static func sparseToDense<T: TensorFlowScalar, Tindices: TensorFlowIndex>(sparseIndices: Tensor<Tindices>, outputShape: Tensor<Tindices>, sparseValues: Tensor<T>, defaultValue: Tensor<T>, validateIndices: Bool = true) -> Tensor<T>
```

Builds an array `dense` with shape `output_shape` such that

``` 
# If sparse_indices is scalar
dense[i] = (i == sparse_indices ? sparse_values : default_value)
 
# If sparse_indices is a vector, then for each i
dense[sparse_indices[i]] = sparse_values[i]
 
# If sparse_indices is an n by d matrix, then for each i in [0, n)
dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
```

All other values in `dense` are set to `default_value`.  If `sparse_values` is a
scalar, all sparse indices are set to this single value.

Indices should be sorted in lexicographic order, and indices must not
contain any repeats. If `validate_indices` is true, these properties
are checked during execution.

### `sparseToSparseSetOperation(set1Indices:set1Values:set1Shape:set2Indices:set2Values:set2Shape:setOperation:validateIndices:)`

Applies set operation along last dimension of 2 `SparseTensor` inputs.

``` swift
@inlinable @inline(__always) public static func sparseToSparseSetOperation<T: TensorFlowInteger>(set1Indices: Tensor<Int64>, set1Values: Tensor<T>, set1Shape: Tensor<Int64>, set2Indices: Tensor<Int64>, set2Values: Tensor<T>, set2Shape: Tensor<Int64>, setOperation: String, validateIndices: Bool = true) -> (resultIndices: Tensor<Int64>, resultValues: Tensor<T>, resultShape: Tensor<Int64>)
```

See SetOperationOp::SetOperationFromContext for values of `set_operation`.

If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
order and range of `set1` and `set2` indices.

Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

If `validate_indices` is `True`, this op validates the order and range of `set1`
and `set2` indices.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

### `sparseToSparseSetOperation(set1Indices:set1Values:set1Shape:set2Indices:set2Values:set2Shape:setOperation:validateIndices:)`

Applies set operation along last dimension of 2 `SparseTensor` inputs.

``` swift
@inlinable @inline(__always) public static func sparseToSparseSetOperation(set1Indices: Tensor<Int64>, set1Values: StringTensor, set1Shape: Tensor<Int64>, set2Indices: Tensor<Int64>, set2Values: StringTensor, set2Shape: Tensor<Int64>, setOperation: String, validateIndices: Bool = true) -> (resultIndices: Tensor<Int64>, resultValues: StringTensor, resultShape: Tensor<Int64>)
```

See SetOperationOp::SetOperationFromContext for values of `set_operation`.

If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
order and range of `set1` and `set2` indices.

Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
ignored.

If `validate_indices` is `True`, this op validates the order and range of `set1`
and `set2` indices.

Output `result` is a `SparseTensor` represented by `result_indices`,
`result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
dimension contains the result of `set_operation` applied to the corresponding
`[0...n-1]` dimension of `set`.

### `split(splitDim:value:numSplit:)`

Splits a tensor into `num_split` tensors along one dimension.

``` swift
@inlinable @inline(__always) public static func split<T: TensorFlowScalar>(splitDim: Tensor<Int32>, value: Tensor<T>, numSplit: Int64) -> [Tensor<T>]
```

#### Parameters

  - value: - value: The tensor to split.

### `splitV(value:sizeSplits:splitDim:numSplit:)`

Splits a tensor into `num_split` tensors along one dimension.

``` swift
@inlinable @inline(__always) public static func splitV<T: TensorFlowScalar, Tlen: TensorFlowIndex>(value: Tensor<T>, sizeSplits: Tensor<Tlen>, splitDim: Tensor<Int32>, numSplit: Int64) -> [Tensor<T>]
```

#### Parameters

  - value: - value: The tensor to split.

### `sqlDataset(driverName:dataSourceName:query:outputTypes:outputShapes:)`

Creates a dataset that executes a SQL query and emits rows of the result set.

``` swift
@inlinable @inline(__always) public static func sqlDataset(driverName: StringTensor, dataSourceName: StringTensor, query: StringTensor, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

#### Parameters

  - query: - query: A SQL query to execute.

### `sqrt(_:)`

Computes square root of x element-wise.

``` swift
@inlinable @inline(__always) public static func sqrt<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

I.e., \\(y = \\sqrt{x} = x^{1/2}\\).

### `sqrtGrad(_:dy:)`

Computes the gradient for the sqrt of `x` wrt its input.

``` swift
@inlinable @inline(__always) public static func sqrtGrad<T: FloatingPoint & TensorFlowScalar>(_ y: Tensor<T>, dy: Tensor<T>) -> Tensor<T>
```

Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
is the corresponding input gradient.

### `square(_:)`

Computes square of x element-wise.

``` swift
@inlinable @inline(__always) public static func square<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

I.e., \\(y = x \* x = x^2\\).

### `squaredDifference(_:_:)`

Returns (x - y)(x - y) element-wise.

``` swift
@inlinable @inline(__always) public static func squaredDifference<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `squeeze(_:squeezeDims:)`

Removes dimensions of size 1 from the shape of a tensor.

``` swift
@inlinable @inline(__always) public static func squeeze<T: TensorFlowScalar>(_ input: Tensor<T>, squeezeDims: [Int32]) -> Tensor<T>
```

Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`axis`.

For example:

``` 
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
```

Or, to remove specific size 1 dimensions:

``` 
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
```

#### Parameters

  - input: - input: The `input` to squeeze.

### `stackCloseV2(handle:)`

Delete the stack from its resource container.

``` swift
@inlinable @inline(__always) public static func stackCloseV2(handle: ResourceHandle)
```

#### Parameters

  - handle: - handle: The handle to a stack.

### `stackPopV2(handle:)`

Pop the element at the top of the stack.

``` swift
@inlinable @inline(__always) public static func stackPopV2<ElemType: TensorFlowScalar>(handle: ResourceHandle) -> Tensor<ElemType>
```

#### Parameters

  - handle: - handle: The handle to a stack.

### `stackPushV2(handle:elem:swapMemory:)`

Push an element onto the stack.

``` swift
@inlinable @inline(__always) public static func stackPushV2<T: TensorFlowScalar>(handle: ResourceHandle, elem: Tensor<T>, swapMemory: Bool = false) -> Tensor<T>
```

#### Parameters

  - handle: - handle: The handle to a stack.
  - elem: - elem: The tensor to be pushed onto the stack.

### `stackV2(maxSize:elemType:stackName:)`

A stack that produces elements in first-in last-out order.

``` swift
@inlinable @inline(__always) public static func stackV2(maxSize: Tensor<Int32>, elemType: TensorDataType, stackName: String) -> ResourceHandle
```

### `stage(_:capacity:memoryLimit:container:sharedName:)`

Stage values similar to a lightweight Enqueue.

``` swift
@inlinable @inline(__always) public static func stage<Dtypes: TensorArrayProtocol>(_ values: Dtypes, capacity: Int64 = 0, memoryLimit: Int64 = 0, container: String, sharedName: String)
```

The basic functionality of this Op is similar to a queue with many
fewer capabilities and options.  This Op is optimized for performance.

#### Parameters

  - values: - values: a list of tensors dtypes A list of data types that inserted values should adhere to.

### `stageClear(capacity:memoryLimit:dtypes:container:sharedName:)`

Op removes all elements in the underlying container.

``` swift
@inlinable @inline(__always) public static func stageClear(capacity: Int64 = 0, memoryLimit: Int64 = 0, dtypes: [TensorDataType], container: String, sharedName: String)
```

### `stagePeek(index:capacity:memoryLimit:container:sharedName:)`

Op peeks at the values at the specified index.  If the

``` swift
@inlinable @inline(__always) public static func stagePeek<Dtypes: TensorGroup>(index: Tensor<Int32>, capacity: Int64 = 0, memoryLimit: Int64 = 0, container: String, sharedName: String) -> Dtypes
```

underlying container does not contain sufficient elements
this op will block until it does.   This Op is optimized for
performance.

### `stageSize(capacity:memoryLimit:dtypes:container:sharedName:)`

Op returns the number of elements in the underlying container.

``` swift
@inlinable @inline(__always) public static func stageSize(capacity: Int64 = 0, memoryLimit: Int64 = 0, dtypes: [TensorDataType], container: String, sharedName: String) -> Tensor<Int32>
```

### `statefulPartitionedCall(args:f:config:configProto:executorType:)`

returns `f(inputs)`, where `f`'s body is placed and partitioned.

``` swift
@inlinable @inline(__always) public static func statefulPartitionedCall<Tin: TensorArrayProtocol, Tout: TensorGroup, FIn: TensorGroup, FOut: TensorGroup>(args: Tin, f: (FIn) -> FOut, config: String, configProto: String, executorType: String) -> Tout
```

#### Parameters

  - args: - args: A list of input tensors.

### `statefulRandomBinomial(resource:algorithm:shape:counts:probs:)`

``` swift
@inlinable @inline(__always) public static func statefulRandomBinomial<S: TensorFlowIndex, T: TensorFlowNumeric, Dtype: TensorFlowNumeric>(resource: ResourceHandle, algorithm: Tensor<Int64>, shape: Tensor<S>, counts: Tensor<T>, probs: Tensor<T>) -> Tensor<Dtype>
```

### `statefulStandardNormal(resource:shape:)`

Outputs random values from a normal distribution. This op is deprecated in favor of op 'StatefulStandardNormalV2'

``` swift
@inlinable @inline(__always) public static func statefulStandardNormal<Dtype: TensorFlowScalar, ShapeDtype: TensorFlowScalar>(resource: ResourceHandle, shape: Tensor<ShapeDtype>) -> Tensor<Dtype>
```

The generated values will have mean 0 and standard deviation 1.

#### Parameters

  - resource: - resource: The handle of the resource variable that stores the state of the RNG.
  - shape: - shape: The shape of the output tensor.

### `statefulStandardNormalV2(resource:algorithm:shape:)`

Outputs random values from a normal distribution.

``` swift
@inlinable @inline(__always) public static func statefulStandardNormalV2<Dtype: TensorFlowScalar, ShapeDtype: TensorFlowScalar>(resource: ResourceHandle, algorithm: Tensor<Int64>, shape: Tensor<ShapeDtype>) -> Tensor<Dtype>
```

The generated values will have mean 0 and standard deviation 1.

#### Parameters

  - resource: - resource: The handle of the resource variable that stores the state of the RNG.
  - algorithm: - algorithm: The RNG algorithm.
  - shape: - shape: The shape of the output tensor.

### `statefulTruncatedNormal(resource:algorithm:shape:)`

Outputs random values from a truncated normal distribution.

``` swift
@inlinable @inline(__always) public static func statefulTruncatedNormal<Dtype: TensorFlowScalar, ShapeDtype: TensorFlowScalar>(resource: ResourceHandle, algorithm: Tensor<Int64>, shape: Tensor<ShapeDtype>) -> Tensor<Dtype>
```

The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.

#### Parameters

  - resource: - resource: The handle of the resource variable that stores the state of the RNG.
  - algorithm: - algorithm: The RNG algorithm.
  - shape: - shape: The shape of the output tensor.

### `statefulUniform(resource:algorithm:shape:)`

Outputs random values from a uniform distribution.

``` swift
@inlinable @inline(__always) public static func statefulUniform<Dtype: TensorFlowScalar, ShapeDtype: TensorFlowScalar>(resource: ResourceHandle, algorithm: Tensor<Int64>, shape: Tensor<ShapeDtype>) -> Tensor<Dtype>
```

The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

#### Parameters

  - resource: - resource: The handle of the resource variable that stores the state of the RNG.
  - algorithm: - algorithm: The RNG algorithm.
  - shape: - shape: The shape of the output tensor.

### `statefulUniformFullInt(resource:algorithm:shape:)`

Outputs random integers from a uniform distribution.

``` swift
@inlinable @inline(__always) public static func statefulUniformFullInt<Dtype: TensorFlowScalar, ShapeDtype: TensorFlowScalar>(resource: ResourceHandle, algorithm: Tensor<Int64>, shape: Tensor<ShapeDtype>) -> Tensor<Dtype>
```

The generated values are uniform integers covering the whole range of `dtype`.

#### Parameters

  - resource: - resource: The handle of the resource variable that stores the state of the RNG.
  - algorithm: - algorithm: The RNG algorithm.
  - shape: - shape: The shape of the output tensor.

### `statefulUniformInt(resource:algorithm:shape:minval:maxval:)`

Outputs random integers from a uniform distribution.

``` swift
@inlinable @inline(__always) public static func statefulUniformInt<Dtype: TensorFlowScalar, ShapeDtype: TensorFlowScalar>(resource: ResourceHandle, algorithm: Tensor<Int64>, shape: Tensor<ShapeDtype>, minval: Tensor<Dtype>, maxval: Tensor<Dtype>) -> Tensor<Dtype>
```

The generated values are uniform integers in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound
`maxval` is excluded.

The random integers are slightly biased unless `maxval - minval` is an exact
power of two.  The bias is small for values of `maxval - minval` significantly
smaller than the range of the output (either `2^32` or `2^64`).

#### Parameters

  - resource: - resource: The handle of the resource variable that stores the state of the RNG.
  - algorithm: - algorithm: The RNG algorithm.
  - shape: - shape: The shape of the output tensor.
  - minval: - minval: Minimum value (inclusive, scalar).
  - maxval: - maxval: Maximum value (exclusive, scalar).

### `statelessIf(cond:_:thenBranch:elseBranch:outputShapes:)`

output = cond ? then\_branch(input) : else\_branch(input)

``` swift
@inlinable @inline(__always) public static func statelessIf<Tcond: TensorFlowScalar, Tin: TensorArrayProtocol, Tout: TensorGroup, ThenbranchIn: TensorGroup, ThenbranchOut: TensorGroup, ElsebranchIn: TensorGroup, ElsebranchOut: TensorGroup>(cond: Tensor<Tcond>, _ input: Tin, thenBranch: (ThenbranchIn) -> ThenbranchOut, elseBranch: (ElsebranchIn) -> ElsebranchOut, outputShapes: [TensorShape?]) -> Tout
```

#### Parameters

  - cond: - cond: \`\`\`
    This should only be used when the if then/else body functions do not
    have stateful ops.
    ``` 
    ```
  - input: - input: A list of input tensors.

### `statelessMultinomial(logits:numSamples:seed:)`

Draws samples from a multinomial distribution.

``` swift
@inlinable @inline(__always) public static func statelessMultinomial<T: TensorFlowNumeric, Tseed: TensorFlowIndex, OutputDtype: TensorFlowIndex>(logits: Tensor<T>, numSamples: Tensor<Int32>, seed: Tensor<Tseed>) -> Tensor<OutputDtype>
```

#### Parameters

  - logits: - logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]` represents the unnormalized log probabilities for all classes.
  - seed: - seed: 2 seeds (shape \[2\]).

### `statelessRandomNormal(shape:seed:)`

Outputs deterministic pseudorandom values from a normal distribution.

``` swift
@inlinable @inline(__always) public static func statelessRandomNormal<Dtype: FloatingPoint & TensorFlowScalar, T: TensorFlowIndex, Tseed: TensorFlowIndex>(shape: Tensor<T>, seed: Tensor<Tseed>) -> Tensor<Dtype>
```

The generated values will have mean 0 and standard deviation 1.

The outputs are a deterministic function of `shape` and `seed`.

#### Parameters

  - shape: - shape: The shape of the output tensor.
  - seed: - seed: 2 seeds (shape \[2\]).

### `statelessRandomUniform(shape:seed:)`

Outputs deterministic pseudorandom random values from a uniform distribution.

``` swift
@inlinable @inline(__always) public static func statelessRandomUniform<Dtype: FloatingPoint & TensorFlowScalar, T: TensorFlowIndex, Tseed: TensorFlowIndex>(shape: Tensor<T>, seed: Tensor<Tseed>) -> Tensor<Dtype>
```

The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

The outputs are a deterministic function of `shape` and `seed`.

#### Parameters

  - shape: - shape: The shape of the output tensor.
  - seed: - seed: 2 seeds (shape \[2\]).

### `statelessRandomUniformInt(shape:seed:minval:maxval:)`

Outputs deterministic pseudorandom random integers from a uniform distribution.

``` swift
@inlinable @inline(__always) public static func statelessRandomUniformInt<Dtype: TensorFlowIndex, T: TensorFlowIndex, Tseed: TensorFlowIndex>(shape: Tensor<T>, seed: Tensor<Tseed>, minval: Tensor<Dtype>, maxval: Tensor<Dtype>) -> Tensor<Dtype>
```

The generated values follow a uniform distribution in the range `[minval, maxval)`.

The outputs are a deterministic function of `shape`, `seed`, `minval`, and `maxval`.

#### Parameters

  - shape: - shape: The shape of the output tensor.
  - seed: - seed: 2 seeds (shape \[2\]).
  - minval: - minval: Minimum value (inclusive, scalar).
  - maxval: - maxval: Maximum value (exclusive, scalar).

### `statelessTruncatedNormal(shape:seed:)`

Outputs deterministic pseudorandom values from a truncated normal distribution.

``` swift
@inlinable @inline(__always) public static func statelessTruncatedNormal<Dtype: FloatingPoint & TensorFlowScalar, T: TensorFlowIndex, Tseed: TensorFlowIndex>(shape: Tensor<T>, seed: Tensor<Tseed>) -> Tensor<Dtype>
```

The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.

The outputs are a deterministic function of `shape` and `seed`.

#### Parameters

  - shape: - shape: The shape of the output tensor.
  - seed: - seed: 2 seeds (shape \[2\]).

### `statelessWhile(_:cond:body:outputShapes:parallelIterations:)`

output = input; While (Cond(output)) { output = Body(output) }

``` swift
@inlinable @inline(__always) public static func statelessWhile<T: TensorArrayProtocol, CondIn: TensorGroup, CondOut: TensorGroup, BodyIn: TensorGroup, BodyOut: TensorGroup>(_ input: T, cond: (CondIn) -> CondOut, body: (BodyIn) -> BodyOut, outputShapes: [TensorShape?], parallelIterations: Int64 = 10) -> T
```

#### Parameters

  - input: - input: A list of input tensors whose types are T.

### `staticRegexFullMatch(_:pattern:)`

Check if the input matches the regex pattern.

``` swift
@inlinable @inline(__always) public static func staticRegexFullMatch(_ input: StringTensor, pattern: String) -> Tensor<Bool>
```

The input is a string tensor of any shape. The pattern is the
regular expression to be matched with every element of the input tensor.
The boolean values (True or False) of the output tensor indicate
if the input matches the regex pattern provided.

The pattern follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)

#### Parameters

  - input: - input: A string tensor of the text to be processed.

### `staticRegexReplace(_:pattern:rewrite:replaceGlobal:)`

Replaces the match of pattern in input with rewrite.

``` swift
@inlinable @inline(__always) public static func staticRegexReplace(_ input: StringTensor, pattern: String, rewrite: String, replaceGlobal: Bool = true) -> StringTensor
```

It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)

#### Parameters

  - input: - input: The text to be processed.

### `statsAggregatorHandle(container:sharedName:)`

Creates a statistics manager resource.

``` swift
@inlinable @inline(__always) public static func statsAggregatorHandle(container: String, sharedName: String) -> ResourceHandle
```

### `statsAggregatorHandleV2(container:sharedName:)`

``` swift
@inlinable @inline(__always) public static func statsAggregatorHandleV2(container: String, sharedName: String) -> ResourceHandle
```

### `statsAggregatorSetSummaryWriter(statsAggregator:summary:)`

Set a summary\_writer\_interface to record statistics using given stats\_aggregator.

``` swift
@inlinable @inline(__always) public static func statsAggregatorSetSummaryWriter(statsAggregator: ResourceHandle, summary: ResourceHandle)
```

### `statsAggregatorSummary(iterator:)`

Produces a summary of any statistics recorded by the given statistics manager.

``` swift
@inlinable @inline(__always) public static func statsAggregatorSummary(iterator: ResourceHandle) -> StringTensor
```

### `stopGradient(_:)`

Stops gradient computation.

``` swift
@inlinable @inline(__always) public static func stopGradient<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

When executed in a graph, this op outputs its input tensor as-is.

When building ops to compute gradients, this op prevents the contribution of
its inputs to be taken into account.  Normally, the gradient generator adds ops
to a graph to compute the derivatives of a specified 'loss' by recursively
finding out inputs that contributed to its computation.  If you insert this op
in the graph it inputs are masked from the gradient generator.  They are not
taken into account for computing gradients.

This is useful any time you want to compute a value with TensorFlow but need
to pretend that the value was a constant. Some examples include:

### `stridedSlice(_:begin:end:strides:beginMask:endMask:ellipsisMask:newAxisMask:shrinkAxisMask:)`

Return a strided slice from `input`.

``` swift
@inlinable @inline(__always) public static func stridedSlice<T: TensorFlowScalar, Index: TensorFlowIndex>(_ input: Tensor<T>, begin: Tensor<Index>, end: Tensor<Index>, strides: Tensor<Index>, beginMask: Int64 = 0, endMask: Int64 = 0, ellipsisMask: Int64 = 0, newAxisMask: Int64 = 0, shrinkAxisMask: Int64 = 0) -> Tensor<T>
```

Note, most python users will want to use the Python `Tensor.__getitem__`
or `Variable.__getitem__` rather than this op directly.

The goal of this op is to produce a new tensor with a subset of
the elements from the `n` dimensional `input` tensor. The subset is chosen using
a sequence of `m` sparse range specifications encoded into the arguments
of this function. Note, in some cases
`m` could be equal to `n`, but this need not be the case. Each
range specification entry can be one of the following:

Each conceptual range specification is encoded in the op's argument. This
encoding is best understand by considering a non-trivial example. In
particular,
`foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as

``` 
begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
end = [2, 4, x, x, -3, x]
strides = [1, 1, x, x, -1, 1]
begin_mask = 1<<4 | 1 << 5 = 48
end_mask = 1<<5 = 32
ellipsis_mask = 1<<3 = 8
new_axis_mask = 1<<2 4
shrink_axis_mask = 1<<0
```

In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
the slice becomes (2, 1, 5, 5, 2, 5).
Let us walk step by step through each argument specification.

1.  The first argument in the example slice is turned into `begin = 1` and
    `end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
    also set the appropriate bit in `shrink_axis_mask`.

2.  `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
    zero bits contributed.

3.  None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
    dimension in the final shape. Dummy values are contributed to begin,
    end and stride, while the new\_axis\_mask bit is set.

4.  `...` grab the full ranges from as many dimensions as needed to
    fully specify a slice for every dimension of the input shape.

5.  `:-3:-1` shows the use of negative indices. A negative index `i` associated
    with a dimension that has shape `s` is converted to a positive index
    `s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
    is done internally so begin, end and strides receive x, -3, and -1.
    The appropriate begin\_mask bit is set to indicate the start range is the
    full range (ignoring the x).

6.  `:` indicates that the entire contents of the corresponding dimension
    is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
    receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
    `end_mask` are also set.

*Requirements*:
`0 != strides[i] for i in [0, m)`
`ellipsis_mask must be a power of two (only one ellipsis)`

#### Parameters

  - begin: - begin: `begin[k]` specifies the offset into the `k`th range specification. The exact dimension this corresponds to will be determined by context. Out-of-bounds values will be silently clamped. If the `k`th bit of `begin_mask` then `begin[k]` is ignored and the full range of the appropriate dimension is used instead. Negative values causes indexing to start from the highest element e.g. If `foo==[1,2,3]` then `foo[-1]==3`.
  - end: - end: `end[i]` is like `begin` with the exception that `end_mask` is used to determine full ranges.
  - strides: - strides: `strides[i]` specifies the increment in the `i`th specification after extracting a given element. Negative indices will reverse the original order. Out or range values are clamped to `[0,dim[i]) if slice[i]>0` or `[-1,dim[i]-1] if slice[i] < 0`

### `stridedSliceGrad(shape:begin:end:strides:dy:beginMask:endMask:ellipsisMask:newAxisMask:shrinkAxisMask:)`

Returns the gradient of `StridedSlice`.

``` swift
@inlinable @inline(__always) public static func stridedSliceGrad<T: TensorFlowScalar, Index: TensorFlowIndex>(shape: Tensor<Index>, begin: Tensor<Index>, end: Tensor<Index>, strides: Tensor<Index>, dy: Tensor<T>, beginMask: Int64 = 0, endMask: Int64 = 0, ellipsisMask: Int64 = 0, newAxisMask: Int64 = 0, shrinkAxisMask: Int64 = 0) -> Tensor<T>
```

Since `StridedSlice` cuts out pieces of its `input` which is size
`shape`, its gradient will have the same shape (which is passed here
as `shape`). The gradient will be zero in any element that the slice
does not select.

Arguments are the same as StridedSliceGrad with the exception that
`dy` is the input gradient to be propagated and `shape` is the
shape of `StridedSlice`'s `input`.

### `stringFormat(inputs:template:placeholder:summarize:)`

Formats a string template using a list of tensors.

``` swift
@inlinable @inline(__always) public static func stringFormat<T: TensorArrayProtocol>(inputs: T, template: String = "%s", placeholder: String = "%s", summarize: Int64 = 3) -> StringTensor
```

Formats a string template using a list of tensors, pretty-printing tensor summaries.

#### Parameters

  - inputs: - inputs: The list of tensors to format into the placeholder string.

### `stringJoin(inputs:separator:)`

Joins the strings in the given list of string tensors into one tensor;

``` swift
@inlinable @inline(__always) public static func stringJoin(inputs: [StringTensor], separator: String) -> StringTensor
```

with the given separator (default is an empty separator).

#### Parameters

  - inputs: - inputs: A list of string tensors.  The tensors must all have the same shape, or be scalars.  Scalars may be mixed in; these will be broadcast to the shape of non-scalar inputs.

### `stringLength(_:unit:)`

String lengths of `input`.

``` swift
@inlinable @inline(__always) public static func stringLength(_ input: StringTensor, unit: Unit = .byte) -> Tensor<Int32>
```

Computes the length of each string given in the input tensor.

> > > strings = tf.constant(\['Hello','TensorFlow', '\\U0001F642'\])
> > > tf.strings.length(strings).numpy() \# default counts bytes
> > > array(\[ 5, 10, 4\], dtype=int32)
> > > tf.strings.length(strings, unit="UTF8\_CHAR").numpy()
> > > array(\[ 5, 10, 1\], dtype=int32)

#### Parameters

  - input: - input: The strings for which to compute the length for each element.

### `stringListAttr(_:_:)`

``` swift
@inlinable @inline(__always) public static func stringListAttr(_ a: [String], _ b: String)
```

### `stringLower(_:encoding:)`

Converts each string in the input Tensor to lowercase.

``` swift
@inlinable @inline(__always) public static func stringLower(_ input: StringTensor, encoding: String) -> StringTensor
```

### `stringNGrams(data:dataSplits:separator:ngramWidths:leftPad:rightPad:padWidth:preserveShortSequences:)`

Creates ngrams from ragged string data.

``` swift
@inlinable @inline(__always) public static func stringNGrams<Tsplits: TensorFlowIndex>(data: StringTensor, dataSplits: Tensor<Tsplits>, separator: String, ngramWidths: [Int32], leftPad: String, rightPad: String, padWidth: Int64, preserveShortSequences: Bool) -> (ngrams: StringTensor, ngramsSplits: Tensor<Tsplits>)
```

This op accepts a ragged tensor with 1 ragged dimension containing only
strings and outputs a ragged tensor with 1 ragged dimension containing ngrams
of that string, joined along the innermost axis.

#### Parameters

  - data: - data: The values tensor of the ragged string tensor to make ngrams out of. Must be a 1D string tensor.

### `stringSplit(_:delimiter:skipEmpty:)`

Split elements of `input` based on `delimiter` into a `SparseTensor`.

``` swift
@inlinable @inline(__always) public static func stringSplit(_ input: StringTensor, delimiter: StringTensor, skipEmpty: Bool = true) -> (indices: Tensor<Int64>, values: StringTensor, shape: Tensor<Int64>)
```

Let N be the size of source (typically N will be the batch size). Split each
element of `input` based on `delimiter` and return a `SparseTensor`
containing the splitted tokens. Empty tokens are ignored.

`delimiter` can be empty, or a string of split characters. If `delimiter` is an
empty string, each element of `input` is split into individual single-byte
character strings, including splitting of UTF-8 multibyte sequences. Otherwise
every character of `delimiter` is a potential split point.

For example:
N = 2, input\[0\] is 'hello world' and input\[1\] is 'a b c', then the output
will be

indices = \[0, 0;
0, 1;
1, 0;
1, 1;
1, 2\]
shape = \[2, 3\]
values = \['hello', 'world', 'a', 'b', 'c'\]

#### Parameters

  - input: - input: 1-D. Strings to split.
  - delimiter: - delimiter: 0-D. Delimiter characters (bytes), or empty string.

### `stringSplitV2(_:sep:maxsplit:)`

Split elements of `source` based on `sep` into a `SparseTensor`.

``` swift
@inlinable @inline(__always) public static func stringSplitV2(_ input: StringTensor, sep: StringTensor, maxsplit: Int64 = -1) -> (indices: Tensor<Int64>, values: StringTensor, shape: Tensor<Int64>)
```

Let N be the size of source (typically N will be the batch size). Split each
element of `source` based on `sep` and return a `SparseTensor`
containing the split tokens. Empty tokens are ignored.

For example, N = 2, source\[0\] is 'hello world' and source\[1\] is 'a b c',
then the output will be

``` 
st.indices = [0, 0;
              0, 1;
              1, 0;
              1, 1;
              1, 2]
st.shape = [2, 3]
st.values = ['hello', 'world', 'a', 'b', 'c']
```

If `sep` is given, consecutive delimiters are not grouped together and are
deemed to delimit empty strings. For example, source of `"1<>2<><>3"` and
sep of `"<>"` returns `["1", "2", "", "3"]`. If `sep` is None or an empty
string, consecutive whitespace are regarded as a single separator, and the
result will contain no empty strings at the startor end if the string has
leading or trailing whitespace.

Note that the above mentioned behavior matches python's str.split.

#### Parameters

  - input: - input: `1-D` string `Tensor`, the strings to split.
  - sep: - sep: `0-D` string `Tensor`, the delimiter character.

### `stringStrip(_:)`

Strip leading and trailing whitespaces from the Tensor.

``` swift
@inlinable @inline(__always) public static func stringStrip(_ input: StringTensor) -> StringTensor
```

#### Parameters

  - input: - input: A string `Tensor` of any shape.

### `stringToHashBucket(stringTensor:numBuckets:)`

Converts each string in the input Tensor to its hash mod by a number of buckets.

``` swift
@inlinable @inline(__always) public static func stringToHashBucket(stringTensor: StringTensor, numBuckets: Int64) -> Tensor<Int64>
```

The hash function is deterministic on the content of the string within the
process.

Note that the hash function may change from time to time.
This functionality will be deprecated and it's recommended to use
`tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.

### `stringToHashBucketFast(_:numBuckets:)`

Converts each string in the input Tensor to its hash mod by a number of buckets.

``` swift
@inlinable @inline(__always) public static func stringToHashBucketFast(_ input: StringTensor, numBuckets: Int64) -> Tensor<Int64>
```

The hash function is deterministic on the content of the string within the
process and will never change. However, it is not suitable for cryptography.
This function may be used when CPU time is scarce and inputs are trusted or
unimportant. There is a risk of adversaries constructing inputs that all hash
to the same bucket. To prevent this problem, use a strong hash function with
`tf.string_to_hash_bucket_strong`.

#### Parameters

  - input: - input: The strings to assign a hash bucket.

### `stringToHashBucketStrong(_:numBuckets:key:)`

Converts each string in the input Tensor to its hash mod by a number of buckets.

``` swift
@inlinable @inline(__always) public static func stringToHashBucketStrong(_ input: StringTensor, numBuckets: Int64, key: [Int32]) -> Tensor<Int64>
```

The hash function is deterministic on the content of the string within the
process. The hash function is a keyed hash function, where attribute `key`
defines the key of the hash function. `key` is an array of 2 elements.

A strong hash is important when inputs may be malicious, e.g. URLs with
additional components. Adversaries could try to make their inputs hash to the
same bucket for a denial-of-service attack or to skew the results. A strong
hash can be used to make it difficult to find inputs with a skewed hash value
distribution over buckets. This requires that the hash function is
seeded by a high-entropy (random) "key" unknown to the adversary.

The additional robustness comes at a cost of roughly 4x higher compute
time than `tf.string_to_hash_bucket_fast`.

#### Parameters

  - input: - input: The strings to assign a hash bucket.

### `stringToNumber(stringTensor:)`

Converts each string in the input Tensor to the specified numeric type.

``` swift
@inlinable @inline(__always) public static func stringToNumber<OutType: TensorFlowNumeric>(stringTensor: StringTensor) -> Tensor<OutType>
```

(Note that int32 overflow results in an error while float overflow
results in a rounded value.)

Example:

> > > strings = \["5.0", "3.0", "7.0"\]
> > > tf.strings.to\_number(strings)
> > > \<tf.Tensor: shape=(3,), dtype=float32, numpy=array(\[5., 3., 7.\], dtype=float32)\>

### `stringUpper(_:encoding:)`

Converts each string in the input Tensor to uppercase.

``` swift
@inlinable @inline(__always) public static func stringUpper(_ input: StringTensor, encoding: String) -> StringTensor
```

### `stubResourceHandleOp(container:sharedName:)`

``` swift
@inlinable @inline(__always) public static func stubResourceHandleOp(container: String, sharedName: String) -> ResourceHandle
```

### `sub(_:_:)`

Returns x - y element-wise.

``` swift
@inlinable @inline(__always) public static func sub<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

*NOTE*: `Subtract` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `substr(_:pos:len:unit:)`

Return substrings from `Tensor` of strings.

``` swift
@inlinable @inline(__always) public static func substr<T: TensorFlowIndex>(_ input: StringTensor, pos: Tensor<T>, len: Tensor<T>, unit: Unit = .byte) -> StringTensor
```

For each string in the input `Tensor`, creates a substring starting at index
`pos` with a total length of `len`.

If `len` defines a substring that would extend beyond the length of the input
string, or if `len` is negative, then as many characters as possible are used.

A negative `pos` indicates distance within the string backwards from the end.

If `pos` specifies an index which is out of range for any of the input strings,
then an `InvalidArgumentError` is thrown.

`pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
Op creation.

*NOTE*: `Substr` supports broadcasting up to two dimensions. More about
broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Examples

Using scalar `pos` and `len`:

``` python
input = [b'Hello', b'World']
position = 1
length = 3
 
output = [b'ell', b'orl']
```

Using `pos` and `len` with same shape as `input`:

``` python
input = [[b'ten', b'eleven', b'twelve'],
         [b'thirteen', b'fourteen', b'fifteen'],
         [b'sixteen', b'seventeen', b'eighteen']]
position = [[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]]
length =   [[2, 3, 4],
            [4, 3, 2],
            [5, 5, 5]]
 
output = [[b'en', b'eve', b'lve'],
          [b'hirt', b'urt', b'te'],
          [b'ixtee', b'vente', b'hteen']]
```

Broadcasting `pos` and `len` onto `input`:

``` 
input = [[b'ten', b'eleven', b'twelve'],
         [b'thirteen', b'fourteen', b'fifteen'],
         [b'sixteen', b'seventeen', b'eighteen'],
         [b'nineteen', b'twenty', b'twentyone']]
position = [1, 2, 3]
length =   [1, 2, 3]
 
output = [[b'e', b'ev', b'lve'],
          [b'h', b'ur', b'tee'],
          [b'i', b've', b'hte'],
          [b'i', b'en', b'nty']]
```

Broadcasting `input` onto `pos` and `len`:

``` 
input = b'thirteen'
position = [1, 5, 7]
length =   [3, 2, 1]
 
output = [b'hir', b'ee', b'n']
```

#### Parameters

  - input: - input: Tensor of strings
  - pos: - pos: Scalar defining the position of first character in each substring
  - len: - len: Scalar defining the number of characters to include in each substring

### `sum(_:reductionIndices:keepDims:)`

Computes the sum of elements across dimensions of a tensor.

``` swift
@inlinable @inline(__always) public static func sum<T: TensorFlowNumeric, Tidx: TensorFlowIndex>(_ input: Tensor<T>, reductionIndices: Tensor<Tidx>, keepDims: Bool = false) -> Tensor<T>
```

Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

#### Parameters

  - input: - input: The tensor to reduce.

### `summaryWriter(sharedName:container:)`

``` swift
@inlinable @inline(__always) public static func summaryWriter(sharedName: String, container: String) -> ResourceHandle
```

### `svd(_:computeUv:fullMatrices:)`

Computes the singular value decompositions of one or more matrices.

``` swift
@inlinable @inline(__always) public static func svd<T: FloatingPoint & TensorFlowScalar>(_ input: Tensor<T>, computeUv: Bool = true, fullMatrices: Bool = false) -> (s: Tensor<T>, u: Tensor<T>, v: Tensor<T>)
```

Computes the SVD of each inner matrix in `input` such that
`input[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :, :])`

``` python
# a is a tensor containing a batch of matrices.
# s is a tensor of singular values for each matrix.
# u is the tensor containing the left singular vectors for each matrix.
# v is the tensor containing the right singular vectors for each matrix.
s, u, v = svd(a)
s, _, _ = svd(a, compute_uv=False)
```

#### Parameters

  - input: - input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.

### `switch_(data:pred:)`

Forwards `data` to the output port determined by `pred`.

``` swift
@inlinable @inline(__always) public static func switch_<T: TensorFlowScalar>(data: Tensor<T>, pred: Tensor<Bool>) -> (outputFalse: Tensor<T>, outputTrue: Tensor<T>)
```

If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.

See also `RefSwitch` and `Merge`.

#### Parameters

  - data: - data: The tensor to be forwarded to the appropriate output.
  - pred: - pred: A scalar that specifies which output port will receive data.

### `symbolicGradient(_:f:)`

Computes the gradient function for function f via backpropagation.

``` swift
@inlinable @inline(__always) public static func symbolicGradient<Tin: TensorArrayProtocol, Tout: TensorGroup, FIn: TensorGroup, FOut: TensorGroup>(_ input: Tin, f: (FIn) -> FOut) -> Tout
```

#### Parameters

  - input: - input: a list of input tensors of size N + M;

### `tFRecordDataset(filenames:compressionType:bufferSize:)`

Creates a dataset that emits the records from one or more TFRecord files.

``` swift
@inlinable @inline(__always) public static func tFRecordDataset(filenames: StringTensor, compressionType: StringTensor, bufferSize: Tensor<Int64>) -> VariantHandle
```

#### Parameters

  - filenames: - filenames: A scalar or vector containing the name(s) of the file(s) to be read.

### `tFRecordReaderV2(container:sharedName:compressionType:)`

A Reader that outputs the records from a TensorFlow Records file.

``` swift
@inlinable @inline(__always) public static func tFRecordReaderV2(container: String, sharedName: String, compressionType: String) -> ResourceHandle
```

### `tPUCompilationResult()`

Returns the result of a TPU compilation.

``` swift
@inlinable @inline(__always) public static func tPUCompilationResult() -> StringTensor
```

This operation returns the result of a TPU compilation as a serialized
CompilationResultProto, which holds a status and an error message if an error
occurred during compilation.

### `tPUEmbeddingActivations(embeddingVariable:slicedActivations:tableId:lookupId:)`

An op enabling differentiation of TPU Embeddings.

``` swift
@inlinable @inline(__always) public static func tPUEmbeddingActivations(embeddingVariable: Tensor<Float>, slicedActivations: Tensor<Float>, tableId: Int64, lookupId: Int64) -> Tensor<Float>
```

This op simply returns its first input, which is assumed to have been sliced
from the Tensors returned by TPUEmbeddingDequeueActivations. The presence of
this op, and its first argument being a trainable Variable, enables automatic
differentiation of graphs containing embeddings via the TPU Embedding Python
libraries.

### `tPUOrdinalSelector()`

A TPU core selector Op.

``` swift
@inlinable @inline(__always) public static func tPUOrdinalSelector() -> Tensor<Int32>
```

This Op produces a set of TPU cores (for warm-up) or a single TPU core
(for regular inference) to execute the TPU program on. The output is
consumed by TPUPartitionedCall.

### `tPUPartitionedCall(args:deviceOrdinal:f:autotunerThresh:)`

Calls a function placed on a specified TPU device.

``` swift
@inlinable @inline(__always) public static func tPUPartitionedCall<Tin: TensorArrayProtocol, Tout: TensorGroup, FIn: TensorGroup, FOut: TensorGroup>(args: Tin, deviceOrdinal: Tensor<Int32>, f: (FIn) -> FOut, autotunerThresh: Int64 = 0) -> Tout
```

#### Parameters

  - args: - args: The arguments to the function.

### `tPUReplicateMetadata(numReplicas:numCoresPerReplica:topology:useTpu:deviceAssignment:computationShape:hostComputeCore:paddingMap:stepMarkerLocation:allowSoftPlacement:)`

Metadata indicating how the TPU computation should be replicated.

``` swift
@inlinable @inline(__always) public static func tPUReplicateMetadata(numReplicas: Int64, numCoresPerReplica: Int64 = 1, topology: String, useTpu: Bool = true, deviceAssignment: [Int32], computationShape: [Int32], hostComputeCore: [String], paddingMap: [String], stepMarkerLocation: String = "STEP_MARK_AT_ENTRY", allowSoftPlacement: Bool = false)
```

This operation holds the metadata common to operations of a `tpu.replicate()` computation subgraph.

### `tPUReplicatedInput(inputs:isMirroredVariable:index:)`

Connects N inputs to an N-way replicated TPU computation.

``` swift
@inlinable @inline(__always) public static func tPUReplicatedInput<T: TensorFlowScalar>(inputs: [Tensor<T>], isMirroredVariable: Bool = false, index: Int64 = -1) -> Tensor<T>
```

This operation holds a replicated input to a `tpu.replicate()` computation subgraph.
Each replicated input has the same shape and type alongside the output.

For example:

``` 
%a = "tf.opA"()
%b = "tf.opB"()
%replicated_input = "tf.TPUReplicatedInput"(%a, %b)
%computation = "tf.Computation"(%replicated_input)
```

The above computation has a replicated input of two replicas.

### `tPUReplicatedOutput(_:numReplicas:)`

Connects N outputs from an N-way replicated TPU computation.

``` swift
@inlinable @inline(__always) public static func tPUReplicatedOutput<T: TensorFlowScalar>(_ input: Tensor<T>, numReplicas: Int64) -> [Tensor<T>]
```

This operation holds a replicated output from a `tpu.replicate()` computation subgraph.
Each replicated output has the same shape and type alongside the input.

For example:

``` 
%computation = "tf.Computation"()
%replicated_output:2 = "tf.TPUReplicatedOutput"(%computation)
```

The above computation has a replicated output of two replicas.

### `takeDataset(inputDataset:count:outputTypes:outputShapes:)`

Creates a dataset that contains `count` elements from the `input_dataset`.

``` swift
@inlinable @inline(__always) public static func takeDataset(inputDataset: VariantHandle, count: Tensor<Int64>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

#### Parameters

  - count: - count: A scalar representing the number of elements from the `input_dataset` that should be taken. A value of `-1` indicates that all of `input_dataset` is taken.

### `takeManySparseFromTensorsMap(sparseHandles:container:sharedName:)`

Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.

``` swift
@inlinable @inline(__always) public static func takeManySparseFromTensorsMap<Dtype: TensorFlowScalar>(sparseHandles: Tensor<Int64>, container: String, sharedName: String) -> (sparseIndices: Tensor<Int64>, sparseValues: Tensor<Dtype>, sparseShape: Tensor<Int64>)
```

The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
`N` is the minibatch size and the rows correspond to the output handles of
`AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
original `SparseTensor` objects that went into the given input ops must all
match.  When the final `SparseTensor` is created, it has rank one
higher than the ranks of the incoming `SparseTensor` objects
(they have been concatenated along a new row dimension on the left).

The output `SparseTensor` object's shape values for all dimensions but the
first are the max across the input `SparseTensor` objects' shape values
for the corresponding dimensions.  Its first shape value is `N`, the minibatch
size.

The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.

For example, if the handles represent an input, which is a `[2, 3]` matrix
representing two original `SparseTensor` objects:

``` 
    index = [ 0]
            [10]
            [20]
    values = [1, 2, 3]
    shape = [50]
```

and

``` 
    index = [ 2]
            [10]
    values = [4, 5]
    shape = [30]
```

then the final `SparseTensor` will be:

``` 
    index = [0  0]
            [0 10]
            [0 20]
            [1  2]
            [1 10]
    values = [1, 2, 3, 4, 5]
    shape = [2 50]
```

### `takeWhileDataset(inputDataset:otherArguments:predicate:outputTypes:outputShapes:)`

Creates a dataset that stops iteration when predicate\` is false.

``` swift
@inlinable @inline(__always) public static func takeWhileDataset<PredicateIn: TensorGroup, PredicateOut: TensorGroup, Targuments: TensorArrayProtocol>(inputDataset: VariantHandle, otherArguments: Targuments, predicate: (PredicateIn) -> PredicateOut, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

The `predicate` function must return a scalar boolean and accept the
following arguments:

### `tan(_:)`

Computes tan of x element-wise.

``` swift
@inlinable @inline(__always) public static func tan<T: TensorFlowNumeric>(_ x: Tensor<T>) -> Tensor<T>
```

Given an input tensor, this function computes tangent of every
element in the tensor. Input range is `(-inf, inf)` and
output range is `(-inf, inf)`. If input lies outside the boundary, `nan`
is returned.

``` python
x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
tf.math.tan(x) ==> [nan 0.45231566 -0.5463025 1.5574077 2.572152 -1.7925274 0.32097113 nan]
```

### `tanh(_:)`

Computes hyperbolic tangent of `x` element-wise.

``` swift
@inlinable @inline(__always) public static func tanh<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

Given an input tensor, this function computes hyperbolic tangent of every
element in the tensor. Input range is `[-inf, inf]` and
output range is `[-1,1]`.

``` python
x = tf.constant([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")])
tf.math.tanh(x) ==> [-1. -0.99990916 -0.46211717 0.7615942 0.8336547 0.9640276 0.9950547 1.]
```

### `tanhGrad(_:dy:)`

Computes the gradient for the tanh of `x` wrt its input.

``` swift
@inlinable @inline(__always) public static func tanhGrad<T: FloatingPoint & TensorFlowScalar>(_ y: Tensor<T>, dy: Tensor<T>) -> Tensor<T>
```

Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
is the corresponding input gradient.

### `tensorArrayCloseV2(handle:)`

Deprecated. Use TensorArrayCloseV3

``` swift
@inlinable @inline(__always) public static func tensorArrayCloseV2(handle: StringTensor)
```

### `tensorArrayCloseV3(handle:)`

Delete the TensorArray from its resource container.

``` swift
@inlinable @inline(__always) public static func tensorArrayCloseV3(handle: ResourceHandle)
```

This enables the user to close and release the resource in the middle
of a step/run.

#### Parameters

  - handle: - handle: The handle to a TensorArray (output of TensorArray or TensorArrayGrad).

### `tensorArrayConcatV2(handle:flowIn:elementShapeExcept0:)`

Deprecated. Use TensorArrayConcatV3

``` swift
@inlinable @inline(__always) public static func tensorArrayConcatV2<Dtype: TensorFlowScalar>(handle: StringTensor, flowIn: Tensor<Float>, elementShapeExcept0: TensorShape?) -> (value: Tensor<Dtype>, lengths: Tensor<Int64>)
```

### `tensorArrayConcatV3(handle:flowIn:elementShapeExcept0:)`

Concat the elements from the TensorArray into value `value`.

``` swift
@inlinable @inline(__always) public static func tensorArrayConcatV3<Dtype: TensorFlowScalar>(handle: ResourceHandle, flowIn: Tensor<Float>, elementShapeExcept0: TensorShape?) -> (value: Tensor<Dtype>, lengths: Tensor<Int64>)
```

Takes `T` elements of shapes

``` 
(n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
```

and concatenates them into a Tensor of shape:

`(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)`

All elements must have the same shape (excepting the first dimension).

#### Parameters

  - handle: - handle: The handle to a TensorArray.

### `tensorArrayGatherV2(handle:indices:flowIn:elementShape:)`

Deprecated. Use TensorArrayGatherV3

``` swift
@inlinable @inline(__always) public static func tensorArrayGatherV2<Dtype: TensorFlowScalar>(handle: StringTensor, indices: Tensor<Int32>, flowIn: Tensor<Float>, elementShape: TensorShape?) -> Tensor<Dtype>
```

### `tensorArrayGatherV3(handle:indices:flowIn:elementShape:)`

Gather specific elements from the TensorArray into output `value`.

``` swift
@inlinable @inline(__always) public static func tensorArrayGatherV3<Dtype: TensorFlowScalar>(handle: ResourceHandle, indices: Tensor<Int32>, flowIn: Tensor<Float>, elementShape: TensorShape?) -> Tensor<Dtype>
```

All elements selected by `indices` must have the same shape.

#### Parameters

  - handle: - handle: The handle to a TensorArray.
  - indices: - indices: The locations in the TensorArray from which to read tensor elements.

### `tensorArrayGradV2(handle:flowIn:source:)`

Deprecated. Use TensorArrayGradV3

``` swift
@inlinable @inline(__always) public static func tensorArrayGradV2(handle: StringTensor, flowIn: Tensor<Float>, source: String) -> StringTensor
```

### `tensorArrayGradV3(handle:flowIn:source:)`

Creates a TensorArray for storing the gradients of values in the given handle.

``` swift
@inlinable @inline(__always) public static func tensorArrayGradV3(handle: ResourceHandle, flowIn: Tensor<Float>, source: String) -> (gradHandle: ResourceHandle, flowOut: Tensor<Float>)
```

If the given TensorArray gradient already exists, returns a reference to it.

Locks the size of the original TensorArray by disabling its dynamic size flag.

**A note about the input flow\_in:**

The handle flow\_in forces the execution of the gradient lookup to occur
only after certain other operations have occurred.  For example, when
the forward TensorArray is dynamically sized, writes to this TensorArray
may resize the object.  The gradient TensorArray is statically sized based
on the size of the forward TensorArray when this operation executes.
Furthermore, the size of the forward TensorArray is frozen by this call.
As a result, the flow is used to ensure that the call to generate the gradient
TensorArray only happens after all writes are executed.

In the case of dynamically sized TensorArrays, gradient computation should
only be performed on read operations that have themselves been chained via
flow to occur only after all writes have executed. That way the final size
of the forward TensorArray is known when this operation is called.

**A note about the source attribute:**

TensorArray gradient calls use an accumulator TensorArray object.  If
multiple gradients are calculated and run in the same session, the multiple
gradient nodes may accidentally flow through the same accumulator TensorArray.
This double counts and generally breaks the TensorArray gradient flow.

The solution is to identify which gradient call this particular
TensorArray gradient is being called in.  This is performed by identifying
a unique string (e.g. "gradients", "gradients\_1", ...) from the input
gradient Tensor's name.  This string is used as a suffix when creating
the TensorArray gradient object here (the attribute `source`).

The attribute `source` is added as a suffix to the forward TensorArray's
name when performing the creation / lookup, so that each separate gradient
calculation gets its own TensorArray accumulator.

#### Parameters

  - handle: - handle: The handle to the forward TensorArray.

### `tensorArrayGradWithShape(handle:flowIn:shapeToPrepend:source:)`

Creates a TensorArray for storing multiple gradients of values in the given handle.

``` swift
@inlinable @inline(__always) public static func tensorArrayGradWithShape(handle: ResourceHandle, flowIn: Tensor<Float>, shapeToPrepend: Tensor<Int32>, source: String) -> (gradHandle: ResourceHandle, flowOut: Tensor<Float>)
```

Similar to TensorArrayGradV3. However it creates an accumulator with an
expanded shape compared to the input TensorArray whose gradient is being
computed. This enables multiple gradients for the same TensorArray to be
calculated using the same accumulator.

#### Parameters

  - handle: - handle: The handle to the forward TensorArray.

### `tensorArrayReadV2(handle:index:flowIn:)`

Deprecated. Use TensorArrayReadV3

``` swift
@inlinable @inline(__always) public static func tensorArrayReadV2<Dtype: TensorFlowScalar>(handle: StringTensor, index: Tensor<Int32>, flowIn: Tensor<Float>) -> Tensor<Dtype>
```

### `tensorArrayReadV3(handle:index:flowIn:)`

Read an element from the TensorArray into output `value`.

``` swift
@inlinable @inline(__always) public static func tensorArrayReadV3<Dtype: TensorFlowScalar>(handle: ResourceHandle, index: Tensor<Int32>, flowIn: Tensor<Float>) -> Tensor<Dtype>
```

#### Parameters

  - handle: - handle: The handle to a TensorArray.

### `tensorArrayScatterV2(handle:indices:value:flowIn:)`

Deprecated. Use TensorArrayScatterV3

``` swift
@inlinable @inline(__always) public static func tensorArrayScatterV2<T: TensorFlowScalar>(handle: StringTensor, indices: Tensor<Int32>, value: Tensor<T>, flowIn: Tensor<Float>) -> Tensor<Float>
```

### `tensorArrayScatterV3(handle:indices:value:flowIn:)`

Scatter the data from the input value into specific TensorArray elements.

``` swift
@inlinable @inline(__always) public static func tensorArrayScatterV3<T: TensorFlowScalar>(handle: ResourceHandle, indices: Tensor<Int32>, value: Tensor<T>, flowIn: Tensor<Float>) -> Tensor<Float>
```

`indices` must be a vector, its length must match the first dim of `value`.

#### Parameters

  - handle: - handle: The handle to a TensorArray.
  - indices: - indices: The locations at which to write the tensor elements.
  - value: - value: The concatenated tensor to write to the TensorArray.

### `tensorArraySizeV2(handle:flowIn:)`

Deprecated. Use TensorArraySizeV3

``` swift
@inlinable @inline(__always) public static func tensorArraySizeV2(handle: StringTensor, flowIn: Tensor<Float>) -> Tensor<Int32>
```

### `tensorArraySizeV3(handle:flowIn:)`

Get the current size of the TensorArray.

``` swift
@inlinable @inline(__always) public static func tensorArraySizeV3(handle: ResourceHandle, flowIn: Tensor<Float>) -> Tensor<Int32>
```

#### Parameters

  - handle: - handle: The handle to a TensorArray (output of TensorArray or TensorArrayGrad).

### `tensorArraySplitV2(handle:value:lengths:flowIn:)`

Deprecated. Use TensorArraySplitV3

``` swift
@inlinable @inline(__always) public static func tensorArraySplitV2<T: TensorFlowScalar>(handle: StringTensor, value: Tensor<T>, lengths: Tensor<Int64>, flowIn: Tensor<Float>) -> Tensor<Float>
```

### `tensorArraySplitV3(handle:value:lengths:flowIn:)`

Split the data from the input value into TensorArray elements.

``` swift
@inlinable @inline(__always) public static func tensorArraySplitV3<T: TensorFlowScalar>(handle: ResourceHandle, value: Tensor<T>, lengths: Tensor<Int64>, flowIn: Tensor<Float>) -> Tensor<Float>
```

Assuming that `lengths` takes on values

`(n0, n1, ..., n(T-1))`

and that `value` has shape

`(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)`,

this splits values into a TensorArray with T tensors.

TensorArray index t will be the subtensor of values with starting position

`(n0 + n1 + ... + n(t-1), 0, 0, ...)`

and having size

`nt x d0 x d1 x ...`

#### Parameters

  - handle: - handle: The handle to a TensorArray.
  - value: - value: The concatenated tensor to write to the TensorArray.
  - lengths: - lengths: The vector of lengths, how to split the rows of value into the TensorArray.

### `tensorArrayV2(size:dtype:elementShape:dynamicSize:clearAfterRead:tensorArrayName:)`

Deprecated. Use TensorArrayV3

``` swift
@inlinable @inline(__always) public static func tensorArrayV2(size: Tensor<Int32>, dtype: TensorDataType, elementShape: TensorShape?, dynamicSize: Bool = false, clearAfterRead: Bool = true, tensorArrayName: String) -> StringTensor
```

### `tensorArrayV3(size:dtype:elementShape:dynamicSize:clearAfterRead:identicalElementShapes:tensorArrayName:)`

An array of Tensors of given size.

``` swift
@inlinable @inline(__always) public static func tensorArrayV3(size: Tensor<Int32>, dtype: TensorDataType, elementShape: TensorShape?, dynamicSize: Bool = false, clearAfterRead: Bool = true, identicalElementShapes: Bool = false, tensorArrayName: String) -> (handle: ResourceHandle, flow: Tensor<Float>)
```

Write data via Write and read via Read or Pack.

#### Parameters

  - size: - size: The size of the array.

### `tensorArrayWriteV2(handle:index:value:flowIn:)`

Deprecated. Use TensorArrayGradV3

``` swift
@inlinable @inline(__always) public static func tensorArrayWriteV2<T: TensorFlowScalar>(handle: StringTensor, index: Tensor<Int32>, value: Tensor<T>, flowIn: Tensor<Float>) -> Tensor<Float>
```

### `tensorArrayWriteV3(handle:index:value:flowIn:)`

Push an element onto the tensor\_array.

``` swift
@inlinable @inline(__always) public static func tensorArrayWriteV3<T: TensorFlowScalar>(handle: ResourceHandle, index: Tensor<Int32>, value: Tensor<T>, flowIn: Tensor<Float>) -> Tensor<Float>
```

#### Parameters

  - handle: - handle: The handle to a TensorArray.
  - index: - index: The position to write to inside the TensorArray.
  - value: - value: The tensor to write to the TensorArray.

### `tensorDataset(components:outputShapes:)`

Creates a dataset that emits `components` as a tuple of tensors once.

``` swift
@inlinable @inline(__always) public static func tensorDataset<ToutputTypes: TensorArrayProtocol>(components: ToutputTypes, outputShapes: [TensorShape?]) -> VariantHandle
```

### `tensorForestCreateTreeVariable(treeHandle:treeConfig:)`

Creates a tree resource and returns a handle to it.

``` swift
@inlinable @inline(__always) public static func tensorForestCreateTreeVariable(treeHandle: ResourceHandle, treeConfig: StringTensor)
```

### `tensorForestTreeDeserialize(treeHandle:treeConfig:)`

Deserializes a proto into the tree handle

``` swift
@inlinable @inline(__always) public static func tensorForestTreeDeserialize(treeHandle: ResourceHandle, treeConfig: StringTensor)
```

### `tensorForestTreeIsInitializedOp(treeHandle:)`

Checks whether a tree has been initialized.

``` swift
@inlinable @inline(__always) public static func tensorForestTreeIsInitializedOp(treeHandle: ResourceHandle) -> Tensor<Bool>
```

### `tensorForestTreePredict(treeHandle:denseFeatures:logitsDimension:)`

Output the logits for the given input data

``` swift
@inlinable @inline(__always) public static func tensorForestTreePredict(treeHandle: ResourceHandle, denseFeatures: Tensor<Float>, logitsDimension: Int64) -> Tensor<Float>
```

### `tensorForestTreeResourceHandleOp(container:sharedName:)`

Creates a handle to a TensorForestTreeResource

``` swift
@inlinable @inline(__always) public static func tensorForestTreeResourceHandleOp(container: String, sharedName: String) -> ResourceHandle
```

### `tensorForestTreeSerialize(treeHandle:)`

Serializes the tree handle to a proto

``` swift
@inlinable @inline(__always) public static func tensorForestTreeSerialize(treeHandle: ResourceHandle) -> StringTensor
```

### `tensorForestTreeSize(treeHandle:)`

Get the number of nodes in a tree

``` swift
@inlinable @inline(__always) public static func tensorForestTreeSize(treeHandle: ResourceHandle) -> Tensor<Int32>
```

### `tensorListConcat(inputHandle:elementShape:)`

Concats all tensors in the list along the 0th dimension.

``` swift
@inlinable @inline(__always) public static func tensorListConcat<ElementDtype: TensorFlowScalar>(inputHandle: VariantHandle, elementShape: TensorShape?) -> (tensor: Tensor<ElementDtype>, lengths: Tensor<Int64>)
```

Requires that all tensors have the same shape except the first dimension.

input\_handle: The input list.
tensor: The concated result.
lengths: Output tensor containing sizes of the 0th dimension of tensors in the list, used for computing the gradient.

### `tensorListConcatLists(inputA:inputB:elementDtype:)`

``` swift
@inlinable @inline(__always) public static func tensorListConcatLists(inputA: VariantHandle, inputB: VariantHandle, elementDtype: TensorDataType) -> VariantHandle
```

### `tensorListConcatV2(inputHandle:elementShape:leadingDims:)`

Concats all tensors in the list along the 0th dimension.

``` swift
@inlinable @inline(__always) public static func tensorListConcatV2<ElementDtype: TensorFlowScalar, ShapeType: TensorFlowIndex>(inputHandle: VariantHandle, elementShape: Tensor<ShapeType>, leadingDims: Tensor<Int64>) -> (tensor: Tensor<ElementDtype>, lengths: Tensor<Int64>)
```

Requires that all tensors have the same shape except the first dimension.

input\_handle: The input list.
element\_shape: The shape of the uninitialized elements in the list. If the first
dimension is not -1, it is assumed that all list elements have the same
leading dim.
leading\_dims: The list of leading dims of uninitialized list elements. Used if
the leading dim of input\_handle.element\_shape or the element\_shape input arg
is not already set.
tensor: The concated result.
lengths: Output tensor containing sizes of the 0th dimension of tensors in the list, used for computing the gradient.

### `tensorListElementShape(inputHandle:)`

The shape of the elements of the given list, as a tensor.

``` swift
@inlinable @inline(__always) public static func tensorListElementShape<ShapeType: TensorFlowIndex>(inputHandle: VariantHandle) -> Tensor<ShapeType>
```

input\_handle: the list
element\_shape: the shape of elements of the list

### `tensorListFromTensor(_:elementShape:)`

Creates a TensorList which, when stacked, has the value of `tensor`.

``` swift
@inlinable @inline(__always) public static func tensorListFromTensor<ElementDtype: TensorFlowScalar, ShapeType: TensorFlowIndex>(_ tensor: Tensor<ElementDtype>, elementShape: Tensor<ShapeType>) -> VariantHandle
```

Each tensor in the result list corresponds to one row of the input tensor.

tensor: The input tensor.
output\_handle: The list.

### `tensorListGather(inputHandle:indices:elementShape:)`

Creates a Tensor by indexing into the TensorList.

``` swift
@inlinable @inline(__always) public static func tensorListGather<ElementDtype: TensorFlowScalar>(inputHandle: VariantHandle, indices: Tensor<Int32>, elementShape: Tensor<Int32>) -> Tensor<ElementDtype>
```

Each row in the produced Tensor corresponds to the element in the TensorList
specified by the given index (see `tf.gather`).

input\_handle: The input tensor list.
indices: The indices used to index into the list.
values: The tensor.

### `tensorListGetItem(inputHandle:index:elementShape:)`

``` swift
@inlinable @inline(__always) public static func tensorListGetItem<ElementDtype: TensorFlowScalar>(inputHandle: VariantHandle, index: Tensor<Int32>, elementShape: Tensor<Int32>) -> Tensor<ElementDtype>
```

### `tensorListLength(inputHandle:)`

Returns the number of tensors in the input tensor list.

``` swift
@inlinable @inline(__always) public static func tensorListLength(inputHandle: VariantHandle) -> Tensor<Int32>
```

input\_handle: the input list
length: the number of tensors in the list

### `tensorListPopBack(inputHandle:elementShape:)`

Returns the last element of the input list as well as a list with all but that element.

``` swift
@inlinable @inline(__always) public static func tensorListPopBack<ElementDtype: TensorFlowScalar>(inputHandle: VariantHandle, elementShape: Tensor<Int32>) -> (outputHandle: VariantHandle, tensor: Tensor<ElementDtype>)
```

Fails if the list is empty.

input\_handle: the input list
tensor: the withdrawn last element of the list
element\_dtype: the type of elements in the list
element\_shape: the shape of the output tensor

### `tensorListPushBack(inputHandle:_:)`

Returns a list which has the passed-in `Tensor` as last element and the other elements of the given list in `input_handle`.

``` swift
@inlinable @inline(__always) public static func tensorListPushBack<ElementDtype: TensorFlowScalar>(inputHandle: VariantHandle, _ tensor: Tensor<ElementDtype>) -> VariantHandle
```

tensor: The tensor to put on the list.
input\_handle: The old list.
output\_handle: A list with the elements of the old list followed by tensor.
element\_dtype: the type of elements in the list.
element\_shape: a shape compatible with that of elements in the list.

### `tensorListPushBackBatch(inputHandles:_:)`

``` swift
@inlinable @inline(__always) public static func tensorListPushBackBatch<ElementDtype: TensorFlowScalar>(inputHandles: VariantHandle, _ tensor: Tensor<ElementDtype>) -> VariantHandle
```

### `tensorListReserve(elementShape:numElements:elementDtype:)`

List of the given size with empty elements.

``` swift
@inlinable @inline(__always) public static func tensorListReserve<ShapeType: TensorFlowIndex>(elementShape: Tensor<ShapeType>, numElements: Tensor<Int32>, elementDtype: TensorDataType) -> VariantHandle
```

element\_shape: the shape of the future elements of the list
num\_elements: the number of elements to reserve
handle: the output list
element\_dtype: the desired type of elements in the list.

### `tensorListResize(inputHandle:size:)`

Resizes the list.

``` swift
@inlinable @inline(__always) public static func tensorListResize(inputHandle: VariantHandle, size: Tensor<Int32>) -> VariantHandle
```

input\_handle: the input list
size: size of the output list

### `tensorListScatter(_:indices:elementShape:)`

Creates a TensorList by indexing into a Tensor.

``` swift
@inlinable @inline(__always) public static func tensorListScatter<ElementDtype: TensorFlowScalar, ShapeType: TensorFlowIndex>(_ tensor: Tensor<ElementDtype>, indices: Tensor<Int32>, elementShape: Tensor<ShapeType>) -> VariantHandle
```

Each member of the TensorList corresponds to one row of the input tensor,
specified by the given index (see `tf.gather`).

tensor: The input tensor.
indices: The indices used to index into the list.
element\_shape: The shape of the elements in the list (can be less specified than
the shape of the tensor).
output\_handle: The TensorList.

### `tensorListScatterIntoExistingList(inputHandle:_:indices:)`

Scatters tensor at indices in an input list.

``` swift
@inlinable @inline(__always) public static func tensorListScatterIntoExistingList<ElementDtype: TensorFlowScalar>(inputHandle: VariantHandle, _ tensor: Tensor<ElementDtype>, indices: Tensor<Int32>) -> VariantHandle
```

Each member of the TensorList corresponds to one row of the input tensor,
specified by the given index (see `tf.gather`).

input\_handle: The list to scatter into.
tensor: The input tensor.
indices: The indices used to index into the list.
output\_handle: The TensorList.

### `tensorListScatterV2(_:indices:elementShape:numElements:)`

Creates a TensorList by indexing into a Tensor.

``` swift
@inlinable @inline(__always) public static func tensorListScatterV2<ElementDtype: TensorFlowScalar, ShapeType: TensorFlowIndex>(_ tensor: Tensor<ElementDtype>, indices: Tensor<Int32>, elementShape: Tensor<ShapeType>, numElements: Tensor<Int32>) -> VariantHandle
```

Each member of the TensorList corresponds to one row of the input tensor,
specified by the given index (see `tf.gather`).

tensor: The input tensor.
indices: The indices used to index into the list.
element\_shape: The shape of the elements in the list (can be less specified than
the shape of the tensor).
num\_elements: The size of the output list. Must be large enough to accommodate
the largest index in indices. If -1, the list is just large enough to include
the largest index in indices.
output\_handle: The TensorList.

### `tensorListSetItem(inputHandle:index:item:)`

``` swift
@inlinable @inline(__always) public static func tensorListSetItem<ElementDtype: TensorFlowScalar>(inputHandle: VariantHandle, index: Tensor<Int32>, item: Tensor<ElementDtype>) -> VariantHandle
```

### `tensorListSplit(_:elementShape:lengths:)`

Splits a tensor into a list.

``` swift
@inlinable @inline(__always) public static func tensorListSplit<ElementDtype: TensorFlowScalar, ShapeType: TensorFlowIndex>(_ tensor: Tensor<ElementDtype>, elementShape: Tensor<ShapeType>, lengths: Tensor<Int64>) -> VariantHandle
```

list\[i\] corresponds to lengths\[i\] tensors from the input tensor.
The tensor must have rank at least 1 and contain exactly sum(lengths) elements.

tensor: The input tensor.
element\_shape: A shape compatible with that of elements in the tensor.
lengths: Vector of sizes of the 0th dimension of tensors in the list.
output\_handle: The list.

### `tensorListStack(inputHandle:elementShape:numElements:)`

Stacks all tensors in the list.

``` swift
@inlinable @inline(__always) public static func tensorListStack<ElementDtype: TensorFlowScalar>(inputHandle: VariantHandle, elementShape: Tensor<Int32>, numElements: Int64 = -1) -> Tensor<ElementDtype>
```

Requires that all tensors have the same shape.

input\_handle: the input list
tensor: the gathered result
num\_elements: optional. If not -1, the number of elements in the list.

### `tensorScatterAdd(_:indices:updates:)`

Adds sparse `updates` to an existing tensor according to `indices`.

``` swift
@inlinable @inline(__always) public static func tensorScatterAdd<T: TensorFlowScalar, Tindices: TensorFlowIndex>(_ tensor: Tensor<T>, indices: Tensor<Tindices>, updates: Tensor<T>) -> Tensor<T>
```

This operation creates a new tensor by adding sparse `updates` to the passed
in `tensor`.
This operation is very similar to `tf.scatter_nd_add`, except that the updates
are added onto an existing tensor (as opposed to a variable). If the memory
for the existing tensor cannot be re-used, a copy is made and updated.

`indices` is an integer tensor containing indices into a new tensor of shape
`shape`.  The last dimension of `indices` can be at most the rank of `shape`:

``` 
indices.shape[-1] <= shape.rank
```

The last dimension of `indices` corresponds to indices into elements
(if `indices.shape[-1] = shape.rank`) or slices
(if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
`shape`.  `updates` is a tensor with shape

``` 
indices.shape[:-1] + shape[indices.shape[-1]:]
```

The simplest form of tensor\_scatter\_add is to add individual elements to a
tensor by index. For example, say we want to add 4 elements in a rank-1
tensor with 8 elements.

In Python, this scatter add operation would look like this:

``` python
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    tensor = tf.ones([8], dtype=tf.int32)
    updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
    print(updated)
```

The resulting tensor would look like this:

``` 
[1, 12, 1, 11, 10, 1, 1, 13]
```

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.

In Python, this scatter add operation would look like this:

``` python
    indices = tf.constant([[0], [2]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    tensor = tf.ones([4, 4, 4],dtype=tf.int32)
    updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
    print(updated)
```

The resulting tensor would look like this:

``` 
[[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
 [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
 [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
 [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
```

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, the index is ignored.

#### Parameters

  - tensor: - tensor: Tensor to copy/update.
  - indices: - indices: Index tensor.
  - updates: - updates: Updates to scatter into output.

### `tensorScatterSub(_:indices:updates:)`

Subtracts sparse `updates` from an existing tensor according to `indices`.

``` swift
@inlinable @inline(__always) public static func tensorScatterSub<T: TensorFlowScalar, Tindices: TensorFlowIndex>(_ tensor: Tensor<T>, indices: Tensor<Tindices>, updates: Tensor<T>) -> Tensor<T>
```

This operation creates a new tensor by subtracting sparse `updates` from the
passed in `tensor`.
This operation is very similar to `tf.scatter_nd_sub`, except that the updates
are subtracted from an existing tensor (as opposed to a variable). If the memory
for the existing tensor cannot be re-used, a copy is made and updated.

`indices` is an integer tensor containing indices into a new tensor of shape
`shape`.  The last dimension of `indices` can be at most the rank of `shape`:

``` 
indices.shape[-1] <= shape.rank
```

The last dimension of `indices` corresponds to indices into elements
(if `indices.shape[-1] = shape.rank`) or slices
(if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
`shape`.  `updates` is a tensor with shape

``` 
indices.shape[:-1] + shape[indices.shape[-1]:]
```

The simplest form of tensor\_scatter\_sub is to subtract individual elements
from a tensor by index. For example, say we want to insert 4 scattered elements
in a rank-1 tensor with 8 elements.

In Python, this scatter subtract operation would look like this:

``` python
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    tensor = tf.ones([8], dtype=tf.int32)
    updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
    print(updated)
```

The resulting tensor would look like this:

``` 
[1, -10, 1, -9, -8, 1, 1, -11]
```

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.

In Python, this scatter add operation would look like this:

``` python
    indices = tf.constant([[0], [2]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    tensor = tf.ones([4, 4, 4],dtype=tf.int32)
    updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
    print(updated)
```

The resulting tensor would look like this:

``` 
[[[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
 [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
 [[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
 [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
```

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, the index is ignored.

#### Parameters

  - tensor: - tensor: Tensor to copy/update.
  - indices: - indices: Index tensor.
  - updates: - updates: Updates to scatter into output.

### `tensorScatterUpdate(_:indices:updates:)`

Scatter `updates` into an existing tensor according to `indices`.

``` swift
@inlinable @inline(__always) public static func tensorScatterUpdate<T: TensorFlowScalar, Tindices: TensorFlowIndex>(_ tensor: Tensor<T>, indices: Tensor<Tindices>, updates: Tensor<T>) -> Tensor<T>
```

This operation creates a new tensor by applying sparse `updates` to the passed
in `tensor`.
This operation is very similar to `tf.scatter_nd`, except that the updates are
scattered onto an existing tensor (as opposed to a zero-tensor). If the memory
for the existing tensor cannot be re-used, a copy is made and updated.

If `indices` contains duplicates, then their updates are accumulated (summed).

**WARNING**: The order in which updates are applied is nondeterministic, so the
output will be nondeterministic if `indices` contains duplicates -- because
of some numerical approximation issues, numbers summed in different order
may yield different results.

`indices` is an integer tensor containing indices into a new tensor of shape
`shape`.  The last dimension of `indices` can be at most the rank of `shape`:

``` 
indices.shape[-1] <= shape.rank
```

The last dimension of `indices` corresponds to indices into elements
(if `indices.shape[-1] = shape.rank`) or slices
(if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
`shape`.  `updates` is a tensor with shape

``` 
indices.shape[:-1] + shape[indices.shape[-1]:]
```

The simplest form of scatter is to insert individual elements in a tensor by
index. For example, say we want to insert 4 scattered elements in a rank-1
tensor with 8 elements.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
</div>

In Python, this scatter operation would look like this:

``` 
>>> indices = tf.constant([[4], [3], [1], [7]])
>>> updates = tf.constant([9, 10, 11, 12])
>>> tensor = tf.ones([8], dtype=tf.int32)
>>> print(tf.tensor_scatter_nd_update(tensor, indices, updates))
tf.Tensor([ 1 11  1 10  9  1  1 12], shape=(8,), dtype=int32)
```

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.

In Python, this scatter operation would look like this:

``` 
>>> indices = tf.constant([[0], [2]])
>>> updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
...                         [7, 7, 7, 7], [8, 8, 8, 8]],
...                        [[5, 5, 5, 5], [6, 6, 6, 6],
...                         [7, 7, 7, 7], [8, 8, 8, 8]]])
>>> tensor = tf.ones([4, 4, 4], dtype=tf.int32)
>>> print(tf.tensor_scatter_nd_update(tensor, indices, updates).numpy())
[[[5 5 5 5]
  [6 6 6 6]
  [7 7 7 7]
  [8 8 8 8]]
 [[1 1 1 1]
  [1 1 1 1]
  [1 1 1 1]
  [1 1 1 1]]
 [[5 5 5 5]
  [6 6 6 6]
  [7 7 7 7]
  [8 8 8 8]]
 [[1 1 1 1]
  [1 1 1 1]
  [1 1 1 1]
  [1 1 1 1]]]
```

Note that on CPU, if an out of bound index is found, an error is returned.
On GPU, if an out of bound index is found, the index is ignored.

#### Parameters

  - tensor: - tensor: Tensor to copy/update.
  - indices: - indices: Index tensor.
  - updates: - updates: Updates to scatter into output.

### `tensorSliceDataset(components:outputShapes:)`

Creates a dataset that emits each dim-0 slice of `components` once.

``` swift
@inlinable @inline(__always) public static func tensorSliceDataset<ToutputTypes: TensorArrayProtocol>(components: ToutputTypes, outputShapes: [TensorShape?]) -> VariantHandle
```

### `tensorStridedSliceUpdate(_:begin:end:strides:value:beginMask:endMask:ellipsisMask:newAxisMask:shrinkAxisMask:)`

Assign `value` to the sliced l-value reference of `input`.

``` swift
@inlinable @inline(__always) public static func tensorStridedSliceUpdate<T: TensorFlowScalar, Index: TensorFlowIndex>(_ input: Tensor<T>, begin: Tensor<Index>, end: Tensor<Index>, strides: Tensor<Index>, value: Tensor<T>, beginMask: Int64 = 0, endMask: Int64 = 0, ellipsisMask: Int64 = 0, newAxisMask: Int64 = 0, shrinkAxisMask: Int64 = 0) -> Tensor<T>
```

The values of `value` are assigned to the positions in the tensor `input` that
are selected by the slice parameters. The slice parameters `begin` `end`
`strides` etc. work exactly as in `StridedSlice`.

NOTE this op currently does not support broadcasting and so `value`'s shape
must be exactly the shape produced by the slice of `input`.

### `tensorSummary(_:description:labels:displayName:)`

Outputs a `Summary` protocol buffer with a tensor.

``` swift
@inlinable @inline(__always) public static func tensorSummary<T: TensorFlowScalar>(_ tensor: Tensor<T>, description: String, labels: [String], displayName: String) -> StringTensor
```

This op is being phased out in favor of TensorSummaryV2, which lets callers pass
a tag as well as a serialized SummaryMetadata proto string that contains
plugin-specific data. We will keep this op to maintain backwards compatibility.

#### Parameters

  - tensor: - tensor: A tensor to serialize.

### `tensorSummaryV2(tag:_:serializedSummaryMetadata:)`

Outputs a `Summary` protocol buffer with a tensor and per-plugin data.

``` swift
@inlinable @inline(__always) public static func tensorSummaryV2<T: TensorFlowScalar>(tag: StringTensor, _ tensor: Tensor<T>, serializedSummaryMetadata: StringTensor) -> StringTensor
```

#### Parameters

  - tag: - tag: A string attached to this summary. Used for organization in TensorBoard.
  - tensor: - tensor: A tensor to serialize.

### `testAttr()`

``` swift
@inlinable @inline(__always) public static func testAttr<T: FloatingPoint & TensorFlowScalar>() -> Tensor<T>
```

### `testStringOutput(_:)`

``` swift
@inlinable @inline(__always) public static func testStringOutput(_ input: Tensor<Float>) -> (output1: Tensor<Float>, output2: StringTensor)
```

### `textLineDataset(filenames:compressionType:bufferSize:)`

Creates a dataset that emits the lines of one or more text files.

``` swift
@inlinable @inline(__always) public static func textLineDataset(filenames: StringTensor, compressionType: StringTensor, bufferSize: Tensor<Int64>) -> VariantHandle
```

#### Parameters

  - filenames: - filenames: A scalar or a vector containing the name(s) of the file(s) to be read.

### `textLineReaderV2(skipHeaderLines:container:sharedName:)`

A Reader that outputs the lines of a file delimited by '\\n'.

``` swift
@inlinable @inline(__always) public static func textLineReaderV2(skipHeaderLines: Int64 = 0, container: String, sharedName: String) -> ResourceHandle
```

### `threadPoolDataset(inputDataset:threadPool:outputTypes:outputShapes:)`

Creates a dataset that uses a custom thread pool to compute `input_dataset`.

``` swift
@inlinable @inline(__always) public static func threadPoolDataset(inputDataset: VariantHandle, threadPool: ResourceHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `threadPoolHandle(numThreads:maxIntraOpParallelism:displayName:container:sharedName:)`

Creates a dataset that uses a custom thread pool to compute `input_dataset`.

``` swift
@inlinable @inline(__always) public static func threadPoolHandle(numThreads: Int64, maxIntraOpParallelism: Int64 = 1, displayName: String, container: String, sharedName: String) -> ResourceHandle
```

### `threadUnsafeUnigramCandidateSampler(trueClasses:numTrue:numSampled:unique:rangeMax:seed:seed2:)`

Generates labels for candidate sampling with a learned unigram distribution.

``` swift
@inlinable @inline(__always) public static func threadUnsafeUnigramCandidateSampler(trueClasses: Tensor<Int64>, numTrue: Int64, numSampled: Int64, unique: Bool, rangeMax: Int64, seed: Int64 = 0, seed2: Int64 = 0) -> (
    sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>,
    sampledExpectedCount: Tensor<Float>
  )
```

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

### `tile(_:multiples:)`

Constructs a tensor by tiling a given tensor.

``` swift
@inlinable @inline(__always) public static func tile<T: TensorFlowScalar, Tmultiples: TensorFlowIndex>(_ input: Tensor<T>, multiples: Tensor<Tmultiples>) -> Tensor<T>
```

This operation creates a new tensor by replicating `input` `multiples` times.
The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
and the values of `input` are replicated `multiples[i]` times along the 'i'th
dimension. For example, tiling `[a b c d]` by `[2]` produces
`[a b c d a b c d]`.

> > > a = tf.constant(\[\[1,2,3\],\[4,5,6\]\], tf.int32)
> > > b = tf.constant(\[1,2\], tf.int32)
> > > tf.tile(a, b)
> > > \<tf.Tensor: shape=(2, 6), dtype=int32, numpy=
> > > array(\[\[1, 2, 3, 1, 2, 3\],
> > > \[4, 5, 6, 4, 5, 6\]\], dtype=int32)\>
> > > c = tf.constant(\[2,1\], tf.int32)
> > > tf.tile(a, c)
> > > \<tf.Tensor: shape=(4, 3), dtype=int32, numpy=
> > > array(\[\[1, 2, 3\],
> > > \[4, 5, 6\],
> > > \[1, 2, 3\],
> > > \[4, 5, 6\]\], dtype=int32)\>
> > > d = tf.constant(\[2,2\], tf.int32)
> > > tf.tile(a, d)
> > > \<tf.Tensor: shape=(4, 6), dtype=int32, numpy=
> > > array(\[\[1, 2, 3, 1, 2, 3\],
> > > \[4, 5, 6, 4, 5, 6\],
> > > \[1, 2, 3, 1, 2, 3\],
> > > \[4, 5, 6, 4, 5, 6\]\], dtype=int32)\>

#### Parameters

  - input: - input: 1-D or higher.
  - multiples: - multiples: 1-D. Length must be the same as the number of dimensions in `input`

### `tileGrad(_:multiples:)`

Returns the gradient of `Tile`.

``` swift
@inlinable @inline(__always) public static func tileGrad<T: TensorFlowScalar>(_ input: Tensor<T>, multiples: Tensor<Int32>) -> Tensor<T>
```

Since `Tile` takes an input and repeats the input `multiples` times
along each dimension, `TileGrad` takes in `multiples` and aggregates
each repeated tile of `input` into `output`.

### `timestamp()`

Provides the time since epoch in seconds.

``` swift
@inlinable @inline(__always) public static func timestamp() -> Tensor<Double>
```

Returns the timestamp as a `float64` for seconds since the Unix epoch.

Note: the timestamp is computed when the op is executed, not when it is added
to the graph.

### `topK(_:k:sorted:)`

Finds values and indices of the `k` largest elements for the last dimension.

``` swift
@inlinable @inline(__always) public static func topK<T: TensorFlowNumeric>(_ input: Tensor<T>, k: Int64, sorted: Bool = true) -> (values: Tensor<T>, indices: Tensor<Int32>)
```

If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

``` 
values.shape = indices.shape = input.shape[:-1] + [k]
```

If two elements are equal, the lower-index element appears first.

If `k` varies dynamically, use `TopKV2` below.

#### Parameters

  - input: - input: 1-D or higher with last dimension at least `k`.

### `topKV2(_:k:sorted:)`

Finds values and indices of the `k` largest elements for the last dimension.

``` swift
@inlinable @inline(__always) public static func topKV2<T: TensorFlowNumeric>(_ input: Tensor<T>, k: Tensor<Int32>, sorted: Bool = true) -> (values: Tensor<T>, indices: Tensor<Int32>)
```

If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

``` 
values.shape = indices.shape = input.shape[:-1] + [k]
```

If two elements are equal, the lower-index element appears first.

#### Parameters

  - input: - input: 1-D or higher with last dimension at least `k`.
  - k: - k: 0-D.  Number of top elements to look for along the last dimension (along each row for matrices).

### `transpose(_:perm:)`

Shuffle dimensions of x according to a permutation.

``` swift
@inlinable @inline(__always) public static func transpose<T: TensorFlowScalar, Tperm: TensorFlowIndex>(_ x: Tensor<T>, perm: Tensor<Tperm>) -> Tensor<T>
```

The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
`y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`

### `tridiagonalMatMul(superdiag:maindiag:subdiag:rhs:)`

Calculate product with tridiagonal matrix.

``` swift
@inlinable @inline(__always) public static func tridiagonalMatMul<T: FloatingPoint & TensorFlowScalar>(superdiag: Tensor<T>, maindiag: Tensor<T>, subdiag: Tensor<T>, rhs: Tensor<T>) -> Tensor<T>
```

Calculates product of two matrices, where left matrix is a tridiagonal matrix.

#### Parameters

  - superdiag: - superdiag: Tensor of shape `[..., 1, M]`, representing superdiagonals of tri-diagonal matrices to the left of multiplication. Last element is ingored.
  - maindiag: - maindiag: Tensor of shape `[..., 1, M]`, representing main diagonals of tri-diagonal matrices to the left of multiplication.
  - subdiag: - subdiag: Tensor of shape `[..., 1, M]`, representing subdiagonals of tri-diagonal matrices to the left of multiplication. First element is ingored.
  - rhs: - rhs: Tensor of shape `[..., M, N]`, representing MxN matrices to the right of multiplication.

### `tridiagonalSolve(diagonals:rhs:partialPivoting:)`

Solves tridiagonal systems of equations.

``` swift
@inlinable @inline(__always) public static func tridiagonalSolve<T: FloatingPoint & TensorFlowScalar>(diagonals: Tensor<T>, rhs: Tensor<T>, partialPivoting: Bool = true) -> Tensor<T>
```

Solves tridiagonal systems of equations.
Supports batch dimensions and multiple right-hand sides per each left-hand
side.
On CPU, solution is computed via Gaussian elimination with or without partial
pivoting, depending on `partial_pivoting` attribute. On GPU, Nvidia's cuSPARSE
library is used: https://docs.nvidia.com/cuda/cusparse/index.html\#gtsv

#### Parameters

  - diagonals: - diagonals: Tensor of shape `[..., 3, M]` whose innermost 2 dimensions represent the tridiagonal matrices with three rows being the superdiagonal, diagonals, and subdiagonals, in order. The last element of the superdiagonal and the first element of the subdiagonal is ignored.
  - rhs: - rhs: Tensor of shape `[..., M, K]`, representing K right-hand sides per each left-hand side.

### `truncateDiv(_:_:)`

Returns x / y element-wise for integer types.

``` swift
@inlinable @inline(__always) public static func truncateDiv<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

Truncation designates that negative numbers will round fractional quantities
toward zero. I.e. -7 / 5 = -1. This matches C semantics but it is different
than Python semantics. See `FloorDiv` for a division function that matches
Python Semantics.

*NOTE*: `TruncateDiv` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `truncateMod(_:_:)`

Returns element-wise remainder of division. This emulates C semantics in that

``` swift
@inlinable @inline(__always) public static func truncateMod<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

the result here is consistent with a truncating divide. E.g. `truncate(x / y) * y + truncate_mod(x, y) = x`.

*NOTE*: `TruncateMod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

### `truncatedNormal(shape:seed:seed2:)`

Outputs random values from a truncated normal distribution.

``` swift
@inlinable @inline(__always) public static func truncatedNormal<Dtype: FloatingPoint & TensorFlowScalar, T: TensorFlowIndex>(shape: Tensor<T>, seed: Int64 = 0, seed2: Int64 = 0) -> Tensor<Dtype>
```

The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.

#### Parameters

  - shape: - shape: The shape of the output tensor.

### `tryRpc(address:method:request:protocol_:failFast:timeoutInMs:)`

Perform batches of RPC requests.

``` swift
@inlinable @inline(__always) public static func tryRpc(address: StringTensor, method: StringTensor, request: StringTensor, protocol_: String, failFast: Bool = true, timeoutInMs: Int64 = 0) -> (response: StringTensor, statusCode: Tensor<Int32>, statusMessage: StringTensor)
```

This op asynchronously performs either a single RPC request, or a batch
of requests.  RPC requests are defined by three main parameters:

For example, if you have an RPC service running on port localhost:2345,
and its interface is configured with the following proto declaration:

``` 
service MyService {
  rpc MyMethod(MyRequestProto) returns (MyResponseProto) {
  }
};
```

then call this op with arguments:

``` 
address = "localhost:2345"
method = "MyService/MyMethod"
```

The `request` tensor is a string tensor representing serialized `MyRequestProto`
strings; and the output string tensor `response` will have the same shape
and contain (upon successful completion) corresponding serialized
`MyResponseProto` strings.

For example, to send a single, empty, `MyRequestProto`, call
this op with `request = ""`.  To send 5 **parallel** empty requests,
call this op with `request = ["", "", "", "", ""]`.

More generally, one can create a batch of `MyRequestProto` serialized protos
from regular batched tensors using the `encode_proto` op, and convert
the response `MyResponseProto` serialized protos to batched tensors
using the `decode_proto` op.

**NOTE** Working with serialized proto strings is faster than instantiating
actual proto objects in memory, so no performance degradation is expected
compared to writing custom kernels for this workflow.

Unlike the standard `Rpc` op, if the connection fails or the remote worker
returns an error status, this op does **not** reraise the exception.
Instead, the `status_code` and `status_message` entry for the corresponding RPC
call is set with the error returned from the RPC call.  The `response` tensor
will contain valid response values for those minibatch entries whose RPCs did
not fail; the rest of the entries will have empty strings.

#### Parameters

  - address: - address: `0-D` or `1-D`.  The address (i.e. host\_name:port) of the RPC server. If this tensor has more than 1 element, then multiple parallel rpc requests are sent.  This argument broadcasts with `method` and `request`.
  - method: - method: `0-D` or `1-D`.  The method address on the RPC server. If this tensor has more than 1 element, then multiple parallel rpc requests are sent.  This argument broadcasts with `address` and `request`.
  - request: - request: `0-D` or `1-D`.  Serialized proto strings: the rpc request argument. If this tensor has more than 1 element, then multiple parallel rpc requests are sent.  This argument broadcasts with `address` and `method`.

### `twoFloatInputs(_:_:)`

``` swift
@inlinable @inline(__always) public static func twoFloatInputs(_ a: Tensor<Float>, _ b: Tensor<Float>)
```

### `twoFloatInputsFloatOutput(_:_:)`

``` swift
@inlinable @inline(__always) public static func twoFloatInputsFloatOutput(_ a: Tensor<Float>, _ b: Tensor<Float>) -> Tensor<Float>
```

### `twoFloatInputsIntOutput(_:_:)`

``` swift
@inlinable @inline(__always) public static func twoFloatInputsIntOutput(_ a: Tensor<Float>, _ b: Tensor<Float>) -> Tensor<Int32>
```

### `twoFloatOutputs()`

``` swift
@inlinable @inline(__always) public static func twoFloatOutputs() -> (a: Tensor<Float>, b: Tensor<Float>)
```

### `twoIntInputs(_:_:)`

``` swift
@inlinable @inline(__always) public static func twoIntInputs(_ a: Tensor<Int32>, _ b: Tensor<Int32>)
```

### `twoIntOutputs()`

``` swift
@inlinable @inline(__always) public static func twoIntOutputs() -> (a: Tensor<Int32>, b: Tensor<Int32>)
```

### `typeList(_:)`

``` swift
@inlinable @inline(__always) public static func typeList<T: TensorArrayProtocol>(_ a: T)
```

### `typeListRestrict(_:)`

``` swift
@inlinable @inline(__always) public static func typeListRestrict<T: TensorArrayProtocol>(_ a: T)
```

### `typeListTwice(_:_:)`

``` swift
@inlinable @inline(__always) public static func typeListTwice<T: TensorArrayProtocol>(_ a: T, _ b: T)
```

### `unary(_:)`

``` swift
@inlinable @inline(__always) public static func unary<T: TensorFlowScalar>(_ a: Tensor<T>) -> Tensor<T>
```

### `unbatch(batchedTensor:batchIndex:id:timeoutMicros:container:sharedName:)`

Reverses the operation of Batch for a single output Tensor.

``` swift
@inlinable @inline(__always) public static func unbatch<T: TensorFlowScalar>(batchedTensor: Tensor<T>, batchIndex: Tensor<Int64>, id: Tensor<Int64>, timeoutMicros: Int64, container: String, sharedName: String) -> Tensor<T>
```

An instance of Unbatch either receives an empty batched\_tensor, in which case it
asynchronously waits until the values become available from a concurrently
running instance of Unbatch with the same container and shared\_name, or receives
a non-empty batched\_tensor in which case it finalizes all other concurrently
running instances and outputs its own element from the batch.

batched\_tensor: The possibly transformed output of Batch. The size of the first
dimension should remain unchanged by the transformations for the operation to
work.
batch\_index: The matching batch\_index obtained from Batch.
id: The id scalar emitted by Batch.
unbatched\_tensor: The Tensor corresponding to this execution.
timeout\_micros: Maximum amount of time (in microseconds) to wait to receive the
batched input tensor associated with a given invocation of the op.
container: Container to control resource sharing.
shared\_name: Instances of Unbatch with the same container and shared\_name are
assumed to possibly belong to the same batch. If left empty, the op name will
be used as the shared name.

### `unbatchDataset(inputDataset:outputTypes:outputShapes:)`

A dataset that splits the elements of its input into multiple elements.

``` swift
@inlinable @inline(__always) public static func unbatchDataset(inputDataset: VariantHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `unbatchGrad(originalInput:batchIndex:grad:id:container:sharedName:)`

Gradient of Unbatch.

``` swift
@inlinable @inline(__always) public static func unbatchGrad<T: TensorFlowScalar>(originalInput: Tensor<T>, batchIndex: Tensor<Int64>, grad: Tensor<T>, id: Tensor<Int64>, container: String, sharedName: String) -> Tensor<T>
```

Acts like Batch but using the given batch\_index index of batching things as they
become available. This ensures that the gradients are propagated back in the
same session which did the forward pass.

original\_input: The input to the Unbatch operation this is the gradient of.
batch\_index: The batch\_index given to the Unbatch operation this is the gradient
of.
grad: The downstream gradient.
id: The id scalar emitted by Batch.
batched\_grad: The return value, either an empty tensor or the batched gradient.
container: Container to control resource sharing.
shared\_name: Instances of UnbatchGrad with the same container and shared\_name
are assumed to possibly belong to the same batch. If left empty, the op name
will be used as the shared name.

### `unicodeDecode(_:inputEncoding:errors:replacementChar:replaceControlCharacters:)`

Decodes each string in `input` into a sequence of Unicode code points.

``` swift
@inlinable @inline(__always) public static func unicodeDecode<Tsplits: TensorFlowIndex>(_ input: StringTensor, inputEncoding: String, errors: Errors = .replace, replacementChar: Int64 = 65533, replaceControlCharacters: Bool = false) -> (rowSplits: Tensor<Tsplits>, charValues: Tensor<Int32>)
```

The character codepoints for all strings are returned using a single vector
`char_values`, with strings expanded to characters in row-major order.

The `row_splits` tensor indicates where the codepoints for
each input string begin and end within the `char_values` tensor.
In particular, the values for the `i`th
string (in row-major order) are stored in the slice
`[row_splits[i]:row_splits[i+1]]`. Thus:

#### Parameters

  - input: - input: The text to be decoded. Can have any shape. Note that the output is flattened to a vector of char values.

### `unicodeDecodeWithOffsets(_:inputEncoding:errors:replacementChar:replaceControlCharacters:)`

Decodes each string in `input` into a sequence of Unicode code points.

``` swift
@inlinable @inline(__always) public static func unicodeDecodeWithOffsets<Tsplits: TensorFlowIndex>(_ input: StringTensor, inputEncoding: String, errors: Errors = .replace, replacementChar: Int64 = 65533, replaceControlCharacters: Bool = false) -> (rowSplits: Tensor<Tsplits>, charValues: Tensor<Int32>, charToByteStarts: Tensor<Int64>)
```

The character codepoints for all strings are returned using a single vector
`char_values`, with strings expanded to characters in row-major order.
Similarly, the character start byte offsets are returned using a single vector
`char_to_byte_starts`, with strings expanded in row-major order.

The `row_splits` tensor indicates where the codepoints and start offsets for
each input string begin and end within the `char_values` and
`char_to_byte_starts` tensors.  In particular, the values for the `i`th
string (in row-major order) are stored in the slice
`[row_splits[i]:row_splits[i+1]]`. Thus:

#### Parameters

  - input: - input: The text to be decoded. Can have any shape. Note that the output is flattened to a vector of char values.

### `unicodeEncode(inputValues:inputSplits:errors:outputEncoding:replacementChar:)`

Encode a tensor of ints into unicode strings.

``` swift
@inlinable @inline(__always) public static func unicodeEncode<Tsplits: TensorFlowIndex>(inputValues: Tensor<Int32>, inputSplits: Tensor<Tsplits>, errors: Errors = .replace, outputEncoding: OutputEncoding, replacementChar: Int64 = 65533) -> StringTensor
```

Returns a vector of strings, where `output[i]` is constructed by encoding the
Unicode codepoints in `input_values[input_splits[i]:input_splits[i+1]]`
using `output_encoding`.

Example:

``` 
input_values = [72, 101, 108, 108, 111, 87, 111, 114, 108, 100]
input_splits = [0, 5, 10]
output_encoding = 'UTF-8'
 
output = ['Hello', 'World']
```

### `unicodeScript(_:)`

Determine the script codes of a given tensor of Unicode integer code points.

``` swift
@inlinable @inline(__always) public static func unicodeScript(_ input: Tensor<Int32>) -> Tensor<Int32>
```

This operation converts Unicode code points to script codes corresponding to
each code point. Script codes correspond to International Components for
Unicode (ICU) UScriptCode values. See http://icu-project.org/apiref/icu4c/uscript\_8h.html.
Returns -1 (USCRIPT\_INVALID\_CODE) for invalid codepoints. Output shape will
match input shape.

#### Parameters

  - input: - input: A Tensor of int32 Unicode code points.

### `unicodeTranscode(_:inputEncoding:outputEncoding:errors:replacementChar:replaceControlCharacters:)`

Transcode the input text from a source encoding to a destination encoding.

``` swift
@inlinable @inline(__always) public static func unicodeTranscode(_ input: StringTensor, inputEncoding: String, outputEncoding: OutputEncoding, errors: Errors = .replace, replacementChar: Int64 = 65533, replaceControlCharacters: Bool = false) -> StringTensor
```

The input is a string tensor of any shape. The output is a string tensor of
the same shape containing the transcoded strings. Output strings are always
valid unicode. If the input contains invalid encoding positions, the
`errors` attribute sets the policy for how to deal with them. If the default
error-handling policy is used, invalid formatting will be substituted in the
output by the `replacement_char`. If the errors policy is to `ignore`, any
invalid encoding positions in the input are skipped and not included in the
output. If it set to `strict` then any invalid formatting will result in an
InvalidArgument error.

This operation can be used with `output_encoding = input_encoding` to enforce
correct formatting for inputs even if they are already in the desired encoding.

If the input is prefixed by a Byte Order Mark needed to determine encoding
(e.g. if the encoding is UTF-16 and the BOM indicates big-endian), then that
BOM will be consumed and not emitted into the output. If the input encoding
is marked with an explicit endianness (e.g. UTF-16-BE), then the BOM is
interpreted as a non-breaking-space and is preserved in the output (including
always for UTF-8).

The end result is that if the input is marked as an explicit endianness the
transcoding is faithful to all codepoints in the source. If it is not marked
with an explicit endianness, the BOM is not considered part of the string itself
but as metadata, and so is not preserved in the output.

#### Parameters

  - input: - input: The text to be processed. Can have any shape.

### `uniformCandidateSampler(trueClasses:numTrue:numSampled:unique:rangeMax:seed:seed2:)`

Generates labels for candidate sampling with a uniform distribution.

``` swift
@inlinable @inline(__always) public static func uniformCandidateSampler(trueClasses: Tensor<Int64>, numTrue: Int64, numSampled: Int64, unique: Bool, rangeMax: Int64, seed: Int64 = 0, seed2: Int64 = 0) -> (
    sampledCandidates: Tensor<Int64>, trueExpectedCount: Tensor<Float>,
    sampledExpectedCount: Tensor<Float>
  )
```

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

### `unique(_:)`

Finds unique elements in a 1-D tensor.

``` swift
@inlinable @inline(__always) public static func unique<T: TensorFlowScalar, OutIdx: TensorFlowIndex>(_ x: Tensor<T>) -> (y: Tensor<T>, idx: Tensor<OutIdx>)
```

This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`; `x` does not need to be sorted.
This operation also returns a tensor `idx` the same size as `x` that contains
the index of each value of `x` in the unique output `y`. In other words:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

Examples:

``` 
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx = unique(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
```

``` 
# tensor 'x' is [4, 5, 1, 2, 3, 3, 4, 5]
y, idx = unique(x)
y ==> [4, 5, 1, 2, 3]
idx ==> [0, 1, 2, 3, 4, 4, 0, 1]
```

#### Parameters

  - x: - x: 1-D.

### `uniqueDataset(inputDataset:outputTypes:outputShapes:)`

Creates a dataset that contains the unique elements of `input_dataset`.

``` swift
@inlinable @inline(__always) public static func uniqueDataset(inputDataset: VariantHandle, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

### `uniqueV2(_:axis:)`

Finds unique elements along an axis of a tensor.

``` swift
@inlinable @inline(__always) public static func uniqueV2<T: TensorFlowScalar, Taxis: TensorFlowIndex, OutIdx: TensorFlowIndex>(_ x: Tensor<T>, axis: Tensor<Taxis>) -> (y: Tensor<T>, idx: Tensor<OutIdx>)
```

This operation either returns a tensor `y` containing unique elements
along the `axis` of a tensor. The returned unique elements is sorted
in the same order as they occur along `axis` in `x`.
This operation also returns a tensor `idx` that is the same size as
the number of the elements in `x` along the `axis` dimension. It
contains the index in the unique output `y`.
In other words, for an `1-D` tensor `x` with \`axis = None:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

For example:

``` 
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx = unique(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
```

For an `2-D` tensor `x` with `axis = 0`:

``` 
# tensor 'x' is [[1, 0, 0],
#                [1, 0, 0],
#                [2, 0, 0]]
y, idx = unique(x, axis=0)
y ==> [[1, 0, 0],
       [2, 0, 0]]
idx ==> [0, 0, 1]
```

For an `2-D` tensor `x` with `axis = 1`:

``` 
# tensor 'x' is [[1, 0, 0],
#                [1, 0, 0],
#                [2, 0, 0]]
y, idx = unique(x, axis=1)
y ==> [[1, 0],
       [1, 0],
       [2, 0]]
idx ==> [0, 1, 1]
```

#### Parameters

  - x: - x: A `Tensor`.
  - axis: - axis: A `Tensor` of type `int32` (default: None). The axis of the Tensor to find the unique elements.

### `uniqueWithCounts(_:)`

Finds unique elements in a 1-D tensor.

``` swift
@inlinable @inline(__always) public static func uniqueWithCounts<T: TensorFlowScalar, OutIdx: TensorFlowIndex>(_ x: Tensor<T>) -> (y: Tensor<T>, idx: Tensor<OutIdx>, count: Tensor<OutIdx>)
```

This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. Finally, it returns a third tensor `count` that
contains the count of each element of `y` in `x`. In other words:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

For example:

``` 
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx, count = unique_with_counts(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
count ==> [2, 1, 3, 1, 2]
```

#### Parameters

  - x: - x: 1-D.

### `uniqueWithCountsV2(_:axis:)`

Finds unique elements along an axis of a tensor.

``` swift
@inlinable @inline(__always) public static func uniqueWithCountsV2<T: TensorFlowScalar, Taxis: TensorFlowIndex, OutIdx: TensorFlowIndex>(_ x: Tensor<T>, axis: Tensor<Taxis>) -> (y: Tensor<T>, idx: Tensor<OutIdx>, count: Tensor<OutIdx>)
```

This operation either returns a tensor `y` containing unique elements
along the `axis` of a tensor. The returned unique elements is sorted
in the same order as they occur along `axis` in `x`.
This operation also returns a tensor `idx` and a tensor `count`
that are the same size as the number of the elements in `x` along the
`axis` dimension. The `idx` contains the index in the unique output `y`
and the `count` contains the count in the unique output `y`.
In other words, for an `1-D` tensor `x` with \`axis = None:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

For example:

``` 
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx, count = unique_with_counts(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
count ==> [2, 1, 3, 1, 2]
```

For an `2-D` tensor `x` with `axis = 0`:

``` 
# tensor 'x' is [[1, 0, 0],
#                [1, 0, 0],
#                [2, 0, 0]]
y, idx, count = unique_with_counts(x, axis=0)
y ==> [[1, 0, 0],
       [2, 0, 0]]
idx ==> [0, 0, 1]
count ==> [2, 1]
```

For an `2-D` tensor `x` with `axis = 1`:

``` 
# tensor 'x' is [[1, 0, 0],
#                [1, 0, 0],
#                [2, 0, 0]]
y, idx, count = unique_with_counts(x, axis=1)
y ==> [[1, 0],
       [1, 0],
       [2, 0]]
idx ==> [0, 1, 1]
count ==> [1, 2]
```

#### Parameters

  - x: - x: A `Tensor`.
  - axis: - axis: A `Tensor` of type `int32` (default: None). The axis of the Tensor to find the unique elements.

### `unpack(value:num:axis:)`

Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.

``` swift
@inlinable @inline(__always) public static func unpack<T: TensorFlowScalar>(value: Tensor<T>, num: Int64, axis: Int64 = 0) -> [Tensor<T>]
```

Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
For example, given a tensor of shape `(A, B, C, D)`;

If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
and each tensor in `output` will have shape `(B, C, D)`. (Note that the
dimension unpacked along is gone, unlike `split`).

If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
and each tensor in `output` will have shape `(A, C, D)`.
Etc.

This is the opposite of `pack`.

#### Parameters

  - value: - value: 1-D or higher, with `axis` dimension size equal to `num`.

### `unravelIndex(indices:dims:)`

Converts an array of flat indices into a tuple of coordinate arrays.

``` swift
@inlinable @inline(__always) public static func unravelIndex<Tidx: TensorFlowIndex>(indices: Tensor<Tidx>, dims: Tensor<Tidx>) -> Tensor<Tidx>
```

Example:

``` 
y = tf.unravel_index(indices=[2, 5, 7], dims=[3, 3])
# 'dims' represent a hypothetical (3, 3) tensor of indices:
# [[0, 1, *2*],
#  [3, 4, *5*],
#  [6, *7*, 8]]
# For each entry from 'indices', this operation returns
# its coordinates (marked with '*'), such as
# 2 ==> (0, 2)
# 5 ==> (1, 2)
# 7 ==> (2, 1)
y ==> [[0, 1, 2], [2, 2, 1]]
```

@compatibility(numpy)
Equivalent to np.unravel\_index
@end\_compatibility

#### Parameters

  - indices: - indices: An 0-D or 1-D `int` Tensor whose elements are indices into the flattened version of an array of dimensions dims.
  - dims: - dims: An 1-D `int` Tensor. The shape of the array to use for unraveling indices.

### `unsortedSegmentJoin(inputs:segmentIds:numSegments:separator:)`

Joins the elements of `inputs` based on `segment_ids`.

``` swift
@inlinable @inline(__always) public static func unsortedSegmentJoin<Tindices: TensorFlowIndex, Tnumsegments: TensorFlowIndex>(inputs: StringTensor, segmentIds: Tensor<Tindices>, numSegments: Tensor<Tnumsegments>, separator: String) -> StringTensor
```

Computes the string join along segments of a tensor.
Given `segment_ids` with rank `N` and `data` with rank `N+M`:

``` 
`output[i, k1...kM] = strings.join([data[j1...jN, k1...kM])`
```

where the join is over all \[j1...jN\] such that segment\_ids\[j1...jN\] = i.
Strings are joined in row-major order.

For example:

``` python
inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
output_array = string_ops.unsorted_segment_join(inputs=inputs,
                                                segment_ids=[1, 0, 1],
                                                num_segments=2,
                                                separator=':'))
# output_array ==> [['Y', '6', '6'], ['Y:p', 'q:G', 'c:a']]
 
 
inputs = ['this', 'is', 'a', 'test']
output_array = string_ops.unsorted_segment_join(inputs=inputs,
                                                segment_ids=[0, 0, 0, 0],
                                                num_segments=1,
                                                separator=':'))
# output_array ==> ['this:is:a:test']
```

#### Parameters

  - inputs: - inputs: The input to be joined.

### `unsortedSegmentMax(data:segmentIds:numSegments:)`

Computes the maximum along segments of a tensor.

``` swift
@inlinable @inline(__always) public static func unsortedSegmentMax<T: TensorFlowNumeric, Tindices: TensorFlowIndex, Tnumsegments: TensorFlowIndex>(data: Tensor<T>, segmentIds: Tensor<Tindices>, numSegments: Tensor<Tnumsegments>) -> Tensor<T>
```

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

This operator is similar to the unsorted segment sum operator found
[(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
Instead of computing the sum over segments, it computes the maximum such that:

\\(output\_i = \\max\_{j...} data\[j...\]\\) where max is over tuples `j...` such
that `segment_ids[j...] == i`.

If the maximum is empty for a given segment ID `i`, it outputs the smallest
possible value for the specific numeric type,
`output[i] = numeric_limits<T>::lowest()`.

If the given segment ID `i` is negative, then the corresponding value is
dropped, and will not be included in the result.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentMax.png" alt>
</div>

For example:

``` python
c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
tf.unsorted_segment_max(c, tf.constant([0, 1, 0]), num_segments=2)
# ==> [[ 4,  3, 3, 4],
#       [5,  6, 7, 8]]
```

### `unsortedSegmentMin(data:segmentIds:numSegments:)`

Computes the minimum along segments of a tensor.

``` swift
@inlinable @inline(__always) public static func unsortedSegmentMin<T: TensorFlowNumeric, Tindices: TensorFlowIndex, Tnumsegments: TensorFlowIndex>(data: Tensor<T>, segmentIds: Tensor<Tindices>, numSegments: Tensor<Tnumsegments>) -> Tensor<T>
```

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

This operator is similar to the unsorted segment sum operator found
[(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
Instead of computing the sum over segments, it computes the minimum such that:

\\(output\_i = \\min\_{j...} data\_\[j...\]\\) where min is over tuples `j...` such
that `segment_ids[j...] == i`.

If the minimum is empty for a given segment ID `i`, it outputs the largest
possible value for the specific numeric type,
`output[i] = numeric_limits<T>::max()`.

For example:

``` python
c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
tf.unsorted_segment_min(c, tf.constant([0, 1, 0]), num_segments=2)
# ==> [[ 1,  2, 2, 1],
#       [5,  6, 7, 8]]
```

If the given segment ID `i` is negative, then the corresponding value is
dropped, and will not be included in the result.

### `unsortedSegmentProd(data:segmentIds:numSegments:)`

Computes the product along segments of a tensor.

``` swift
@inlinable @inline(__always) public static func unsortedSegmentProd<T: TensorFlowNumeric, Tindices: TensorFlowIndex, Tnumsegments: TensorFlowIndex>(data: Tensor<T>, segmentIds: Tensor<Tindices>, numSegments: Tensor<Tnumsegments>) -> Tensor<T>
```

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

This operator is similar to the unsorted segment sum operator found
[(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
Instead of computing the sum over segments, it computes the product of all
entries belonging to a segment such that:

\\(output\_i = \\prod\_{j...} data\[j...\]\\) where the product is over tuples
`j...` such that `segment_ids[j...] == i`.

For example:

``` python
c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
tf.unsorted_segment_prod(c, tf.constant([0, 1, 0]), num_segments=2)
# ==> [[ 4,  6, 6, 4],
#       [5,  6, 7, 8]]
```

If there is no entry for a given segment ID `i`, it outputs 1.

If the given segment ID `i` is negative, then the corresponding value is
dropped, and will not be included in the result.

### `unsortedSegmentSum(data:segmentIds:numSegments:)`

Computes the sum along segments of a tensor.

``` swift
@inlinable @inline(__always) public static func unsortedSegmentSum<T: TensorFlowNumeric, Tindices: TensorFlowIndex, Tnumsegments: TensorFlowIndex>(data: Tensor<T>, segmentIds: Tensor<Tindices>, numSegments: Tensor<Tnumsegments>) -> Tensor<T>
```

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output\[i\] = \\sum\_{j...} data\[j...\]\\) where the sum is over tuples `j...` such
that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
need not be sorted and need not cover all values in the full
range of valid values.

If the sum is empty for a given segment ID `i`, `output[i] = 0`.
If the given segment ID `i` is negative, the value is dropped and will not be
added to the sum of the segment.

`num_segments` should equal the number of distinct segment IDs.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
</div>

``` python
c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
tf.unsorted_segment_sum(c, tf.constant([0, 1, 0]), num_segments=2)
# ==> [[ 5,  5, 5, 5],
#       [5,  6, 7, 8]]
```

### `unstage(capacity:memoryLimit:container:sharedName:)`

Op is similar to a lightweight Dequeue.

``` swift
@inlinable @inline(__always) public static func unstage<Dtypes: TensorGroup>(capacity: Int64 = 0, memoryLimit: Int64 = 0, container: String, sharedName: String) -> Dtypes
```

The basic functionality is similar to dequeue with many fewer
capabilities and options.  This Op is optimized for performance.

### `unwrapDatasetVariant(inputHandle:)`

``` swift
@inlinable @inline(__always) public static func unwrapDatasetVariant(inputHandle: VariantHandle) -> VariantHandle
```

### `upperBound(sortedInputs:_:)`

Applies upper\_bound(sorted\_search\_values, values) along each row.

``` swift
@inlinable @inline(__always) public static func upperBound<T: TensorFlowScalar, OutType: TensorFlowIndex>(sortedInputs: Tensor<T>, _ values: Tensor<T>) -> Tensor<OutType>
```

Each set of rows with the same index in (sorted\_inputs, values) is treated
independently.  The resulting row is the equivalent of calling
`np.searchsorted(sorted_inputs, values, side='right')`.

The result is not a global index to the entire
`Tensor`, but rather just the index in the last dimension.

A 2-D example:
sorted\_sequence = \[\[0, 3, 9, 9, 10\],
\[1, 2, 3, 4, 5\]\]
values = \[\[2, 4, 9\],
\[0, 2, 6\]\]

result = UpperBound(sorted\_sequence, values)

result == \[\[1, 2, 4\],
\[0, 2, 5\]\]

#### Parameters

  - values: - values: 2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains the values that will be searched for in `sorted_search_values`.

### `varHandleOp(container:sharedName:dtype:shape:)`

Creates a handle to a Variable resource.

``` swift
@inlinable @inline(__always) public static func varHandleOp(container: String, sharedName: String, dtype: TensorDataType, shape: TensorShape?) -> ResourceHandle
```

### `varIsInitializedOp(resource:)`

Checks whether a resource handle-based variable has been initialized.

``` swift
@inlinable @inline(__always) public static func varIsInitializedOp(resource: ResourceHandle) -> Tensor<Bool>
```

#### Parameters

  - resource: - resource: the input resource handle.

### `variableShape(_:)`

Returns the shape of the variable pointed to by `resource`.

``` swift
@inlinable @inline(__always) public static func variableShape<OutType: TensorFlowIndex>(_ input: ResourceHandle) -> Tensor<OutType>
```

This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

``` 
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```

### `where_(_:)`

Returns locations of nonzero / true values in a tensor.

``` swift
@inlinable @inline(__always) public static func where_<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<Int64>
```

This operation returns the coordinates of true elements in `condition`. The
coordinates are returned in a 2-D tensor where the first dimension (rows)
represents the number of true elements, and the second dimension (columns)
represents the coordinates of the true elements. Keep in mind, the shape of
the output tensor can vary depending on how many true values there are in
`condition`. Indices are output in row-major order.

For example:

``` 
# 'input' tensor is [[True, False]
#                    [True, False]]
# 'input' has two true values, so output has two coordinates.
# 'input' has rank of 2, so coordinates have two indices.
where(input) ==> [[0, 0],
                  [1, 0]]
 
# `condition` tensor is [[[True, False]
#                     [True, False]]
#                    [[False, True]
#                     [False, True]]
#                    [[False, False]
#                     [False, True]]]
# 'input' has 5 true values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]
 
# `condition` tensor is [[[1.5,  0.0]
#                     [-0.5, 0.0]]
#                    [[0.0,  0.25]
#                     [0.0,  0.75]]
#                    [[0.0,  0.0]
#                     [0.0,  0.01]]]
# 'input' has 5 nonzero values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]
 
# `condition` tensor is [[[1.5 + 0.0j, 0.0  + 0.0j]
#                     [0.0 + 0.5j, 0.0  + 0.0j]]
#                    [[0.0 + 0.0j, 0.25 + 1.5j]
#                     [0.0 + 0.0j, 0.75 + 0.0j]]
#                    [[0.0 + 0.0j, 0.0  + 0.0j]
#                     [0.0 + 0.0j, 0.01 + 0.0j]]]
# 'input' has 5 nonzero magnitude values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]
```

### `while_(_:cond:body:outputShapes:parallelIterations:)`

output = input; While (Cond(output)) { output = Body(output) }

``` swift
@inlinable @inline(__always) public static func while_<T: TensorArrayProtocol, CondIn: TensorGroup, CondOut: TensorGroup, BodyIn: TensorGroup, BodyOut: TensorGroup>(_ input: T, cond: (CondIn) -> CondOut, body: (BodyIn) -> BodyOut, outputShapes: [TensorShape?], parallelIterations: Int64 = 10) -> T
```

#### Parameters

  - input: - input: A list of input tensors whose types are T.

### `wholeFileReaderV2(container:sharedName:)`

A Reader that outputs the entire contents of a file as a value.

``` swift
@inlinable @inline(__always) public static func wholeFileReaderV2(container: String, sharedName: String) -> ResourceHandle
```

To use, enqueue filenames in a Queue.  The output of ReaderRead will
be a filename (key) and the contents of that file (value).

### `windowDataset(inputDataset:size:shift:stride:dropRemainder:outputTypes:outputShapes:)`

A dataset that creates window datasets from the input dataset.

``` swift
@inlinable @inline(__always) public static func windowDataset(inputDataset: VariantHandle, size: Tensor<Int64>, shift: Tensor<Int64>, stride: Tensor<Int64>, dropRemainder: Tensor<Bool>, outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

#### Parameters

  - size: - size: A scalar representing the number of elements to accumulate in a window.
  - shift: - shift: A scalar representing the steps moving the sliding window forward in one iteration. It must be positive.
  - stride: - stride: A scalar representing the stride of the input elements of the sliding window. It must be positive.

### `workerHeartbeat(request:)`

Worker heartbeat op.

``` swift
@inlinable @inline(__always) public static func workerHeartbeat(request: StringTensor) -> StringTensor
```

Heartbeats may be sent periodically to indicate the coordinator is still active,
to retrieve the current worker status and to expedite shutdown when necessary.

#### Parameters

  - request: - request: A string tensor containing a serialized WorkerHeartbeatRequest

### `wrapDatasetVariant(inputHandle:)`

``` swift
@inlinable @inline(__always) public static func wrapDatasetVariant(inputHandle: VariantHandle) -> VariantHandle
```

### `writeAudioSummary(writer:step:tag:_:sampleRate:maxOutputs:)`

``` swift
@inlinable @inline(__always) public static func writeAudioSummary(writer: ResourceHandle, step: Tensor<Int64>, tag: StringTensor, _ tensor: Tensor<Float>, sampleRate: Tensor<Float>, maxOutputs: Int64 = 3)
```

### `writeFile(filename:contents:)`

Writes contents to the file at input filename. Creates file and recursively

``` swift
@inlinable @inline(__always) public static func writeFile(filename: StringTensor, contents: StringTensor)
```

creates directory if not existing.

#### Parameters

  - filename: - filename: scalar. The name of the file to which we write the contents.
  - contents: - contents: scalar. The content to be written to the output file.

### `writeGraphSummary(writer:step:_:)`

``` swift
@inlinable @inline(__always) public static func writeGraphSummary(writer: ResourceHandle, step: Tensor<Int64>, _ tensor: StringTensor)
```

### `writeHistogramSummary(writer:step:tag:_:)`

``` swift
@inlinable @inline(__always) public static func writeHistogramSummary<T: TensorFlowNumeric>(writer: ResourceHandle, step: Tensor<Int64>, tag: StringTensor, _ values: Tensor<T>)
```

### `writeImageSummary(writer:step:tag:_:badColor:maxImages:)`

``` swift
@inlinable @inline(__always) public static func writeImageSummary<T: TensorFlowNumeric>(writer: ResourceHandle, step: Tensor<Int64>, tag: StringTensor, _ tensor: Tensor<T>, badColor: Tensor<UInt8>, maxImages: Int64 = 3)
```

### `writeRawProtoSummary(writer:step:_:)`

``` swift
@inlinable @inline(__always) public static func writeRawProtoSummary(writer: ResourceHandle, step: Tensor<Int64>, _ tensor: StringTensor)
```

### `writeScalarSummary(writer:step:tag:value:)`

``` swift
@inlinable @inline(__always) public static func writeScalarSummary<T: TensorFlowNumeric>(writer: ResourceHandle, step: Tensor<Int64>, tag: StringTensor, value: Tensor<T>)
```

### `writeSummary(writer:step:_:tag:summaryMetadata:)`

``` swift
@inlinable @inline(__always) public static func writeSummary<T: TensorFlowScalar>(writer: ResourceHandle, step: Tensor<Int64>, _ tensor: Tensor<T>, tag: StringTensor, summaryMetadata: StringTensor)
```

### `xdivy(_:_:)`

Returns 0 if x == 0, and x / y otherwise, elementwise.

``` swift
@inlinable @inline(__always) public static func xdivy<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

### `xlaBroadcastHelper(lhs:rhs:broadcastDims:)`

Helper operator for performing XLA-style broadcasts

``` swift
@inlinable @inline(__always) public static func xlaBroadcastHelper<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(lhs: Tensor<T>, rhs: Tensor<T>, broadcastDims: Tensor<Tindices>) -> (lhsOutput: Tensor<T>, rhsOutput: Tensor<T>)
```

Broadcasts `lhs` and `rhs` to the same rank, by adding size 1 dimensions to
whichever of `lhs` and `rhs` has the lower rank, using XLA's broadcasting rules
for binary operators.

#### Parameters

  - lhs: - lhs: the LHS input tensor
  - rhs: - rhs: the RHS input tensor

### `xlaConv(lhs:rhs:windowStrides:padding:lhsDilation:rhsDilation:featureGroupCount:dimensionNumbers:precisionConfig:)`

Wraps the XLA ConvGeneralDilated operator, documented at

``` swift
@inlinable @inline(__always) public static func xlaConv<T: TensorFlowNumeric, Tindices: TensorFlowIndex>(lhs: Tensor<T>, rhs: Tensor<T>, windowStrides: Tensor<Tindices>, padding: Tensor<Tindices>, lhsDilation: Tensor<Tindices>, rhsDilation: Tensor<Tindices>, featureGroupCount: Tensor<Tindices>, dimensionNumbers: String, precisionConfig: String) -> Tensor<T>
```

https://www.tensorflow.org/performance/xla/operation\_semantics\#conv\_convolution
.

#### Parameters

  - lhs: - lhs: the input tensor
  - rhs: - rhs: the kernel tensor
  - padding: - padding: the padding to apply at the start and end of each input dimensions

### `xlaDot(lhs:rhs:dimensionNumbers:precisionConfig:)`

Wraps the XLA DotGeneral operator, documented at

``` swift
@inlinable @inline(__always) public static func xlaDot<T: TensorFlowNumeric>(lhs: Tensor<T>, rhs: Tensor<T>, dimensionNumbers: String, precisionConfig: String) -> Tensor<T>
```

https://www.tensorflow.org/performance/xla/operation\_semantics\#dotgeneral
.

#### Parameters

  - lhs: - lhs: the LHS tensor
  - rhs: - rhs: the RHS tensor

### `xlaDynamicSlice(_:startIndices:sizeIndices:)`

Wraps the XLA DynamicSlice operator, documented at

``` swift
@inlinable @inline(__always) public static func xlaDynamicSlice<T: TensorFlowScalar, Tindices: TensorFlowIndex>(_ input: Tensor<T>, startIndices: Tensor<Tindices>, sizeIndices: Tensor<Tindices>) -> Tensor<T>
```

https://www.tensorflow.org/performance/xla/operation\_semantics\#dynamicslice
.

DynamicSlice extracts a sub-array from the input array at dynamic
start\_indices. The size of the slice in each dimension is passed in
size\_indices, which specify the end point of exclusive slice intervals in each
dimension -- \[start, start + size). The shape of start\_indices must have rank 1,
with dimension size equal to the rank of operand.

#### Parameters

  - input: - input: A `Tensor` of type T.

### `xlaDynamicUpdateSlice(_:update:indices:)`

Wraps the XLA DynamicUpdateSlice operator, documented at

``` swift
@inlinable @inline(__always) public static func xlaDynamicUpdateSlice<T: TensorFlowScalar, Tindices: TensorFlowIndex>(_ input: Tensor<T>, update: Tensor<T>, indices: Tensor<Tindices>) -> Tensor<T>
```

https://www.tensorflow.org/performance/xla/operation\_semantics\#dynamicupdateslice
.

XlaDynamicUpdateSlice generates a result which is the value of the `input`
operand, with a slice update overwritten at `indices`. The shape of `update`
determines the shape of the sub-array of the result which is updated. The shape
of indices must be rank == 1, with dimension size equal to the rank of `input`.

Handling of out-of-bounds slice indices is implementation-defined.

#### Parameters

  - input: - input: A `Tensor` of type T.
  - update: - update: A `Tensor` of type T. Same rank as `input`.
  - indices: - indices: A vector of indices into `input`. Must have length equal to the rank of `input`.

### `xlaEinsum(_:_:equation:)`

An op which supports basic einsum op with 2 inputs and 1 output.

``` swift
@inlinable @inline(__always) public static func xlaEinsum<T: FloatingPoint & TensorFlowScalar>(_ a: Tensor<T>, _ b: Tensor<T>, equation: String) -> Tensor<T>
```

This op has better TPU performnce since it doesn't have explicitly reshape and
transpose operations as tf.einsum does.

### `xlaIf(cond:inputs:thenBranch:elseBranch:)`

output = cond ? then\_branch(inputs) : else\_branch(inputs).

``` swift
@inlinable @inline(__always) public static func xlaIf<Tcond: TensorFlowScalar, ThenbranchIn: TensorGroup, ThenbranchOut: TensorGroup, ElsebranchIn: TensorGroup, ElsebranchOut: TensorGroup, Tin: TensorArrayProtocol, Tout: TensorGroup>(cond: Tensor<Tcond>, inputs: Tin, thenBranch: (ThenbranchIn) -> ThenbranchOut, elseBranch: (ElsebranchIn) -> ElsebranchOut) -> Tout
```

#### Parameters

  - cond: - cond: A boolean scalar.
  - inputs: - inputs: A list of input tensors.

### `xlaKeyValueSort(keys:_:)`

Wraps the XLA Sort operator, documented at

``` swift
@inlinable @inline(__always) public static func xlaKeyValueSort<K: TensorFlowNumeric, V: TensorFlowScalar>(keys: Tensor<K>, _ values: Tensor<V>) -> (sortedKeys: Tensor<K>, sortedValues: Tensor<V>)
```

https://www.tensorflow.org/performance/xla/operation\_semantics\#sort
.

Sorts a tensor. Currently only sorts in ascending order are supported.

#### Parameters

  - keys: - keys: A `Tensor` of type K.
  - values: - values: A `Tensor` of type V.

### `xlaPad(_:paddingValue:paddingLow:paddingHigh:paddingInterior:)`

Wraps the XLA Pad operator, documented at

``` swift
@inlinable @inline(__always) public static func xlaPad<T: TensorFlowScalar, Tindices: TensorFlowIndex>(_ input: Tensor<T>, paddingValue: Tensor<T>, paddingLow: Tensor<Tindices>, paddingHigh: Tensor<Tindices>, paddingInterior: Tensor<Tindices>) -> Tensor<T>
```

https://www.tensorflow.org/performance/xla/operation\_semantics\#pad
.

#### Parameters

  - input: - input: A `Tensor` of type T.

### `xlaRecv(tensorName:shape:)`

Receives the named tensor from another XLA computation. Wraps the XLA Recv

``` swift
@inlinable @inline(__always) public static func xlaRecv<Dtype: TensorFlowScalar>(tensorName: String, shape: TensorShape?) -> Tensor<Dtype>
```

operator documented at
https://www.tensorflow.org/performance/xla/operation\_semantics\#recv .

### `xlaReduce(_:initValue:dimensionsToReduce:reducer:)`

Wraps the XLA Reduce operator, documented at

``` swift
@inlinable @inline(__always) public static func xlaReduce<T: TensorFlowNumeric, ReducerIn: TensorGroup, ReducerOut: TensorGroup>(_ input: Tensor<T>, initValue: Tensor<T>, dimensionsToReduce: [Int32], reducer: (ReducerIn) -> ReducerOut) -> Tensor<T>
```

https://www.tensorflow.org/performance/xla/operation\_semantics\#reduce .

#### Parameters

  - input: - input: the input tensor

### `xlaReduceWindow(_:initValue:windowDimensions:windowStrides:baseDilations:windowDilations:padding:computation:)`

Wraps the XLA ReduceWindow operator, documented at

``` swift
@inlinable @inline(__always) public static func xlaReduceWindow<T: TensorFlowNumeric, Tindices: TensorFlowIndex, ComputationIn: TensorGroup, ComputationOut: TensorGroup>(_ input: Tensor<T>, initValue: Tensor<T>, windowDimensions: Tensor<Tindices>, windowStrides: Tensor<Tindices>, baseDilations: Tensor<Tindices>, windowDilations: Tensor<Tindices>, padding: Tensor<Tindices>, computation: (ComputationIn) -> ComputationOut) -> Tensor<T>
```

https://www.tensorflow.org/performance/xla/operation\_semantics\#reducewindow .

#### Parameters

  - input: - input: the input tensor
  - padding: - padding: the padding to apply at the start and end of each input dimensions

### `xlaReplicaId()`

Replica ID.

``` swift
@inlinable @inline(__always) public static func xlaReplicaId() -> Tensor<Int32>
```

### `xlaSelectAndScatter(operand:windowDimensions:windowStrides:padding:source:initValue:select:scatter:)`

Wraps the XLA SelectAndScatter operator, documented at

``` swift
@inlinable @inline(__always) public static func xlaSelectAndScatter<T: TensorFlowNumeric, Tindices: TensorFlowIndex, SelectIn: TensorGroup, SelectOut: TensorGroup, ScatterIn: TensorGroup, ScatterOut: TensorGroup>(operand: Tensor<T>, windowDimensions: Tensor<Tindices>, windowStrides: Tensor<Tindices>, padding: Tensor<Tindices>, source: Tensor<T>, initValue: Tensor<T>, select: (SelectIn) -> SelectOut, scatter: (ScatterIn) -> ScatterOut) -> Tensor<T>
```

https://www.tensorflow.org/performance/xla/operation\_semantics\#selectandscatter
.

#### Parameters

  - operand: - operand: the input tensor
  - padding: - padding: the padding to apply at the start and end of each input dimensions
  - source: - source: a tensor of values to scatter

### `xlaSelfAdjointEig(_:lower:maxIter:epsilon:)`

Computes the eigen decomposition of a batch of self-adjoint matrices

``` swift
@inlinable @inline(__always) public static func xlaSelfAdjointEig<T: TensorFlowNumeric>(_ a: Tensor<T>, lower: Bool, maxIter: Int64, epsilon: Double) -> (w: Tensor<T>, v: Tensor<T>)
```

(Note: Only real inputs are supported).

Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices in
tensor such that tensor\[...,:,:\] \* v\[..., :,i\] = e\[..., i\] \* v\[...,:,i\], for
i=0...N-1.

#### Parameters

  - a: - a: the input tensor.

### `xlaSend(_:tensorName:)`

Sends the named tensor to another XLA computation. Wraps the XLA Send operator

``` swift
@inlinable @inline(__always) public static func xlaSend<T: TensorFlowScalar>(_ tensor: Tensor<T>, tensorName: String)
```

documented at
https://www.tensorflow.org/performance/xla/operation\_semantics\#send .

#### Parameters

  - tensor: - tensor: The tensor to send.

### `xlaSharding(_:)`

An op which shards the input based on the given sharding attribute.

``` swift
@inlinable @inline(__always) public static func xlaSharding<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

### `xlaSort(_:)`

Wraps the XLA Sort operator, documented at

``` swift
@inlinable @inline(__always) public static func xlaSort<T: TensorFlowScalar>(_ input: Tensor<T>) -> Tensor<T>
```

https://www.tensorflow.org/performance/xla/operation\_semantics\#sort
.

Sorts a tensor. Currently only sorts in ascending order are supported.

#### Parameters

  - input: - input: A `Tensor` of type T.

### `xlaSvd(_:maxIter:epsilon:precisionConfig:)`

Computes the eigen decomposition of a batch of self-adjoint matrices

``` swift
@inlinable @inline(__always) public static func xlaSvd<T: TensorFlowNumeric>(_ a: Tensor<T>, maxIter: Int64, epsilon: Double, precisionConfig: String) -> (s: Tensor<T>, u: Tensor<T>, v: Tensor<T>)
```

(Note: Only real inputs are supported).

Computes the eigenvalues and eigenvectors of the innermost M-by-N matrices in
tensor such that tensor\[...,:,:\] = u\[..., :, :\] \* Diag(s\[..., :\]) \* Transpose(v\[...,:,:\]).

#### Parameters

  - a: - a: the input tensor.

### `xlaWhile(_:cond:body:)`

output = input; While (Cond(output)) { output = Body(output) }

``` swift
@inlinable @inline(__always) public static func xlaWhile<T: TensorArrayProtocol, CondIn: TensorGroup, CondOut: TensorGroup, BodyIn: TensorGroup, BodyOut: TensorGroup>(_ input: T, cond: (CondIn) -> CondOut, body: (BodyIn) -> BodyOut) -> T
```

#### Parameters

  - input: - input: A list of input tensors whose types are T.

### `xlogy(_:_:)`

Returns 0 if x == 0, and x \* log(y) otherwise, elementwise.

``` swift
@inlinable @inline(__always) public static func xlogy<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T>
```

### `zerosLike(_:)`

Returns a tensor of zeros with the same shape and type as x.

``` swift
@inlinable @inline(__always) public static func zerosLike<T: TensorFlowScalar>(_ x: Tensor<T>) -> Tensor<T>
```

#### Parameters

  - x: - x: a tensor of type T.

### `zeta(_:q:)`

Compute the Hurwitz zeta function \\(\\zeta(x, q)\\).

``` swift
@inlinable @inline(__always) public static func zeta<T: FloatingPoint & TensorFlowScalar>(_ x: Tensor<T>, q: Tensor<T>) -> Tensor<T>
```

The Hurwitz zeta function is defined as:

\\(\\zeta(x, q) = \\sum\_{n=0}^{\\infty} (q + n)^{-x}\\)

### `zipDataset(inputDatasets:outputTypes:outputShapes:)`

Creates a dataset that zips together `input_datasets`.

``` swift
@inlinable @inline(__always) public static func zipDataset(inputDatasets: [VariantHandle], outputTypes: [TensorDataType], outputShapes: [TensorShape?]) -> VariantHandle
```

The elements of the resulting dataset are created by zipping corresponding
elements from each of the input datasets.

The size of the resulting dataset will match the size of the smallest input
dataset, and no error will be raised if input datasets have different sizes.

### `saveV2(prefix:tensorNames:shapeAndSlices:tensors:)`

Saves tensors in V2 checkpoint format.

``` swift
@inlinable public static func saveV2(prefix: StringTensor, tensorNames: StringTensor, shapeAndSlices: StringTensor, tensors: [AnyTensor])
```

By default, saves the named tensors in full.  If the caller wishes to save specific slices
of full tensors, "shape\_and\_slices" should be non-empty strings and correspondingly
well-formed.

#### Parameters

  - prefix: - prefix: Must have a single element. The prefix of the V2 checkpoint to which we write the tensors.
  - tensors: - tensors: `N` tensors to save.

### `restoreV2(prefix:tensorNames:shapeAndSlices:dtypes:)`

Restores tensors from a V2 checkpoint.

``` swift
@inlinable public static func restoreV2(prefix: StringTensor, tensorNames: StringTensor, shapeAndSlices: StringTensor, dtypes: [TensorDataType]) -> [AnyTensor]
```

For backward compatibility with the V1 format, this Op currently allows restoring from a V1
checkpoint as well:

By default, restores the named tensors in full.  If the caller wishes to restore specific
slices of stored tensors, "shape\_and\_slices" should be non-empty strings and correspondingly
well-formed.

Callers must ensure all the named tensors are indeed stored in the checkpoint.

#### Parameters

  - prefix: - prefix: Must have a single element.  The prefix of a V2 checkpoint.

### `toDevice(_:_:)`

``` swift
static func toDevice<T: TensorFlowScalar>(_ x: Tensor<T>, _ device: Device) -> Tensor<T>
```
