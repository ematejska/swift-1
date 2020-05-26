# Types

  - [PythonObject](/PythonObject):
    `PythonObject` represents an object in Python and supports dynamic member
    lookup. Any member access like `object.foo` will dynamically request the
    Python runtime for a member with the specified name in this object.
  - [PythonError](/PythonError):
    An error produced by a failable Python operation.
  - [ThrowingPythonObject](/ThrowingPythonObject):
    A `PythonObject` wrapper that enables throwing method calls.
    Exceptions produced by Python functions are reflected as Swift errors and
    thrown.
  - [CheckingPythonObject](/CheckingPythonObject):
    A `PythonObject` wrapper that enables member accesses.
    Member access operations return an `Optional` result. When member access
    fails, `nil` is returned.
  - [PythonInterface](/PythonInterface):
    An interface for Python.
  - [PythonObject.Iterator](/PythonObject_Iterator)
  - [PythonLibrary](/PythonLibrary)
  - [Array.DifferentiableView](/Array_DifferentiableView):
    The view of an array as the differentiable product manifold of `Element`
    multiplied with itself `count` times.
  - [AnyRandomNumberGenerator](/AnyRandomNumberGenerator):
    A type-erased random number generator.
  - [ARC4RandomNumberGenerator](/ARC4RandomNumberGenerator):
    An implementation of `SeedableRandomNumberGenerator` using ARC4.
  - [ThreefryRandomNumberGenerator](/ThreefryRandomNumberGenerator):
    An implementation of `SeedableRandomNumberGenerator` using Threefry.
    Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
    http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
  - [PhiloxRandomNumberGenerator](/PhiloxRandomNumberGenerator):
    An implementation of `SeedableRandomNumberGenerator` using Philox.
    Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
    http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
  - [UniformIntegerDistribution](/UniformIntegerDistribution)
  - [UniformFloatingPointDistribution](/UniformFloatingPointDistribution)
  - [NormalDistribution](/NormalDistribution)
  - [BetaDistribution](/BetaDistribution)
  - [\_Raw](/_Raw)
  - [\_Raw.A](/_Raw_A)
  - [\_Raw.DataFormat](/_Raw_DataFormat)
  - [\_Raw.DataFormat1](/_Raw_DataFormat1)
  - [\_Raw.DataFormat5](/_Raw_DataFormat5)
  - [\_Raw.DensityUnit](/_Raw_DensityUnit)
  - [\_Raw.Direction](/_Raw_Direction)
  - [\_Raw.Errors](/_Raw_Errors)
  - [\_Raw.FinalOp](/_Raw_FinalOp)
  - [\_Raw.Format](/_Raw_Format)
  - [\_Raw.InputMode](/_Raw_InputMode)
  - [\_Raw.InputQuantMode](/_Raw_InputQuantMode)
  - [\_Raw.LossType](/_Raw_LossType)
  - [\_Raw.MergeOp](/_Raw_MergeOp)
  - [\_Raw.Method](/_Raw_Method)
  - [\_Raw.Method4](/_Raw_Method4)
  - [\_Raw.Mode](/_Raw_Mode)
  - [\_Raw.Mode6](/_Raw_Mode6)
  - [\_Raw.OutputEncoding](/_Raw_OutputEncoding)
  - [\_Raw.Padding](/_Raw_Padding)
  - [\_Raw.Padding2](/_Raw_Padding2)
  - [\_Raw.Reduction](/_Raw_Reduction)
  - [\_Raw.ReductionType](/_Raw_ReductionType)
  - [\_Raw.RnnMode](/_Raw_RnnMode)
  - [\_Raw.RoundMode](/_Raw_RoundMode)
  - [\_Raw.RoundMode7](/_Raw_RoundMode7)
  - [\_Raw.SplitType](/_Raw_SplitType)
  - [\_Raw.SplitType2](/_Raw_SplitType2)
  - [\_Raw.Unit](/_Raw_Unit)
  - [\_TensorFunctionPointer](/_TensorFunctionPointer):
    Opaque reference to a function that has been made callable by loading it
    into the runtime.
  - [LearningPhase](/LearningPhase):
    A value that indicates the phase of using a machine learning model.
  - [Context](/Context):
    A context that stores thread-local contextual information used by deep learning APIs such as
    layers.
  - [BroadcastingPullback](/BroadcastingPullback):
    A pullback function that performs the transpose of broadcasting two `Tensors`.
  - [TensorDataType](/TensorDataType):
    A TensorFlow dynamic type value that can be created from types that conform to
    `TensorFlowScalar`.
  - [BFloat16](/BFloat16)
  - [Device](/Device):
    A device on which `Tensor`s can be allocated.
  - [DeviceKind](/DeviceKind):
    A TensorFlow device kind.
  - [\_RuntimeConfig](/_RuntimeConfig):
    The configuration for the compiler runtime.
  - [\_RuntimeConfig.RuntimeSession](/_RuntimeConfig_RuntimeSession):
    Specifies whether the TensorFlow computation runs in a local (in-process) session, or a
    remote session with the specified server definition.
  - [\_ExecutionContext](/_ExecutionContext):
    The host of any tensor computation.
  - [TensorFlowCheckpointReader](/TensorFlowCheckpointReader):
    A TensorFlow checkpoint file reader.
  - [ShapedArray](/ShapedArray):
    `ShapedArray` is a multi-dimensional array. It has a shape, which has type `[Int]` and defines
    the array dimensions, and uses a `TensorBuffer` internally as storage.
  - [ShapedArraySlice](/ShapedArraySlice):
    A contiguous slice of a `ShapedArray` or `ShapedArraySlice` instance.
  - [StringTensor](/StringTensor):
    `StringTensor` is a multi-dimensional array whose elements are `String`s.
  - [Tensor](/Tensor):
    A multidimensional array of elements that is a generalization of vectors and matrices to
    potentially higher dimensions.
  - [\_TensorElementLiteral](/_TensorElementLiteral):
    Represents a literal element for conversion to a `Tensor`.
  - [TFETensorHandle](/TFETensorHandle):
    Class wrapping a C pointer to a TensorHandle.  This class owns the
    TensorHandle and is responsible for destroying it.
  - [TensorHandle](/TensorHandle):
    `TensorHandle` is the type used by ops. It includes a `Scalar` type, which
    compiler internals can use to determine the datatypes of parameters when
    they are extracted into a tensor program.
  - [ResourceHandle](/ResourceHandle)
  - [VariantHandle](/VariantHandle)
  - [TensorShape](/TensorShape):
    A struct representing the shape of a tensor.
  - [\_Freezable](/_Freezable):
    A wrapper around a differentiable value with "freezable" derivatives.
  - [EmptyTangentVector](/EmptyTangentVector):
    An empty struct representing empty `TangentVector`s for parameterless layers.
  - [Parameter](/Parameter):
    A mutable, shareable, owning reference to a tensor.
  - [Conv1D](/Conv1D):
    A 1-D convolution layer (e.g. temporal convolution over a time-series).
  - [Conv2D](/Conv2D):
    A 2-D convolution layer (e.g. spatial convolution over images).
  - [Conv3D](/Conv3D):
    A 3-D convolution layer for spatial/spatio-temporal convolution over images.
  - [TransposedConv1D](/TransposedConv1D):
    A 1-D transposed convolution layer (e.g. temporal transposed convolution over images).
  - [TransposedConv2D](/TransposedConv2D):
    A 2-D transposed convolution layer (e.g. spatial transposed convolution over images).
  - [TransposedConv3D](/TransposedConv3D):
    A 3-D transposed convolution layer (e.g. spatial transposed convolution over images).
  - [DepthwiseConv2D](/DepthwiseConv2D):
    A 2-D depthwise convolution layer.
  - [ZeroPadding1D](/ZeroPadding1D):
    A layer for adding zero-padding in the temporal dimension.
  - [ZeroPadding2D](/ZeroPadding2D):
    A layer for adding zero-padding in the spatial dimensions.
  - [ZeroPadding3D](/ZeroPadding3D):
    A layer for adding zero-padding in the spatial/spatio-temporal dimensions.
  - [SeparableConv1D](/SeparableConv1D):
    A 1-D separable convolution layer.
  - [SeparableConv2D](/SeparableConv2D):
    A 2-D Separable convolution layer.
  - [Flatten](/Flatten):
    A flatten layer.
  - [Reshape](/Reshape):
    A reshape layer.
  - [Function](/Function):
    A layer that encloses a custom differentiable function.
  - [Dropout](/Dropout):
    A dropout layer.
  - [Embedding](/Embedding):
    An embedding layer.
  - [BatchNorm](/BatchNorm):
    A batch normalization layer.
  - [LayerNorm](/LayerNorm):
    A layer that applies layer normalization over a mini-batch of inputs.
  - [MaxPool1D](/MaxPool1D):
    A max pooling layer for temporal data.
  - [MaxPool2D](/MaxPool2D):
    A max pooling layer for spatial data.
  - [MaxPool3D](/MaxPool3D):
    A max pooling layer for spatial or spatio-temporal data.
  - [AvgPool1D](/AvgPool1D):
    An average pooling layer for temporal data.
  - [AvgPool2D](/AvgPool2D):
    An average pooling layer for spatial data.
  - [AvgPool3D](/AvgPool3D):
    An average pooling layer for spatial or spatio-temporal data.
  - [GlobalAvgPool1D](/GlobalAvgPool1D):
    A global average pooling layer for temporal data.
  - [GlobalAvgPool2D](/GlobalAvgPool2D):
    A global average pooling layer for spatial data.
  - [GlobalAvgPool3D](/GlobalAvgPool3D):
    A global average pooling layer for spatial and spatio-temporal data.
  - [GlobalMaxPool1D](/GlobalMaxPool1D):
    A global max pooling layer for temporal data.
  - [GlobalMaxPool2D](/GlobalMaxPool2D):
    A global max pooling layer for spatial data.
  - [GlobalMaxPool3D](/GlobalMaxPool3D):
    A global max pooling layer for spatial and spatio-temporal data.
  - [RNNCellInput](/RNNCellInput):
    An input to a recurrent neural network.
  - [RNNCellOutput](/RNNCellOutput):
    An output to a recurrent neural network.
  - [SimpleRNNCell](/SimpleRNNCell):
    A simple RNN cell.
  - [SimpleRNNCell.State](/SimpleRNNCell_State)
  - [LSTMCell](/LSTMCell):
    An LSTM cell.
  - [LSTMCell.State](/LSTMCell_State)
  - [GRUCell](/GRUCell):
    An GRU cell.
  - [GRUCell.State](/GRUCell_State)
  - [RNN](/RNN)
  - [Sequential](/Sequential):
    A layer that sequentially composes two or more other layers.
  - [LayerBuilder](/LayerBuilder)
  - [UpSampling1D](/UpSampling1D):
    An upsampling layer for 1-D inputs.
  - [UpSampling2D](/UpSampling2D):
    An upsampling layer for 2-D inputs.
  - [UpSampling3D](/UpSampling3D):
    An upsampling layer for 3-D inputs.
  - [Tensor.PaddingMode](/Tensor_PaddingMode):
    A mode that dictates how a tensor is padded.
  - [TensorRange](/TensorRange)
  - [Dataset](/Dataset):
    Represents a potentially large set of elements.
  - [DatasetIterator](/DatasetIterator):
    The type that allows iteration over a dataset's elements.
  - [Zip2TensorGroup](/Zip2TensorGroup):
    A 2-tuple-like struct that conforms to TensorGroup that represents a tuple of 2 types conforming
    to `TensorGroup`.
  - [ResizeMethod](/ResizeMethod):
    A resize algorithm.
  - [Moments](/Moments):
    Pair of first and second moments (i.e., mean and variance).
  - [Padding](/Padding):
    A padding scheme. Used by padding, convolution, and pooling ops.
  - [RMSProp](/RMSProp):
    RMSProp optimizer.
  - [AdaGrad](/AdaGrad):
    AdaGrad optimizer.
  - [AdaDelta](/AdaDelta):
    ADADELTA optimizer.
  - [Adam](/Adam):
    Adam optimizer.
  - [AdaMax](/AdaMax):
    AdaMax optimizer.
  - [AMSGrad](/AMSGrad):
    AMSGrad optimizer.
  - [RAdam](/RAdam):
    RAdam optimizer.
  - [SGD](/SGD):
    Stochastic gradient descent (SGD) optimizer.

# Protocols

  - [ConvertibleFromNumpyArray](/ConvertibleFromNumpyArray):
    A type that can be initialized from a `numpy.ndarray` instance represented
    as a `PythonObject`.
  - [NumpyScalarCompatible](/NumpyScalarCompatible):
    A type that is bitwise compatible with one or more NumPy scalar types.
  - [PythonConvertible](/PythonConvertible):
    A type whose values can be converted to a `PythonObject`.
  - [ConvertibleFromPython](/ConvertibleFromPython):
    A type that can be initialized from a `PythonObject`.
  - [Differentiable](/Differentiable):
    A type that mathematically represents a differentiable manifold whose
    tangent spaces are finite-dimensional.
  - [PointwiseMultiplicative](/PointwiseMultiplicative):
    A type with values that support pointwise multiplication.
  - [VectorProtocol](/VectorProtocol):
    A type that represents an unranked vector space. Values of this type are
    elements in this vector space and have either no shape or a static shape.
  - [EuclideanDifferentiable](/EuclideanDifferentiable):
    A type that is differentiable in the Euclidean space.
    The type may represent a vector space, or consist of a vector space and some
    other non-differentiable component.
  - [SeedableRandomNumberGenerator](/SeedableRandomNumberGenerator):
    A type that provides seedable deterministic pseudo-random data.
  - [RandomDistribution](/RandomDistribution)
  - [TensorOperation](/TensorOperation)
  - [TFTensorOperation](/TFTensorOperation)
  - [\_CopyableToDevice](/_CopyableToDevice):
    A type whose nested properties and elements can be copied to `Device`s.
  - [CopyableToDevice](/CopyableToDevice):
    A type whose nested properties and elements can be copied to a `Device`.
  - [\_TensorFlowDataTypeCompatible](/_TensorFlowDataTypeCompatible):
    A data type compatible with TensorFlow.
  - [TensorFlowScalar](/TensorFlowScalar):
    A scalar data type compatible with TensorFlow.
  - [TensorFlowIndex](/TensorFlowIndex):
    An integer data type that represents integer types which can be used as tensor indices in
    TensorFlow.
  - [TensorFlowFloatingPoint](/TensorFlowFloatingPoint):
    A floating-point data type that conforms to `Differentiable` and is compatible with TensorFlow.
  - [\_ShapedArrayProtocol](/_ShapedArrayProtocol)
  - [AnyTensor](/AnyTensor):
    Special protocol for calling tensorflow operations that take heterogeneous arrays as input.
  - [TensorArrayProtocol](/TensorArrayProtocol):
    A protocol representing types that can be mapped to `Array<CTensorHandle>`.
  - [TensorGroup](/TensorGroup):
    A protocol representing types that can be mapped to and from `Array<CTensorHandle>`.
  - [\_AnyTensorHandle](/_AnyTensorHandle):
    This protocol abstracts the underlying representation of a tensor. Any type
    that conforms to this protocol can be used as a `TensorHandle` in the
    `TensorFlow` library, as it much provide a way to convert the underlying tensor
    handle into a `ConcreteTensorHandle`, which wraps a `TFE_TensorHandle *`
    TODO(https://bugs.swift.org/browse/TF-527): This is defined as a class-bound
  - [Module](/Module)
  - [Layer](/Layer):
    A neural network layer.
  - [ParameterlessLayer](/ParameterlessLayer):
    A parameterless neural network layer.
  - [RNNCell](/RNNCell):
    A recurrent neural network cell.
  - [TensorRangeExpression](/TensorRangeExpression)
  - [Optimizer](/Optimizer):
    A numerical optimizer.

# Operators

  - [==(lhs:rhs:)](/==\(lhs:rhs:\))
  - [\!=(lhs:rhs:)](/!=\(lhs:rhs:\))
  - [+(lhs:rhs:)](/+\(lhs:rhs:\))
  - [-(lhs:rhs:)](/-\(lhs:rhs:\))
  - [==(\_:\_:)](/==\(_:_:\))

# Global Typealiases

  - [TangentVector](/TangentVector)
  - [TangentVector](/TangentVector)
  - [TensorFlowSeed](/TensorFlowSeed)
  - [Raw](/Raw)
  - [TensorFlowNumeric](/TensorFlowNumeric)
  - [TensorFlowSignedNumeric](/TensorFlowSignedNumeric)
  - [TensorFlowInteger](/TensorFlowInteger)
  - [CTensorHandle](/CTensorHandle):
    The `TF_TensorHandle *` type.
  - [Activation](/Activation):
    The element-wise activation function type.
  - [ParameterInitializer](/ParameterInitializer)
  - [SimpleRNN](/SimpleRNN)
  - [LSTM](/LSTM)

# Global Functions

  - [withoutDerivative(at:)](/withoutDerivative\(at:\)):
    Returns `x` like an identity function. When used in a context where `x` is
    being differentiated with respect to, this function will not produce any
    derivative at `x`.
  - [withoutDerivative(at:in:)](/withoutDerivative\(at:in:\)):
    Applies the given closure `body` to `x`. When used in a context where `x` is
    being differentiated with respect to, this function will not produce any
    derivative at `x`.
  - [differentiableFunction(from:)](/differentiableFunction\(from:\)):
    Create a differentiable function from a vector-Jacobian products function.
  - [differentiableFunction(from:)](/differentiableFunction\(from:\)):
    Create a differentiable function from a vector-Jacobian products function.
  - [withRecomputationInPullbacks(\_:)](/withRecomputationInPullbacks\(_:\)):
    Make a function be recomputed in its pullback, known as "checkpointing" in
    traditional automatic differentiation.
  - [transpose(of:)](/transpose\(of:\))
  - [valueWithDifferential(at:in:)](/valueWithDifferential\(at:in:\))
  - [valueWithDifferential(at:\_:in:)](/valueWithDifferential\(at:_:in:\))
  - [valueWithDifferential(at:\_:\_:in:)](/valueWithDifferential\(at:_:_:in:\))
  - [valueWithPullback(at:in:)](/valueWithPullback\(at:in:\))
  - [valueWithPullback(at:\_:in:)](/valueWithPullback\(at:_:in:\))
  - [valueWithPullback(at:\_:\_:in:)](/valueWithPullback\(at:_:_:in:\))
  - [differential(at:in:)](/differential\(at:in:\))
  - [differential(at:\_:in:)](/differential\(at:_:in:\))
  - [differential(at:\_:\_:in:)](/differential\(at:_:_:in:\))
  - [pullback(at:in:)](/pullback\(at:in:\))
  - [pullback(at:\_:in:)](/pullback\(at:_:in:\))
  - [pullback(at:\_:\_:in:)](/pullback\(at:_:_:in:\))
  - [derivative(at:in:)](/derivative\(at:in:\))
  - [derivative(at:\_:in:)](/derivative\(at:_:in:\))
  - [derivative(at:\_:\_:in:)](/derivative\(at:_:_:in:\))
  - [gradient(at:in:)](/gradient\(at:in:\))
  - [gradient(at:\_:in:)](/gradient\(at:_:in:\))
  - [gradient(at:\_:\_:in:)](/gradient\(at:_:_:in:\))
  - [valueWithDerivative(at:in:)](/valueWithDerivative\(at:in:\))
  - [valueWithDerivative(at:\_:in:)](/valueWithDerivative\(at:_:in:\))
  - [valueWithDerivative(at:\_:\_:in:)](/valueWithDerivative\(at:_:_:in:\))
  - [valueWithGradient(at:in:)](/valueWithGradient\(at:in:\))
  - [valueWithGradient(at:\_:in:)](/valueWithGradient\(at:_:in:\))
  - [valueWithGradient(at:\_:\_:in:)](/valueWithGradient\(at:_:_:in:\))
  - [derivative(of:)](/derivative\(of:\))
  - [derivative(of:)](/derivative\(of:\))
  - [derivative(of:)](/derivative\(of:\))
  - [gradient(of:)](/gradient\(of:\))
  - [gradient(of:)](/gradient\(of:\))
  - [gradient(of:)](/gradient\(of:\))
  - [valueWithDerivative(of:)](/valueWithDerivative\(of:\))
  - [valueWithDerivative(of:)](/valueWithDerivative\(of:\))
  - [valueWithDerivative(of:)](/valueWithDerivative\(of:\))
  - [valueWithGradient(of:)](/valueWithGradient\(of:\))
  - [valueWithGradient(of:)](/valueWithGradient\(of:\))
  - [valueWithGradient(of:)](/valueWithGradient\(of:\))
  - [move(along:)](/move\(along:\))
  - [move(along:)](/move\(along:\))
  - [\_printJVPErrorAndExit()](/_printJVPErrorAndExit\(\))
  - [randomSeedForTensorFlow(using:)](/randomSeedForTensorFlow\(using:\)):
    Generates a new random seed for TensorFlow.
  - [l1Loss(predicted:expected:)](/l1Loss\(predicted:expected:\)):
    Returns the L1 loss between predictions and expectations.
  - [l2Loss(predicted:expected:)](/l2Loss\(predicted:expected:\)):
    Returns the L2 loss between predictions and expectations.
  - [hingeLoss(predicted:expected:)](/hingeLoss\(predicted:expected:\)):
    Returns the hinge loss between predictions and expectations.
  - [squaredHingeLoss(predicted:expected:)](/squaredHingeLoss\(predicted:expected:\)):
    Returns the squared hinge loss between predictions and expectations.
  - [categoricalHingeLoss(predicted:expected:)](/categoricalHingeLoss\(predicted:expected:\)):
    Returns the categorical hinge loss between predictions and expectations.
  - [logCoshLoss(predicted:expected:)](/logCoshLoss\(predicted:expected:\)):
    Returns the logarithm of the hyperbolic cosine of the error between predictions and
    expectations.
  - [poissonLoss(predicted:expected:)](/poissonLoss\(predicted:expected:\)):
    Returns the Poisson loss between predictions and expectations.
  - [kullbackLeiblerDivergence(predicted:expected:)](/kullbackLeiblerDivergence\(predicted:expected:\)):
    Returns the Kullback-Leibler divergence (KL divergence) between between expectations and
    predictions. Given two distributions `p` and `q`, KL divergence computes `p * log(p / q)`.
  - [softmaxCrossEntropy(logits:probabilities:)](/softmaxCrossEntropy\(logits:probabilities:\)):
    Returns the softmax cross entropy (categorical cross entropy) between logits and labels.
  - [sigmoidCrossEntropy(logits:labels:)](/sigmoidCrossEntropy\(logits:labels:\)):
    Returns the sigmoid cross entropy (binary cross entropy) between logits and labels.
  - [withContext(\_:\_:)](/withContext\(_:_:\)):
    Calls the given closure within a context that has everything identical to the current context
    except for the given learning phase.
  - [withLearningPhase(\_:\_:)](/withLearningPhase\(_:_:\)):
    Calls the given closure within a context that has everything identical to the current context
    except for the given learning phase.
  - [withRandomSeedForTensorFlow(\_:\_:)](/withRandomSeedForTensorFlow\(_:_:\)):
    Calls the given closure within a context that has everything identical to the current context
    except for the given random seed.
  - [withRandomNumberGeneratorForTensorFlow(\_:\_:)](/withRandomNumberGeneratorForTensorFlow\(_:_:\)):
    Calls the given closure within a context that has everything identical to the current context
    except for the given random number generator.
  - [valueWithGradient(at:in:)](/valueWithGradient\(at:in:\))
  - [valueWithGradient(at:\_:in:)](/valueWithGradient\(at:_:in:\))
  - [valueWithGradient(at:\_:\_:in:)](/valueWithGradient\(at:_:_:in:\))
  - [valueWithGradient(of:)](/valueWithGradient\(of:\))
  - [valueWithGradient(of:)](/valueWithGradient\(of:\))
  - [valueWithGradient(of:)](/valueWithGradient\(of:\))
  - [gradient(at:in:)](/gradient\(at:in:\))
  - [gradient(at:\_:in:)](/gradient\(at:_:in:\))
  - [gradient(at:\_:\_:in:)](/gradient\(at:_:_:in:\))
  - [gradient(of:)](/gradient\(of:\))
  - [gradient(of:)](/gradient\(of:\))
  - [gradient(of:)](/gradient\(of:\))
  - [withDevice(\_:\_:perform:)](/withDevice\(_:_:perform:\)):
    Executes a closure, making TensorFlow operations run on a specific kind of device.
  - [withDevice(named:perform:)](/withDevice\(named:perform:\)):
    Executes a closure, making TensorFlow operations run on a device with
    a specific name.
  - [withDefaultDevice(perform:)](/withDefaultDevice\(perform:\)):
    Executes a closure, allowing TensorFlow to place TensorFlow operations on any device. This
    should restore the default placement behavior.
  - [\_graph(\_:useXLA:)](/_graph\(_:useXLA:\))
  - [\_tffunc(\_:)](/_tffunc\(_:\)):
    Trace the given function and return the name of the corresponding `TF_Function: In -> Out` that
    was created.
  - [\_runOnNDevices(\_:perform:)](/_runOnNDevices\(_:perform:\))
  - [callAsFunction(\_:)](/callAsFunction\(_:\)):
    Returns the output obtained from applying the layer to the given input.
  - [zeros()](/zeros\(\)):
    Returns a function that creates a tensor by initializing all its values to zeros.
  - [constantInitializer(value:)](/constantInitializer\(value:\)):
    Returns a function that creates a tensor by initializing all its values to the provided value.
  - [constantInitializer(value:)](/constantInitializer\(value:\)):
    Returns a function that creates a tensor by initializing it to the provided value. Note that
    broadcasting of the provided value is *not* supported.
  - [glorotUniform(seed:)](/glorotUniform\(seed:\)):
    Returns a function that creates a tensor by performing Glorot (Xavier) uniform initialization
    for the specified shape, randomly sampling scalar values from a uniform distribution between
    `-limit` and `limit`, generated by the default random number generator, where limit is
    `sqrt(6 / (fanIn + fanOut))`, and `fanIn`/`fanOut` represent the number of input and output
    features multiplied by the receptive field, if present.
  - [glorotNormal(seed:)](/glorotNormal\(seed:\)):
    Returns a function that creates a tensor by performing Glorot (Xavier) normal initialization for
    the specified shape, randomly sampling scalar values from a truncated normal distribution centered
    on `0` with standard deviation `sqrt(2 / (fanIn + fanOut))`, where `fanIn`/`fanOut` represent
    the number of input and output features multiplied by the receptive field size, if present.
  - [heUniform(seed:)](/heUniform\(seed:\)):
    Returns a function that creates a tensor by performing He (Kaiming) uniform initialization for
    the specified shape, randomly sampling scalar values from a uniform distribution between `-limit`
    and `limit`, generated by the default random number generator, where limit is
    `sqrt(6 / fanIn)`, and `fanIn` represents the number of input features multiplied by the
    receptive field, if present.
  - [heNormal(seed:)](/heNormal\(seed:\)):
    Returns a function that creates a tensor by performing He (Kaiming) normal initialization for the
    specified shape, randomly sampling scalar values from a truncated normal distribution centered
    on `0` with standard deviation `sqrt(2 / fanIn)`, where `fanIn` represents the number of input
    features multiplied by the receptive field size, if present.
  - [leCunUniform(seed:)](/leCunUniform\(seed:\)):
    Returns a function that creates a tensor by performing LeCun uniform initialization for
    the specified shape, randomly sampling scalar values from a uniform distribution between `-limit`
    and `limit`, generated by the default random number generator, where limit is
    `sqrt(3 / fanIn)`, and `fanIn` represents the number of input features multiplied by the
    receptive field, if present.
  - [leCunNormal(seed:)](/leCunNormal\(seed:\)):
    Returns a function that creates a tensor by performing LeCun normal initialization for the
    specified shape, randomly sampling scalar values from a truncated normal distribution centered
    on `0` with standard deviation `sqrt(1 / fanIn)`, where `fanIn` represents the number of input
    features multiplied by the receptive field size, if present.
  - [truncatedNormalInitializer(mean:standardDeviation:seed:)](/truncatedNormalInitializer\(mean:standardDeviation:seed:\)):
    Returns a function that creates a tensor by initializing all its values randomly from a
    truncated Normal distribution. The generated values follow a Normal distribution with mean
    `mean` and standard deviation `standardDeviation`, except that values whose magnitude is more
    than two standard deviations from the mean are dropped and resampled.
  - [l1Loss(predicted:expected:reduction:)](/l1Loss\(predicted:expected:reduction:\)):
    Computes the L1 loss between `expected` and `predicted`.
    `loss = reduction(abs(expected - predicted))`
  - [l2Loss(predicted:expected:reduction:)](/l2Loss\(predicted:expected:reduction:\)):
    Computes the L2 loss between `expected` and `predicted`.
    `loss = reduction(square(expected - predicted))`
  - [meanAbsoluteError(predicted:expected:)](/meanAbsoluteError\(predicted:expected:\)):
    Computes the mean of absolute difference between labels and predictions.
    `loss = mean(abs(expected - predicted))`
  - [meanSquaredError(predicted:expected:)](/meanSquaredError\(predicted:expected:\)):
    Computes the mean of squares of errors between labels and predictions.
    `loss = mean(square(expected - predicted))`
  - [meanSquaredLogarithmicError(predicted:expected:)](/meanSquaredLogarithmicError\(predicted:expected:\)):
    Computes the mean squared logarithmic error between `predicted` and `expected`
    `loss = square(log(expected) - log(predicted))`
  - [meanAbsolutePercentageError(predicted:expected:)](/meanAbsolutePercentageError\(predicted:expected:\)):
    Computes the mean absolute percentage error between `predicted` and `expected`.
    `loss = 100 * mean(abs((expected - predicted) / abs(expected)))`
  - [hingeLoss(predicted:expected:reduction:)](/hingeLoss\(predicted:expected:reduction:\)):
    Computes the hinge loss between `predicted` and `expected`.
    `loss = reduction(max(0, 1 - predicted * expected))`
    `expected` values are expected to be -1 or 1.
  - [squaredHingeLoss(predicted:expected:reduction:)](/squaredHingeLoss\(predicted:expected:reduction:\)):
    Computes the squared hinge loss between `predicted` and `expected`.
    `loss = reduction(square(max(0, 1 - predicted * expected)))`
    `expected` values are expected to be -1 or 1.
  - [categoricalHingeLoss(predicted:expected:reduction:)](/categoricalHingeLoss\(predicted:expected:reduction:\)):
    Computes the categorical hinge loss between `predicted` and `expected`.
    `loss = maximum(negative - positive + 1, 0)`
    where `negative = max((1 - expected) * predicted)` and
    `positive = sum(predicted * expected)`
  - [logCoshLoss(predicted:expected:reduction:)](/logCoshLoss\(predicted:expected:reduction:\)):
    Computes the logarithm of the hyperbolic cosine of the prediction error.
    `logcosh = log((exp(x) + exp(-x))/2)`,
    where x is the error `predicted - expected`
  - [poissonLoss(predicted:expected:reduction:)](/poissonLoss\(predicted:expected:reduction:\)):
    Computes the Poisson loss between predicted and expected
    The Poisson loss is the mean of the elements of the `Tensor`
    `predicted - expected * log(predicted)`.
  - [kullbackLeiblerDivergence(predicted:expected:reduction:)](/kullbackLeiblerDivergence\(predicted:expected:reduction:\)):
    Computes Kullback-Leibler divergence loss between `expected` and `predicted`.
    `loss = reduction(expected * log(expected / predicted))`
  - [softmaxCrossEntropy(logits:labels:reduction:)](/softmaxCrossEntropy\(logits:labels:reduction:\)):
    Computes the sparse softmax cross entropy (categorical cross entropy) between logits and labels.
    Use this crossentropy loss function when there are two or more label classes.
    We expect labels to be provided as integers. There should be `# classes`
    floating point values per feature for `logits` and a single floating point value per feature for `expected`.
  - [softmaxCrossEntropy(logits:probabilities:reduction:)](/softmaxCrossEntropy\(logits:probabilities:reduction:\)):
    Computes the sparse softmax cross entropy (categorical cross entropy) between logits and labels.
    Use this crossentropy loss function when there are two or more label classes.
    We expect labels to be provided provided in a `one_hot` representation.
    There should be `# classes` floating point values per feature.
  - [sigmoidCrossEntropy(logits:labels:reduction:)](/sigmoidCrossEntropy\(logits:labels:reduction:\)):
    Computes the sigmoid cross entropy (binary cross entropy) between logits and labels.
    Use this cross-entropy loss when there are only two label classes (assumed to
    be 0 and 1). For each example, there should be a single floating-point value
    per prediction.
  - [huberLoss(predicted:expected:delta:reduction:)](/huberLoss\(predicted:expected:delta:reduction:\)):
    Computes the Huber loss between `predicted` and `expected`.
  - [\_sum(\_:)](/_sum\(_:\)):
    Workaround for TF-1030 so that we can use sum as a default argument for reductions.
    `Tensor<Scalar>.sum()` is the preferred way to do this.
  - [\_mean(\_:)](/_mean\(_:\)):
    Workaround for TF-1030 so that we can use mean as a default argument for reductions.
    `Tensor<Scalar>.mean()` is the preferred way to do this.
  - [identity(\_:)](/identity\(_:\)):
    Returns a tensor with the same shape and scalars as the specified tensor.
  - [zip(\_:\_:)](/zip\(_:_:\))
  - [resize(images:size:method:antialias:)](/resize\(images:size:method:antialias:\)):
    Resize images to size using the specified method.
  - [resizeArea(images:size:alignCorners:)](/resizeArea\(images:size:alignCorners:\)):
    Resize images to size using area interpolation.
  - [eye(rowCount:columnCount:batchShape:)](/eye\(rowCount:columnCount:batchShape:\)):
    Returns an identity matrix or a batch of matrices.
  - [trace(\_:)](/trace\(_:\)):
    Computes the trace of an optionally batched matrix.
    The trace is the the sum along the main diagonal of each inner-most matrix.
  - [cholesky(\_:)](/cholesky\(_:\)):
    Returns the Cholesky decomposition of one or more square matrices.
  - [triangularSolve(matrix:rhs:lower:adjoint:)](/triangularSolve\(matrix:rhs:lower:adjoint:\)):
    Returns the solution `x` to the system of linear equations represented by `Ax = b`.
  - [abs(\_:)](/abs\(_:\)):
    Returns the absolute value of the specified tensor element-wise.
  - [log(\_:)](/log\(_:\)):
    Returns the natural logarithm of the specified tensor element-wise.
  - [log2(\_:)](/log2\(_:\)):
    Returns the base-two logarithm of the specified tensor element-wise.
  - [log10(\_:)](/log10\(_:\)):
    Returns the base-ten logarithm of the specified tensor element-wise.
  - [log1p(\_:)](/log1p\(_:\)):
    Returns the logarithm of `1 + x` element-wise.
  - [log1mexp(\_:)](/log1mexp\(_:\)):
    Returns `log(1 - exp(x))` using a numerically stable approach.
  - [sin(\_:)](/sin\(_:\)):
    Returns the sine of the specified tensor element-wise.
  - [cos(\_:)](/cos\(_:\)):
    Returns the cosine of the specified tensor element-wise.
  - [tan(\_:)](/tan\(_:\)):
    Returns the tangent of the specified tensor element-wise.
  - [sinh(\_:)](/sinh\(_:\)):
    Returns the hyperbolic sine of the specified tensor element-wise.
  - [cosh(\_:)](/cosh\(_:\)):
    Returns the hyperbolic cosine of the specified tensor element-wise.
  - [tanh(\_:)](/tanh\(_:\)):
    Returns the hyperbolic tangent of the specified tensor element-wise.
  - [acos(\_:)](/acos\(_:\)):
    Returns the inverse cosine of the specified tensor element-wise.
  - [asin(\_:)](/asin\(_:\)):
    Returns the inverse sine of the specified tensor element-wise.
  - [atan(\_:)](/atan\(_:\)):
    Returns the inverse tangent of the specified tensor element-wise.
  - [acosh(\_:)](/acosh\(_:\)):
    Returns the inverse hyperbolic cosine of the specified tensor element-wise.
  - [asinh(\_:)](/asinh\(_:\)):
    Returns the inverse hyperbolic sine of the specified tensor element-wise.
  - [atanh(\_:)](/atanh\(_:\)):
    Returns the inverse hyperbolic tangent of the specified tensor element-wise.
  - [sqrt(\_:)](/sqrt\(_:\)):
    Returns the square root of the specified tensor element-wise.
  - [rsqrt(\_:)](/rsqrt\(_:\)):
    Returns the inverse square root of the specified tensor element-wise.
  - [exp(\_:)](/exp\(_:\)):
    Returns the exponential of the specified tensor element-wise.
  - [exp2(\_:)](/exp2\(_:\)):
    Returns two raised to the power of the specified tensor element-wise.
  - [exp10(\_:)](/exp10\(_:\)):
    Returns ten raised to the power of the specified tensor element-wise.
  - [expm1(\_:)](/expm1\(_:\)):
    Returns the exponential of `x - 1` element-wise.
  - [round(\_:)](/round\(_:\)):
    Returns the values of the specified tensor rounded to the nearest integer, element-wise.
  - [ceil(\_:)](/ceil\(_:\)):
    Returns the ceiling of the specified tensor element-wise.
  - [floor(\_:)](/floor\(_:\)):
    Returns the floor of the specified tensor element-wise.
  - [sign(\_:)](/sign\(_:\)):
    Returns an indication of the sign of the specified tensor element-wise.
    Specifically, computes `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
  - [sigmoid(\_:)](/sigmoid\(_:\)):
    Returns the sigmoid of the specified tensor element-wise.
    Specifically, computes `1 / (1 + exp(-x))`.
  - [logSigmoid(\_:)](/logSigmoid\(_:\)):
    Returns the log-sigmoid of the specified tensor element-wise. Specifically,
    `log(1 / (1 + exp(-x)))`. For numerical stability, we use `-softplus(-x)`.
  - [softplus(\_:)](/softplus\(_:\)):
    Returns the softplus of the specified tensor element-wise.
    Specifically, computes `log(exp(features) + 1)`.
  - [softsign(\_:)](/softsign\(_:\)):
    Returns the softsign of the specified tensor element-wise.
    Specifically, computes `features/ (abs(features) + 1)`.
  - [softmax(\_:)](/softmax\(_:\)):
    Returns the softmax of the specified tensor along the last axis.
    Specifically, computes `exp(x) / exp(x).sum(alongAxes: -1)`.
  - [softmax(\_:alongAxis:)](/softmax\(_:alongAxis:\)):
    Returns the softmax of the specified tensor along the specified axis.
    Specifically, computes `exp(x) / exp(x).sum(alongAxes: axis)`.
  - [logSoftmax(\_:)](/logSoftmax\(_:\)):
    Returns the log-softmax of the specified tensor element-wise.
  - [elu(\_:)](/elu\(_:\)):
    Returns a tensor by applying an exponential linear unit.
    Specifically, computes `exp(x) - 1` if \< 0, `x` otherwise.
    See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
    ](http://arxiv.org/abs/1511.07289)
  - [gelu(\_:)](/gelu\(_:\)):
    Returns the Gaussian Error Linear Unit (GELU) activations of the specified tensor element-wise.
  - [relu(\_:)](/relu\(_:\)):
    Returns a tensor by applying the ReLU activation function to the specified tensor element-wise.
    Specifically, computes `max(0, x)`.
  - [relu6(\_:)](/relu6\(_:\)):
    Returns a tensor by applying the ReLU6 activation function, namely `min(max(0, x), 6)`.
  - [leakyRelu(\_:alpha:)](/leakyRelu\(_:alpha:\)):
    Returns a tensor by applying the leaky ReLU activation function
    to the specified tensor element-wise.
    Specifically, computes `max(x, x * alpha)`.
  - [selu(\_:)](/selu\(_:\)):
    Returns a tensor by applying the SeLU activation function, namely
    `scale * alpha * (exp(x) - 1)` if `x < 0`, and `scale * x` otherwise.
  - [swish(\_:)](/swish\(_:\)):
    Returns a tensor by applying the swish activation function, namely
    `x * sigmoid(x)`.
  - [pow(\_:\_:)](/pow\(_:_:\)):
    Returns the power of the first tensor to the second tensor.
  - [pow(\_:\_:)](/pow\(_:_:\)):
    Returns the power of the scalar to the tensor, broadcasting the scalar.
  - [pow(\_:\_:)](/pow\(_:_:\)):
    Returns the power of the tensor to the scalar, broadcasting the scalar.
  - [pow(\_:\_:)](/pow\(_:_:\)):
    Returns the power of the tensor to the scalar, broadcasting the scalar.
  - [root(\_:\_:)](/root\(_:_:\)):
    Returns the element-wise `n`th root of the tensor.
  - [squaredDifference(\_:\_:)](/squaredDifference\(_:_:\)):
    Returns the squared difference between `x` and `y`.
  - [max(\_:\_:)](/max\(_:_:\)):
    Returns the element-wise maximum of two tensors.
  - [max(\_:\_:)](/max\(_:_:\)):
    Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
  - [max(\_:\_:)](/max\(_:_:\)):
    Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
  - [min(\_:\_:)](/min\(_:_:\)):
    Returns the element-wise minimum of two tensors.
  - [min(\_:\_:)](/min\(_:_:\)):
    Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
  - [min(\_:\_:)](/min\(_:_:\)):
    Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
  - [cosineSimilarity(\_:\_:)](/cosineSimilarity\(_:_:\)):
    Returns the cosine similarity between `x` and `y`.
  - [cosineDistance(\_:\_:)](/cosineDistance\(_:_:\)):
    Returns the cosine distance between `x` and `y`. Cosine distance is defined as
    `1 - cosineSimilarity(x, y)`.
  - [matmul(\_:transposed:\_:transposed:)](/matmul\(_:transposed:_:transposed:\)):
    Performs matrix multiplication with another tensor and produces the result.
  - [conv1D(\_:filter:stride:padding:dilation:)](/conv1D\(_:filter:stride:padding:dilation:\)):
    Returns a 1-D convolution with the specified input, filter, stride, and padding.
  - [conv2D(\_:filter:strides:padding:dilations:)](/conv2D\(_:filter:strides:padding:dilations:\)):
    Returns a 2-D convolution with the specified input, filter, strides, and padding.
  - [transposedConv2D(\_:shape:filter:strides:padding:dilations:)](/transposedConv2D\(_:shape:filter:strides:padding:dilations:\)):
    Returns a 2-D transposed convolution with the specified input, filter, strides, and padding.
  - [conv3D(\_:filter:strides:padding:dilations:)](/conv3D\(_:filter:strides:padding:dilations:\)):
    Returns a 3-D convolution with the specified input, filter, strides, padding and dilations.
  - [depthwiseConv2D(\_:filter:strides:padding:)](/depthwiseConv2D\(_:filter:strides:padding:\)):
    Returns a 2-D depthwise convolution with the specified input, filter, strides, and padding.
  - [maxPool2D(\_:filterSize:strides:padding:)](/maxPool2D\(_:filterSize:strides:padding:\)):
    Returns a 2-D max pooling, with the specified filter sizes, strides, and
    padding.
  - [maxPool3D(\_:filterSize:strides:padding:)](/maxPool3D\(_:filterSize:strides:padding:\)):
    Returns a 3-D max pooling, with the specified filter sizes, strides, and
    padding.
  - [avgPool2D(\_:filterSize:strides:padding:)](/avgPool2D\(_:filterSize:strides:padding:\)):
    Returns a 2-D average pooling, with the specified filter sizes, strides,
    and padding.
  - [avgPool3D(\_:filterSize:strides:padding:)](/avgPool3D\(_:filterSize:strides:padding:\)):
    Returns a 3-D average pooling, with the specified filter sizes, strides,
    and padding.
  - [allTests()](/allTests\(\))
  - [allTests()](/allTests\(\))
  - [allTests()](/allTests\(\))

# Global Variables

  - [Python](/Python):
    The global Python interface.
  - [base](/base):
    The underlying base value.
  - [zero](/zero)
  - [differentiableVectorView](/differentiableVectorView)
  - [base](/base):
    The viewed array.
  - [weight](/weight):
    The weight matrix.
  - [bias](/bias):
    The bias vector.
  - [activation](/activation):
    The element-wise activation function.
