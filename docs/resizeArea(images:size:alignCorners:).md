# resizeArea(images:size:alignCorners:)

Resize images to size using area interpolation.

``` swift
@inlinable public func resizeArea<Scalar: TensorFlowNumeric>(images: Tensor<Scalar>, size: (newHeight: Int, newWidth: Int), alignCorners: Bool = false) -> Tensor<Float>
```

> Precondition: The images must have rank \`3\` or \`4\`.

> Precondition: The size must be positive.

## Parameters

  - images: - images: 4-D `Tensor` of shape `[batch, height, width, channels]` or 3-D `Tensor` of shape `[height, width, channels]`.
  - size: - size: The new size of the images.
