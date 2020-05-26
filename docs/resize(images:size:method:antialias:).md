# resize(images:size:method:antialias:)

Resize images to size using the specified method.

``` swift
@differentiable(wrt: images) public func resize(images: Tensor<Float>, size: (newHeight: Int, newWidth: Int), method: ResizeMethod = .bilinear, antialias: Bool = false) -> Tensor<Float>
```

> Precondition: The images must have rank \`3\` or \`4\`.

> Precondition: The size must be positive.

## Parameters

  - images: - images: 4-D `Tensor` of shape `[batch, height, width, channels]` or 3-D `Tensor` of shape `[height, width, channels]`.
  - size: - size: The new size of the images.
  - method: - method: The resize method. The default value is `.bilinear`.
  - antialias: - antialias: Iff `true`, use an anti-aliasing filter when downsampling an image.
