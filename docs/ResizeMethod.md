# ResizeMethod

A resize algorithm.

``` swift
public enum ResizeMethod
```

## Enumeration Cases

### `nearest`

Nearest neighbor interpolation.

``` swift
case nearest
```

### `bilinear`

Bilinear interpolation.

``` swift
case bilinear
```

### `bicubic`

Bicubic interpolation.

``` swift
case bicubic
```

### `lanczos3`

Lanczos kernel with radius `3`.

``` swift
case lanczos3
```

### `lanczos5`

Lanczos kernel with radius `5`.

``` swift
case lanczos5
```

### `gaussian`

Gaussian kernel with radius `3`, sigma `1.5 / 3.0`.

``` swift
case gaussian
```

### `mitchellcubic`

Mitchell-Netravali Cubic non-interpolating filter.

``` swift
case mitchellcubic
```
