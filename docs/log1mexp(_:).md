# log1mexp(\_:)

Returns `log(1 - exp(x))` using a numerically stable approach.

``` swift
@inlinable @differentiable public func log1mexp<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T>
```

> Note: The approach is shown in Equation 7 of: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
