# eye(rowCount:columnCount:batchShape:)

Returns an identity matrix or a batch of matrices.

``` swift
public func eye<Scalar: Numeric>(rowCount: Int, columnCount: Int? = nil, batchShape: [Int] = []) -> Tensor<Scalar>
```

## Parameters

  - rowCount: - rowCount: The number of rows in each batch matrix.
  - columnCount: - columnCount: The number of columns in each batch matrix.
  - batchShape: - batchShape: The leading batch dimensions of the returned tensor.
