# TensorRange

``` swift
public enum TensorRange
```

## Inheritance

`Equatable`, [`TensorRangeExpression`](/TensorRangeExpression)

## Enumeration Cases

### `ellipsis`

``` swift
case ellipsis
```

### `newAxis`

``` swift
case newAxis
```

### `squeezeAxis`

``` swift
case squeezeAxis
```

### `index`

``` swift
case index(: Int)
```

### `range`

``` swift
case range(: Range<Int>, stride: Int)
```

### `closedRange`

``` swift
case closedRange(: ClosedRange<Int>, stride: Int)
```

### `partialRangeFrom`

``` swift
case partialRangeFrom(: PartialRangeFrom<Int>, stride: Int)
```

### `partialRangeUpTo`

``` swift
case partialRangeUpTo(: PartialRangeUpTo<Int>, stride: Int)
```

### `partialRangeThrough`

``` swift
case partialRangeThrough(: PartialRangeThrough<Int>, stride: Int)
```

## Properties

### `tensorRange`

``` swift
var tensorRange: TensorRange
```

## Methods

### `==(lhs:rhs:)`

``` swift
public static func ==(lhs: TensorRange, rhs: TensorRange) -> Bool
```
