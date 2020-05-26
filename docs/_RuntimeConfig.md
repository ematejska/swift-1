# \_RuntimeConfig

The configuration for the compiler runtime.

``` swift
public enum _RuntimeConfig
```

## Properties

### `tensorFlowRuntimeInitialized`

When false, tensorflow runtime will be initialized before running any tensor program in this
process.

``` swift
var tensorFlowRuntimeInitialized
```

### `gpuMemoryAllowGrowth`

When true, let TensorFlow GPU memory allocation start small and grow as needed. Otherwise,
The entire GPU memory region is pre-allocated.

``` swift
var gpuMemoryAllowGrowth
```

### `cpuDeviceCount`

The number of CPU devices.

``` swift
var cpuDeviceCount: UInt32
```

### `session`

``` swift
var session: RuntimeSession
```

### `useLazyTensor`

When true, use lazy evaluation.

``` swift
var useLazyTensor: Bool
```

### `printsDebugLog`

When true, prints various debug messages on the runtime state.

``` swift
var printsDebugLog
```

If the value is true when running tensor computation for the first time in the process, INFO
log from TensorFlow will also get printed.

### `tensorflowVerboseLogLevel`

Specifies the verbose log level in TensorFlow; a higher level prints out more log. Only
meaningful when `printsDebugLog` is true, and must be within \[0, 4\] in that case.

``` swift
var tensorflowVerboseLogLevel: Int32
```
