# \_RuntimeConfig.RuntimeSession

Specifies whether the TensorFlow computation runs in a local (in-process) session, or a
remote session with the specified server definition.

``` swift
public enum RuntimeSession
```

## Enumeration Cases

### `local`

``` swift
case local
```

### `remote`

``` swift
case remote(serverDef: String)
```
