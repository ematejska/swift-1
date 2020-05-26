# PythonLibrary

``` swift
public struct PythonLibrary
```

## Initializers

### `init()`

``` swift
private init()
```

## Properties

### `shared`

``` swift
let shared
```

### `pythonLegacySymbolName`

``` swift
let pythonLegacySymbolName
```

### `librarySymbolsLoaded`

``` swift
var librarySymbolsLoaded
```

### `pythonLibraryHandle`

``` swift
let pythonLibraryHandle: UnsafeMutableRawPointer
```

### `isLegacyPython`

``` swift
let isLegacyPython: Bool
```

### `supportedMajorVersions`

``` swift
let supportedMajorVersions: [Int]
```

### `supportedMinorVersions`

``` swift
let supportedMinorVersions: [Int]
```

### `libraryPathVersionCharacter`

``` swift
let libraryPathVersionCharacter: Character
```

### `libraryNames`

<dl>
<dt><code>canImport(Darwin)</code></dt>
<dd>

``` swift
var libraryNames
```

</dd>
</dl>

### `libraryPathExtensions`

<dl>
<dt><code>canImport(Darwin)</code></dt>
<dd>

``` swift
var libraryPathExtensions
```

</dd>
</dl>

### `librarySearchPaths`

<dl>
<dt><code>canImport(Darwin)</code></dt>
<dd>

``` swift
var librarySearchPaths
```

</dd>
</dl>

### `libraryVersionSeparator`

<dl>
<dt><code>canImport(Darwin)</code></dt>
<dd>

``` swift
var libraryVersionSeparator
```

</dd>
</dl>

### `libraryNames`

<dl>
<dt><code>os(Linux)</code></dt>
<dd>

``` swift
var libraryNames
```

</dd>
</dl>

### `libraryPathExtensions`

<dl>
<dt><code>os(Linux)</code></dt>
<dd>

``` swift
var libraryPathExtensions
```

</dd>
</dl>

### `librarySearchPaths`

<dl>
<dt><code>os(Linux)</code></dt>
<dd>

``` swift
var librarySearchPaths
```

</dd>
</dl>

### `libraryVersionSeparator`

<dl>
<dt><code>os(Linux)</code></dt>
<dd>

``` swift
var libraryVersionSeparator
```

</dd>
</dl>

### `libraryNames`

<dl>
<dt><code>os(Windows)</code></dt>
<dd>

``` swift
var libraryNames
```

</dd>
</dl>

### `libraryPathExtensions`

<dl>
<dt><code>os(Windows)</code></dt>
<dd>

``` swift
var libraryPathExtensions
```

</dd>
</dl>

### `librarySearchPaths`

<dl>
<dt><code>os(Windows)</code></dt>
<dd>

``` swift
var librarySearchPaths
```

</dd>
</dl>

### `libraryVersionSeparator`

<dl>
<dt><code>os(Windows)</code></dt>
<dd>

``` swift
var libraryVersionSeparator
```

</dd>
</dl>

### `libraryPaths`

``` swift
let libraryPaths: [String]
```

## Methods

### `loadSymbol(_:_:)`

``` swift
static func loadSymbol(_ libraryHandle: UnsafeMutableRawPointer, _ name: String) -> UnsafeMutableRawPointer?
```

### `loadSymbol(name:legacyName:type:)`

``` swift
static func loadSymbol<T>(name: String, legacyName: String? = nil, type: T.Type = T.self) -> T
```

### `useVersion(_:_:)`

``` swift
static func useVersion(_ major: Int, _ minor: Int? = nil)
```

### `loadPythonLibrary()`

``` swift
static func loadPythonLibrary() -> UnsafeMutableRawPointer?
```

### `loadPythonLibrary(at:version:)`

``` swift
static func loadPythonLibrary(at path: String, version: PythonVersion) -> UnsafeMutableRawPointer?
```

### `loadPythonLibrary(at:)`

``` swift
static func loadPythonLibrary(at path: String) -> UnsafeMutableRawPointer?
```

### `log(_:)`

``` swift
static func log(_ message: String)
```
