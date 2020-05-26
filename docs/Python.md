# Python

The global Python interface.

``` swift
let Python
```

You can import Python modules and access Python builtin types and functions
via the `Python` global variable.

``` 
import Python
// Import modules.
let os = Python.import("os")
let np = Python.import("numpy")

// Use builtin types and functions.
let list: PythonObject = [1, 2, 3]
print(Python.len(list)) // Prints 3.
print(Python.type(list) == Python.list) // Prints true.
```
