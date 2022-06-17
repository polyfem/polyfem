# PolyFEM

## Building PolyFEM

All the C++ dependencies required to build the code are included. It should work on Windows, macOS, and Linux, and it should build out-of-the-box with CMake:


```bash
mkdir build
cd build
cmake ..
make -j4
```

## Using PolyFEM as dependency

**Polyfem** uses modern `cmake`, so it it should be enough to add this line
```cmake
add_subdirectory(<path-to-polyfem> polyfem)
```
in your cmake project, and then simply add
```cmake
target_link_library(<your_target> polyfem::polyfem)
```
in your cmake script.
Polyfem will download the dependencies that it needs with the version that it needs.


## Interface

This is the main interface of `polyfem::State`.

