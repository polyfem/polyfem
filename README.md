PolyFEM
=======


### Compilation

To get the code to run with Geogram 1.5.4, you need to do a minor modification to the file `3rdparty/geogram/src/lib/third_party/CMakeLists.txt`. Replace the line 24

```
if(TARGET glfw3)
```

by the following:

```
if(TARGET glfw)
```

Then build the C++ project as usual.
