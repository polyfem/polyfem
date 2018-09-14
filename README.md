PolyFEM
=======

[![Build Status](https://travis-ci.com/geometryprocessing/polyfem.svg?token=euzAY1sxC114E8ufzcZx&branch=master)](https://travis-ci.com/geometryprocessing/polyfem)


Compilation
-----------

All the dependencies required to build the code are included. It should work on Windows, macOS and Linux, and it should build out of the box with CMake:

    ```
    mkdir build
    cd build
    cmake ..
    make -j4
    ```

On Linux you need `zenity` for the file dialog window to work. On macOS and Windows it should use the native windows directly.
Note that the formula for higher order bases and quadrature points are pre-computed using Python. CMake can call those python as part of the compilation process, but by default this automatic generation is disabled, since it requires a working python installation and additional packages (`sympy` and `quadpy`).


Usage
-----

The main executable, `./Polyfem_bin`, can be called with a GUI or through a command-line interface. The GUI is pretty simple and should be self-explanatory. To call the command-line interface, set the mesh path in the file `example.json`, and run as follows:

    ./Polyfem_bin --cmd --json ../example.json
  
  
 For the complete list of options use

    ./Polyfem_bin -h


