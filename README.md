PolyFEM
=======

[![Build Status](https://travis-ci.com/polyfem/polyfem.svg?branch=master)](https://travis-ci.com/polyfem/polyfem)
[![Build status](https://ci.appveyor.com/api/projects/status/tseks5d0kydqhjot/branch/master?svg=true)](https://ci.appveyor.com/project/teseoch/polyfem/branch/master)


Compilation
-----------

All the C++ dependencies required to build the code are included. It should work on Windows, macOS and Linux, and it should build out of the box with CMake:

    mkdir build
    cd build
    cmake ..
    make -j4

On Linux `zenity` is required for the file dialog window to work. On macOS and Windows the native windows are used directly.
The formula for higher order bases are computed at CMake time using an external python script. Consequently, PolyFEM requires a working installation of Python and some additional packages in order to build correctly:

- `numpy` and `sympy`
- `quadpy` (optional)

Usage
-----

The main executable, `./PolyFEM_bin`, can be called with a GUI or through a command-line interface. The GUI is pretty simple and should be self-explanatory. To call the command-line interface, set the setup an `example.json` file, and run as follows:

    ./PolyFEM_bin --cmd --json ../example.json
    
 For the complete list of options use

    ./PolyFEM_bin -h

Documentation
-------------

A more detailed documentation can be found on the [website](https://polyfem.github.io/).

License
-------

The code of PolyFEM itself is licensed under [MIT License](LICENSE). However, please be mindful of third-party libraries which are used by PolyFEM, and may be available under a different license.

Citation
--------

If you use PolyFEM in your project, please consider citing our work:

```bibtex
@article{Schneider:2018:DSA,
    author = {Teseo Schneider and Yixin Hu and Jérémie Dumas and Xifeng Gao and Daniele Panozzo and Denis Zorin},
    journal = {ACM Transactions on Graphics},
    link = {},
    month = {10},
    number = {6},
    publisher = {Association for Computing Machinery (ACM)},
    title = {Decoupling Simulation Accuracy from Mesh Quality},
    volume = {37},
    year = {2018}
}
```
