<h1 align="center">
<a href="https://polyfem.github.io/"><img alt="polyfem" src="https://polyfem.github.io/img/polyfem.png" width="60%"></a>
</h1><br>

[![Build](https://github.com/polyfem/polyfem/actions/workflows/continuous.yml/badge.svg?label=test)](https://github.com/polyfem/polyfem/actions/workflows/continuous.yml)
[![codecov](https://codecov.io/github/polyfem/polyfem/graph/badge.svg?token=ZU9KLLTTDT)](https://codecov.io/github/polyfem/polyfem)
[![Nightly](https://github.com/polyfem/polyfem/actions/workflows/nightly.yml/badge.svg)](https://github.com/polyfem/polyfem/actions/workflows/nightly.yml)
[![Docs](https://github.com/polyfem/polyfem/actions/workflows/docs.yml/badge.svg)](https://polyfem.github.io/polyfem)

PolyFEM is a polyvalent C++ FEM library.

Compilation
-----------

All the C++ dependencies required to build the code are included. It should work on Windows, macOS, and Linux, and it should build out-of-the-box with CMake:

    mkdir build
    cd build
    cmake ..
    make -j4

On Linux, `zenity` is required for the file dialog window to work. On macOS and Windows, the native windows are used directly.


### Optional
The formula for higher-order bases is optionally computed at CMake time using an external python script. Consequently, PolyFEM might require a working installation of Python and some additional packages to build correctly:

- `numpy` and `sympy` (optional)
- `quadpy` (optional)

Usage
-----

The main executable, `./PolyFEM_bin`, can be called with a GUI or through a command-line interface. Simply run:

    ./PolyFEM_bin

A more detailed documentation can be found on the [website](https://polyfem.github.io/).

Documentation
-------------

The full documentation can be found at [https://polyfem.github.io/](https://polyfem.github.io/)



License
-------

The code of PolyFEM itself is licensed under [MIT License](LICENSE). However, please be mindful of third-party libraries which are used by PolyFEM and may be available under a different license.

Citation
--------

If you use PolyFEM in your project, please consider citing our work:

```bibtex
@misc{polyfem,
  author = {Teseo Schneider and Jérémie Dumas and Xifeng Gao and Denis Zorin and Daniele Panozzo},
  title = {{Polyfem}},
  howpublished = "\url{https://polyfem.github.io/}",
  year = {2019},
}
```

```bibtex
@article{Schneider:2019:PFM,
  author = {Schneider, Teseo and Dumas, J{\'e}r{\'e}mie and Gao, Xifeng and Botsch, Mario and Panozzo, Daniele and Zorin, Denis},
  title = {Poly-Spline Finite-Element Method},
  journal = {ACM Trans. Graph.},
  volume = {38},
  number = {3},
  month = mar,
  year = {2019},
  url = {http://doi.acm.org/10.1145/3313797},
  publisher = {ACM}
}
```

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

Acknowledgments & Funding
--------
The software is being developed in the [Geometric Computing Lab](https://cims.nyu.edu/gcl/index.html) at NYU Courant Institute of Mathematical Sciences and the University of Victoria, Canada.


This work was partially supported by:

* the NSF CAREER award 1652515
* the NSF grant IIS-1320635
* the NSF grant DMS-1436591
* the NSF grant 1835712
* the SNSF grant P2TIP2_175859
* the NSERC grant RGPIN-2021-03707
* the NSERC grant DGECR-2021-00461
* Adobe Research
* nTopology
