PolyFEM
=======

[![Build Status](https://travis-ci.com/geometryprocessing/polyfem.svg?token=euzAY1sxC114E8ufzcZx&branch=master)](https://travis-ci.com/geometryprocessing/polyfem)


Compilation
-----------

All the dependencies required to build the code are included. It should work on Windows, macOS and Linux, and it should build out of the box with CMake:

    mkdir build
    cd build
    cmake ..
    make -j4

On Linux you need `zenity` for the file dialog window to work. On macOS and Windows it should use the native windows directly.
Note that the formula for higher order bases and quadrature points are pre-computed using Python. CMake can call those python as part of the compilation process, but by default this automatic generation is disabled, since it requires a working python installation and additional packages (`sympy` and `quadpy`).


Usage
-----

The main executable, `./Polyfem_bin`, can be called with a GUI or through a command-line interface. The GUI is pretty simple and should be self-explanatory. To call the command-line interface, set the mesh path in the file `example.json`, and run as follows:

    ./Polyfem_bin --cmd --json ../example.json
  
  
 For the complete list of options use

    ./Polyfem_bin -h





Json files
----------
Complete example
```json
{
    "mesh": " ",                    "Mesh path"
    "bc_tag": " ",                  "Path to the boundary tag file, each face/edge is associated with an unique number (you can use bc_setter for setting them in 3d)"
    "normalize_mesh": true,         "Normalize mesh such that it fits in the [0,1] bounding box"

    "n_refs": 0,                    "Number of uniform refinement"
    "refinenemt_location": 0.5,     "Refiniement location of polyhedra"

    "scalar_formulation": "Laplacian",
    "tensor_formulation": "LinearElasticity",
    "mixed_formulation": "Stokes",

    "count_flipped_els": false,     "Count (or not) flipped elements"

    "iso_parametric": false,        "Force isoparametric elements"
    "discr_order": 1,               "Dicretization order, supports P1, P2, P3, P4, Q1, Q2"

    "pressure_discr_order": 1,      "Pressure dicrezation order, for mixed formulation"

    "output": "",                   "Output json path"

    "problem": "Franke",            "Problem to solve"
    "problem_params": {},           "Problem specific parameters"

    "n_boundary_samples": 6,        "number of boundary samples (Dirichelt) or quadrature points (Neumann)"
    "quadrature_order": 4,          "quadrature order"

    "export": {                     "Export options"
        "full_mat": "",             "Stiffnes matrix without boundary conditions"
        "iso_mesh": "",             "Isolines mesh"
        "nodes": "",                "FE nodes"
        "solution": "",             "Solution vector"
        "spectrum": false,          "Spectrum of the stiffness matrix"
        "stiffness_mat": "",        "Stiffmess matrix after setting boundary conditions"
        "vis_mesh": "",             "Path for the vtu mesh"
        "wire_mesh": ""             "Wireframe of the mesh"
    },

    "use_spline": false,            "Use spline for quad/hex elements"
    "fit_nodes": false,             "Fit nodes for spline basis"

    "integral_constraints": 2,      "Number of constraints for polygonal basis 0, 1, or 2"
    "n_harmonic_samples": 10,       "Number of face/line samples for harmonic bases for polyhedra"

    "B": 3,                         "User provided parameter for pref tolerance"
    "use_p_ref": false,             "Enable prefinement for badly shaped simplices"
    "discr_order_max": 4,           "Maximum allowed dicrezation oder, used in p pref"
    "h1_formula": false,            "Use pref formula for h1 bound"

    "force_linear_geometry": false, "Force Linear geometric maps"

    "solver_type": "Pardiso",       "Library for linear solver"
    "precond_type": "Eigen::DiagonalPreconditioner",
    "solver_params": {},            "solver specific parameters"

    "nl_solver": "newton",          "Non linear solver"
    "line_search": "armijo",        "Line search for newton solver"
    "nl_solver_rhs_steps": 1,       "Number of incremental load steps"
    "save_solve_sequence": false,   "Save all incremental load steps"

    "params": {                     "Material parameter"
        "k": 1.0,                   "Constant in helmolz"

        "elasticity_tensor": {},    "Elasticity tensor, used in hooke and saint ventant"

        "E": 1.5,                   "Young modulus"
        "nu": 0.3,                  "Poisson's ratio"

        "lambda": 0.329670329,      "Lame parameter, E, nu have priority"
        "mu": 0.384615384,
    },


    "tend": 1,                      "End time for time dependent simulations"
    "time_steps": 10,               "Number of time steps for time dependent simulations"

    "vismesh_rel_area": 1e-05       "Relative resolution of the output mesh"
}
```

### Optionals

* **scalar_formulation**: Helmholtz, Laplacian
* **tensor_formulation**: HookeLinearElasticity, LinearElasticity, NeoHookean, Ogden, SaintVenant
* **mixed_formulation**: IncompressibleLinearElasticity, Stokes

* **problem**: CompressionElasticExact, Cubic, DrivenCavity, Elastic, ElasticExact, ElasticZeroBC, Flow, Franke, GenericTensor, Gravity, Kernel, Linear, LinearElasticExact, MinSurf, PointBasedTensor, Quadratic, QuadraticElasticExact, Sine, TestProblem, TimeDependentFlow, TimeDependentScalar, TorsionElastic, Zero_BC


* **solver_type**: Eigen::BiCGSTAB, Eigen::ConjugateGradient, Eigen::GMRES, Eigen::MINRES, Eigen::SimplicialLDLT, Eigen::SparseLU, Hypre,Pardiso
* **nl_solver**: lbfgs, newton
* **line_search**: armijo, armijo_alt, bisection, more_thuente


Problems
--------
Each problem has a specific set of optional `problem_params` described here.

##### CompressionElasticExact

##### Cubic

##### DrivenCavity

##### Elastic

##### ElasticExact

##### ElasticZeroBC

##### Flow

##### Franke

##### GenericTensor

##### Gravity

##### Kernel

##### Linear

##### LinearElasticExact

##### MinSurf

##### PointBasedTensor

##### Quadratic

##### QuadraticElasticExact

##### Sine

##### TestProblem

##### TimeDependentFlow

##### TimeDependentScalar

##### TorsionElastic

##### Zero_BC


