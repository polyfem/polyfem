# PolyFEM

![Build](https://github.com/polyfem/polyfem/workflows/Build/badge.svg)
![Commit](https://img.shields.io/github/last-commit/polyfem/polyfem)
[![License](https://img.shields.io/github/license/polyfem/polyfem.svg?color=blue)](https://github.com/polyfem/polyfem/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/polyfem/polyfem?style=social)](https://github.com/polyfem/polyfem)

## Code Structure

The high-level functions of PolyFEM are in `polyfem::State`, a class containing the application state and depending on all the PolyFEM submodules.

### Finite Element Assembler

`polyfem::assembler` assembles the system matrix and right hand side (rhs). The assembler relies on finite element basis (`polyfem::basis`), quadrature rules (`polyfem::quadrature`), and for transient problems on `polyfem::time_integrator`.

### Finite Element Bases

`polyfem::basis` contains the implementation of finite element basis functions.

The module supports the following 2 dimensional elements:
* Lagrangian triangular elements of degrees 1 to 8 (`LagrangeBasis2d.hpp`)
* Lagrangian quadrilateral elements of degree 1 to 8 (`LagrangeBasis2d.hpp`)
* Spline (IGA) elements of degree 1 to 4 (`SplineBasis2d.hpp`)
* Polygonal elements (`PolygonalBasis2d.hpp`)

The module supports the following 3 dimensional elements:
* Lagrangian tetrahedral elements of degrees 1 to 8 (`LagrangeBasis3d.hpp`)
* Lagrangian hexahedral elements of degree 1 to 8 (`LagrangeBasis3d.hpp`)
* Spline (IGA) elements of degree 1 to 4 (`SplineBasis3d.hpp`)
* Polygonal elements (`PolygonalBasis3d.hpp`)

### Mesh data structures

`polyfem::mesh` contains the meshes used to define the finite element spaces, the obstacles used for contact, and their input/output. A mesh is built in two steps:

1. `MeshLoader` decodes a file (MSH, HYBRID or OBJ, through `MeshReader`) or an HDF5 group into a `MeshData`: vertices, element connectivity padded with `-1`, and optionally body/boundary/node/geometry ids and curved (higher-order) nodes.
2. `Mesh::create(MeshData, non_conforming)` builds the runtime mesh: a conforming or non-conforming 2D or 3D mesh.

`GeometryLoader` implements the `geometry` section of the JSON input on top of this: for every entry it loads the mesh, normalizes and transforms it, refines it (`n_refs`), applies the volume/surface/point selections, and appends all the simulated meshes into one; obstacles (surfaces in MSH, OBJ or the triangle formats read by libigl) are collected into an `Obstacle`.

| Class | Represents | Storage |
|---|---|---|
| `Mesh` | Abstract interface: entity counts, geometry, navigation, labels, curved geometry | Labels and curved nodes shared by all meshes |
| `Mesh2D`, `Mesh3D` | Dimension-specific interface; 2D elements are faces, 3D elements are cells | - |
| `CMesh2D` | Conforming mesh of arbitrary polygons | `PolygonMesh`: vertices, facets as ranges of corners, facet adjacency and edges |
| `CMesh3D` | Conforming mesh of tetrahedra, hexahedra, prisms, pyramids and polyhedra | `Mesh3DStorage`: per-entity incidence lists, built from standard cells (`CellConnectivity`) by `MeshProcessing3D` |
| `NCMesh2D`, `NCMesh3D` | Triangle/tetrahedral meshes with hanging nodes, for adaptive h-refinement | Refinement tree of elements, edges, faces and vertices |
| `MeshNodes` | Finite element nodes (vertices, edge/face/cell nodes), created lazily | Packed primitive slots, one block per entity type |
| `LocalBoundary` | Boundary primitives of one element, used to assemble boundary conditions | - |
| `Obstacle` | Non-simulated geometry for contact: meshes with prescribed displacements, and planes | Concatenated vertices/edges/faces of all obstacles |
| `CollisionProxy` | Separate (usually finer) surface used for contact instead of the FE boundary | Proxy mesh + sparse map from FE nodes to proxy vertices |

**Numbering.** Every entity type is numbered from 0. In conforming meshes, vertices and elements keep the input order and edges are sorted by their vertex pair; 3D faces are numbered cell by cell. Non-conforming meshes keep every entity ever created ("all" ids) and number the current ones ("valid" ids) in the same order; the `Mesh` interface always uses valid ids.

**Labels.** Body ids (per element, default 0) select materials, boundary ids (per boundary element: edges in 2D, faces in 3D; default max int on the boundary and -1 inside) select boundary conditions, node ids (per vertex, default -1) select nodal boundary conditions, and geometry ids (per element, default 0) identify the input geometry of each element. Refinement drops the labels of conforming meshes (the loader recomputes them from the selections) and passes the body/boundary ids of non-conforming meshes to the children; appending keeps the labels of both meshes.

**Navigation.** `Navigation::Index` (2D) and `Navigation3D::Index` (3D) are flags of incident entities (vertex, edge, face, and in 3D element); the `switch_*` functions move to the flag that differs in one entity. Non-conforming meshes must be prepared (`prepare_mesh()`) after their topology changes and before they are queried.

Meshes are written in VTU (the format used by paraview/VTK) by `polyfem::io`.

### Quadrature Rules

`polyfem::quadrature` contains a set of quadrature rules for all elements implemented in `polyfem::basis`. The quadrature library is mostly independent from the rest of the codebase and it is based on the open-source library quadpy.

### Time Integration

For transient problems, `polyfem::time_integrator` implements first-order (Implicit Euler `ImplicitEuler.hpp`) and higher-order time integrators (Implicit Newmak `ImplicitNewmark.hpp`, BDF `BDF.hpp`).

### Collection of PDEs

Polyfem contains right hand side, boundary conditions, exact solutions, and initial conditions for many common PDEs. We refer to the documentation in the `polyfem::problem` section for more details. The major physics supported by polyfem are:

1. Laplace
2. Helmholtz
3. Linear Elasticity
4. Saint-Venant Elasticity
5. Neo-Hookean Elasticity
6. Stokes
7. Navier-Stokes

From these, 1, 3--5, and 7 supports both static and transient problems, and 3--5 model contact and friction forces using the Incremental Potential Contact formulation (https://github.com/ipc-sim/ipc-toolkit).

### Automatic Code Generation

`polyfem::autogen` contains python scripts to generate C++ code for:
* Low-order basis in `polyfem::basis`
* All quadrature rules used in `polyfem::quadrature`
* All non-trivial rhs assembly in `polyfem::assembler`

The code autogenerated by the python scripts is directly committed in the code repository, as the autogeneration is time consuming.

### Utilities

Smaller utilities are grouped in the `polyfem::utils` module, including mesh IO, hashing functions, logging, matrix utilities, parallelization, rasterization, function interpolation (`RBFInterpolation.hpp`), and timers (`Timer.hpp`).

## Simple Example / Code Walkthrough

In this section, a brief, high-level code walkthrough for a simple example problem is given.

### JSON File

For this example, we solve the Laplacian on a (triangular) plate-hole mesh with Dirichlet boundary conditions and a constant RHS. We choose P1 elements implicitly (as this is the default). See the [JSON specification](https://polyfem.github.io/json_defaults_and_spec/) for more details.

```json
{
    "geometry": {
        "advanced": {
            "normalize_mesh": true
        },
        "mesh": "plate_hole.obj",
        "surface_selection": {
            "threshold": 1e-08
        }
    },

    "materials": {
        "type": "Laplacian"
    },

    "boundary_conditions": {
        "dirichlet_boundary": [{
            "id": 1,
            "value": 0.0
        }, {
            "id": 4,
            "value": 1.0
        }],
        "rhs": 10
    },

    "output": {
        "json": "stats.json",
        "paraview": {
            "file_name": "result.vtu"
        }
    }
}
```

### State

Internally, PolyFEM keeps a `polyfem::State` object that contains the entire state of the solve. The mesh, basis, boundary conditions, stiffness/mass matrices, RHS, and user-selected settings are all stored in `polyfem::State`. The `polyfem::State` class is also responsible for dispatch, meaning it contains member functions for the basic steps of a solve, including building a basis, assembling matrices, and actually performing the linear or nonlinear solve.

### Basis Building

After creating a `polyfem::State` object and loading the input mesh, the next step in the code is to build the basis. In PolyFEM, there are two separate notions of basis: the finite element (FE) basis and the geometric basis. (They are the same in the isoparametric case.) The former is used to construct the solution whereas the latter is used to represent the geometric mapping, which is especially useful to implement curved elements.

Both the FE and geometric basis are stored as vectors of `basis::ElementBases`. Each `basis::ElementBases` object stores the basis functions for a given element as a vector of `basis::Basis`. Each `basis::Basis` stores functions representing a local basis function and its gradient.

The high-level function that constructs the FE and geometric bases is `State::build_basis`. In our example, it uses `basis::LagrangeBasis2d` to create the vector of `basis::ElementBases`.

### Basis/Gradient Evaluation and Storage

In order to evaluate local basis functions and construct mass/stiffness matrices, PolyFEM makes use of a series of data storage classes. The lowest level of these classes is `assembler::AssemblyValues`, which stores the evaluation and gradient of a single local basis function on its element's quadrature points. A vector of `basis::Local2Global` objects is also stored. This is used to constrain nodes at the interface between different order elements (allowing for refinement).

The `assembler::ElementAssemblyValues` class stores per element information, including quadrature points in real space, the geometric mapping's determinant, and a vector of `assembler::AssemblyValues` objects (one for each local basis function).

Finally, `assembler::AssemblyValsCache` stores a vector of `assembler::ElementAssemblyValues`, boosting performance by caching the per element information. Its data can be accessed through its `compute` function.

These three classes are the core data storage objects used during matrix assembly. These classes use the previously constructed `basis::ElementBases` objects to evaluate basis functions and their gradients.

### RHS Assembly

After building the basis, the next step is to assemble the RHS. This is done through the `State::assemble_rhs` function, which itself creates a `assembler::RhsAssembler` object. The `assembler::RhsAssembler::assemble` function is then called in order to construct the RHS vector of the weak form using `assembler::AssemblyValsCache` and the user-provided RHS function. In our simple example, the latter is just $10$. Note that the RHS function and boundary conditions are stored in `assembler::GenericProblem`.

### Matrix Assembly

The virtual class `assembler::Assembler` provides the interface used to actually construct matrices through its `assemble` function. For our simple example, the relevant subclass is `assembler::LinearAssembler`, which is itself a virtual class that defines how to assemble global matrices. However, the function that defines how to evaluate local matrix entries according to the chosen bilinear form is a pure virtual function. In our example, the stiffness matrix is built using `assembler::Laplacian`, which is a subclass of `assembler::LinearAssembler` that defines the local matrix entries (ie grad u dot grad v). The `assembler::Mass` class is similar except for building the mass matrix.

The relevant high-level functions for matrix assembly are `State::assemble_mass_mat` and `State::build_stiffness_mat`. Note that the stiffness matrix is built later on (in the `State::solve_linear` function).

### Solver

After assembling the necessary matrices/vectors, the `State::solve_problem` function dispatches `State::init_linear_solve` then `State::solve_problem`, which solves the linear system.

### Diagram

A diagram showing the relationships discussed above is included below. Each block represents its own class. `polyfem::State` is shown in orange and `basis::LagrangeBasis2d` in red. The basis storage classes are blue, the value storage classes purple, and the assembly-related classes green. Relationships between classes are shown with arrows. Bold arrows represent calling a member function and dashed ones represent inheritance.

![Overall diagram](example_diagram.png)

#### Technical Note: Matrix Caching

In order to boost performance, PolyFEM caches sparsity structure when assembling matrices, taking advantage of the fact that the structure of stiffness/mass matrices remains unchanged assuming the basis is unchanged. The details can be found in `MatrixCache.hpp`. A brief overview of `polyfem::utils::SparseMatrixCache` is included here for reference.

In PolyFEM, sparse matrices are stored in [Compressed Sparse Column (CSC)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)) form. Thus, the matrix is represented as three arrays: two for storing non-zero indices and one for storing values.

Consider the first time a `polyfem::utils::SparseMatrixCache` object is filled with values. Each time `SparseMatrixCache::add_value` is called, the values and their global indices are placed in a temporary buffer. It is assumed that a given `polyfem::utils::SparseMatrixCache` always receives each entry from a given element in the same order.

The first time the matrix is retrieved using `SparseMatrixCache::get_matrix`, `SparseMatrixCache::prune` is called, which writes the buffered values to the actual matrix. Then, the stored index information is used to create a mapping from the ordering of values received from a given element to the corresponding CSC-style index. In future iterations, instead of recomputing the matrix structure, `polyfem::utils::SparseMatrixCache` uses this cached index to save the given value directly to the CSC-style values array at the correct position.

## Building PolyFEM as a stand-alone executable

All the C++ dependencies required to build the code are included and are compiled statically in a single executable. PolyFEM is compiled using automatic builds on Windows, macOS, and Linux using:

```bash
mkdir build
cd build
cmake ..
make -j4
```

After compilation, unit and system tests (`tests/main.cpp`) can be run with:

```bash
./tests/unit_tests
```

## Building PolyFEM as a static library

**Polyfem** can be added to an existing `cmake` project with
```cmake
add_subdirectory(<path-to-polyfem> polyfem)
```
and linked with
```cmake
target_link_library(<your_target> polyfem::polyfem)
```
in your cmake script. We do not currently support other building systems. PolyFEM will download the dependencies that it needs in the build folder.
