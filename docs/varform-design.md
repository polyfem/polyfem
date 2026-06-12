# VarForm Architecture

## Status

This document describes the VarForm architecture on the `varform` branch after
the legacy `State` split.

The current migration target is:

- preserve unsupported functionality through `polyfem::legacy::State`;
- make the new forward-simulation path small and understandable;
- move physics-specific solve and output behavior into VarForms;
- keep shared FEM, mesh, solver, and assembler code independent of either
  `State`;

## Runtime Routing

`src/polyfem/main.cpp` and `app/app.cpp` are the application boundaries between
the new and legacy paths.

1. The input JSON is loaded and command-line overrides are applied.
2. `varform::uses_varform_state(args)` asks `VarFormFactory` whether the setup is
   supported by the new path.
3. Supported setups run through `polyfem::State`.
4. Unsupported setups run through `polyfem::legacy::State`.

The routing decision is intentionally temporary. It allows the new path to have
one coherent design without partially supporting legacy-only features.

Current reasons to select the legacy path include:

- optimization and adjoint workflows;
- remeshing;
- periodic boundary conditions and periodic contact;
- formulations not represented by a VarForm;
- homogenisation;
- unsupported combinations of contact, pressure boundary conditions, and
  constraints.

## Ownership Boundaries

### `polyfem::State`

The new `State` is a thin application facade. It owns:

- validated and defaulted input arguments;
- logger and thread initialization;
- mesh loading from application-facing input formats;
- the selected `VarForm`;
- the high-level `load_mesh()` and `solve()` entry points.

It does not own FE bases, assemblers, solver state, output sampling, mesh, or the
solution.

### `varform::VarForm`

Every current VarForm has one primary FE space. The base class owns the data and
operations shared by that primary space:

- the mesh and geometric mapping data;
- the primary FE bases and discretization orders;
- the primary assembler, mass assemblers, RHS assembler, and assembly caches;
- boundary-node and boundary-element data;
- preparation, common output, statistics, and output-file lifecycle;
- the formulation's input arguments, units, paths, and timings.

The base class does not attempt to represent every FE space in a future
multiphysics problem.

### Derived VarForms

Derived VarForms own physics-specific behavior and any additional FE spaces.

- `ScalarVarForm` owns scalar linear solves.
- `ElasticVarForm` owns output and initialization shared by elasticity forms.
- `LinearElasticVarForm` owns linear static and transient elasticity solves.
- `NonlinearElasticVarForm` owns nonlinear/contact state shared by its static
  and transient subclasses.
- `FluidVarForm` owns velocity-pressure mixed data shared by Stokes and
  Navier-Stokes.
- `IncompressibleElasticVarForm` owns its duplicated pressure space and mixed
  assemblers.
- `BilaplacianVarForm` owns its duplicated auxiliary scalar space and mixed
  assemblers.

Duplicating auxiliary-space bookkeeping in mixed VarForms is intentional. It
keeps the base class honest: the only universal FE-space assumption is that a
formulation has at least one primary space.

### `polyfem::legacy::State`

The legacy state is a preserved implementation for functionality that has not
been migrated. New VarForms and new I/O must not depend on it.

Optimization and adjoint code currently use the legacy state explicitly.

## Lifecycle

The intended new-path lifecycle is:

```text
State::init(args)
  -> validate/default args
  -> select and initialize VarForm

State::load_mesh(...)
  -> construct mesh
  -> VarForm::set_mesh(mesh)
  -> VarForm-specific mesh initialization

State::solve(solution)
  -> VarForm::solve(solution)
  -> prepare primary and auxiliary spaces
  -> solve static or transient problem

caller
  -> compute_errors(solution)
  -> save_json(solution)
  -> export_data(solution)
```

Preparation is an internal, idempotent lifecycle step. Production callers use
`solve()`, while tests that need prepared FEM data use `VarFormTestAccess`.
`init()` and `set_mesh()` invalidate prepared data, and derived VarForms clear
their owned solve state during reset.

## Solution Layout

All VarForms expose one `Eigen::MatrixXd` solution.

Single-space forms store only their primary unknowns.

Mixed forms store one stacked solution:

```text
[ primary-space DOFs ]
[ auxiliary-space DOFs ]
```

Each mixed VarForm is responsible for:

- computing its block sizes;
- expanding primary matrices into the stacked system;
- splitting the stacked solution for solving and output;
- keeping algebraic auxiliary variables out of primary time integration.

This is the intended direction for future multiphysics work: callers see one
solution, while the VarForm owns its block interpretation.

## Output Architecture

Output intentionally remains part of the VarForm lifecycle for this change.
The writer and VarForm are coupled by small data contracts rather than by
writer access to VarForm internals.

### Geometry Contract

`io::OutputSpace` contains only data needed to construct output geometry:

- mesh;
- geometric mapping bases;
- output sampling orders;
- polygon/polyhedron geometry;
- local boundary data;
- optional obstacle and collision geometry;
- optional output point-node information.

It does not contain physics assemblers, FE solution bases, or a `State`.

### Sampling Contract

`io::OutGeometryData` constructs volume, surface, wire, contact, grid, and point
samples. It passes each sample to an `io::OutputFieldFunction`.

`io::OutputSample` describes the sampled locations and their association with
the simulation mesh:

- physical and local coordinates;
- element, primitive, and node IDs where applicable;
- normals;
- sample domain and output-cell count;
- time and time step;
- optional explicitly requested fields.

### Physics Contract

`VarForm::output_fields(sample, solution, options)` returns named
`io::OutputField` values. Each field declares point or cell association.
The writer validates each field's row count against the sample before passing
it to Paraview.

The VarForm owns:

- the meaning of the solution blocks;
- interpolation using its FE spaces;
- physics-specific derived quantities;
- deciding whether an expensive field should be computed;
- compatibility aliases such as `solution`.

The geometry writer owns:

- output mesh construction and sampling;
- Paraview/HDF5/VTM/PVD serialization;
- geometry-only fields such as normals, sidesets, and discretization order.

The writer must not depend on `State`, `legacy::State`, or a concrete VarForm
subclass.

## Adding a VarForm

A new single-space VarForm should:

1. derive from `VarForm`, `ScalarVarForm`, or `ElasticVarForm` as appropriate;
2. initialize its assembler, mass assembler, and problem;
3. override `solve_problem()`;
4. implement `output_fields()` and `save_json()`;
5. add factory support and routing tests;
6. add representative numerical, output, and restart tests.

A new mixed or multiphysics VarForm should additionally:

1. own each auxiliary FE space and assembler explicitly;
2. define and document the stacked solution layout;
3. split the solution internally for solves and output;
4. define which blocks are time integrated;
5. avoid adding auxiliary-space assumptions to the base `VarForm`.

## Test Strategy

The migration needs four complementary kinds of tests:

- routing tests: verify supported and unsupported scenes select the intended
  state path;
- numerical integration tests: verify stored error metrics for representative
  formulations;
- output contract tests: verify field names, associations, dimensions, and
  representative values;
- restart tests: verify every transient VarForm can save and resume its own
  time-integrator state.

Direct legacy-versus-VarForm comparisons are useful for a small representative
set while the migration is active. They should not become a permanent
requirement after the legacy path is removed.

## Known Gaps

The following are known migration or hardening tasks, not intended parts of the
final architecture:

- restore and diagnose the disabled 2D homogenization integration test;
- design state-file naming and restart behavior for VarForms with multiple
  independent time integrators;
- add stronger output-field and legacy-equivalence tests;
- remove remaining compatibility-only output after the legacy path is retired.

## Deferred Work

Thermoelasticity is intentionally deferred. It will be the first substantial
test of:

- multiple named or identified FE spaces;
- multiple time integrators in one VarForm;
- coupled nonlinear and auxiliary blocks;
- FE-space selection in materials, boundary conditions, and discretization
  configuration.

That work should build on the current ownership boundary rather than making the
base `VarForm` a container for every possible auxiliary space.
