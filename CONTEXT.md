# PolyFEM Domain Context

## Terms

### VarForm

A variational formulation runtime for the new forward-simulation path. It owns formulation-specific solve behavior while the new `State` owns application setup and mesh loading.

### General FE Space

A finite element space for one formulation field, without primary or auxiliary status. Scalar value, displacement, velocity, pressure, and bilaplacian helper fields should all use the same concept.

Each general FE space carries a shared geometry mapping handle, represented in code as a `std::shared_ptr` to the geometry mapping module. Geometry is not itself a solved FE space; it is the coordinate map used to interpret a field space.

A general FE space does not own an intrinsic name. Field names such as `velocity`, `pressure`, and `displacement` belong to VarForm-specific spaces structs because they are semantic lookup, output, and configuration labels.

### Geometry Mapping

The finite element coordinate map used to interpret one or more general FE spaces on the mesh. It owns geometry bases and geometry-specific polygonal data. Most VarForms use one shared geometry mapping.

### FE Space Collection

The set of general FE spaces used by a VarForm. A single-field formulation has one general FE space; mixed and multiphysics formulations have more than one.

The default invariant is that all general FE spaces in a VarForm share one geometry mapping. Multiple geometry mappings are allowed only when a formulation has a concrete reason, such as fields on different meshes or a future ALE-style reference/current mapping split.

Do not introduce a central FE space registry unless multiple independent modules need shared lookup into the same FE spaces. Current VarForm architecture does not have that use case; each VarForm should own the spaces it requires through a concrete spaces struct.

### Solution Layout

The mapping between named formulation fields and rows in the exposed `Eigen::MatrixXd` solution. The layout belongs with the FE space collection, not with individual VarForm inheritance hooks.

A solution layout may contain FE-space blocks and algebraic blocks. FE-space blocks refer to a named general FE space. Algebraic blocks represent rows that are not sampled as a field, such as an average-pressure constraint row. The layout also records which blocks participate in time integration.

The solution layout interface uses `total_dof` terminology rather than matrix-row terminology. Its public operations should be vector-based because the solved state is a vector. It should not provide assertion helpers for shape validation; callers can compare a vector size against `total_dof()` directly.

A `SolutionBlock` stores one contiguous segment in the solver vector: `offset`, `dof`, `is_time_integrated`, and `is_algebraic`. The name `is_time_integrated` is preferred over `is_time_integrand` because the block is state advanced by a time integrator, not an integrand. The name `is_algebraic` marks constraint or multiplier rows that are not sampled as FE-space fields.

The initial `SolutionLayout` interface should stay minimal: `int add_block(...)`, `SolutionBlock get_block(int id) const`, and `int total_dof() const`. Do not add block-view helpers, assertion helpers, public block iteration, or cached totals until current VarForm code proves the need.
