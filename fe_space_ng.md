# FE Space Next Generation Design

## Goal

Replace the current primary/auxiliary FE-space split with a small set of concrete data modules:

- `GeometryMapping`
- `FESpace`
- `SolutionBlock`
- `SolutionLayout`
- VarForm-specific spaces structs such as `FluidSpaces` or `BilaplacianSpaces`

The design should reduce duplicated mixed-form layout logic without creating a central FE-space registry or a new inheritance hierarchy.

## Current Problem

`VarForm` currently owns the first FE space directly:

- `bases`
- `n_bases`
- `disc_orders`
- `disc_ordersq`
- `geom_bases_`
- `n_geom_bases`
- `mesh_nodes`
- `geom_mesh_nodes`
- polygonal geometry data

Mixed forms then duplicate additional space state:

- `FluidVarForm`
- `IncompressibleElasticVarForm`
- `BilaplacianVarForm`

Each repeats pressure/helper basis construction, stacked-solution offsets, splitting, RHS resizing, mass expansion, and pressure output handling.

The desired design has no privileged "primary" FE space. A formulation owns the FE spaces it needs.

## GeometryMapping

`GeometryMapping` represents the finite element coordinate map used to interpret one or more FE spaces on a mesh.

Geometry is not a solved field. It is not itself an `FESpace`.

```cpp
struct GeometryMapping
{
    bool isoparametric = false;

    std::vector<basis::ElementBases> bases;
    int n_bases = 0;

    std::map<int, Eigen::MatrixXd> polys;
    std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

    std::shared_ptr<mesh::MeshNodes> mesh_nodes;
};
```

Most VarForms should use one shared `GeometryMapping`. Multiple geometry mappings are not justified today, but the `std::shared_ptr` handle on each `FESpace` leaves that option open if a future formulation has fields on different meshes or an ALE-style reference/current mapping split.

## FESpace

`FESpace` is an anonymous finite element discretization for one formulation field.

It does not own a name. Names like `velocity`, `pressure`, and `displacement` belong to VarForm-specific spaces structs because they are semantic labels, not basis data.

```cpp
struct FESpace
{
    int value_dim = 1;
    int n_bases = 0;

    Eigen::VectorXi disc_orders;
    Eigen::VectorXi disc_ordersq;

    std::vector<basis::ElementBases> bases;
    std::shared_ptr<const GeometryMapping> geometry;

    std::shared_ptr<mesh::MeshNodes> mesh_nodes;
    assembler::AssemblyValsCache cache;

    int ndof() const
    {
        return n_bases * value_dim;
    }
};
```

`FESpace` is intentionally a data module. It should not become an abstract getter interface. Existing assembly and output code can take `const FESpace &` and use the data it needs.

The useful invariant is that the field bases, value dimension, node mapping, assembly cache, and geometry mapping handle travel together.

## No Central FE Space Registry

A central FE-space registry is not justified in the current architecture.

Only one VarForm is active for a simulation. Two independent VarForms do not share one FE space. If future multiphysics work needs several fields, it should be represented as one coupled VarForm or a composite formulation owning the spaces it needs.

Use concrete VarForm-specific spaces structs:

```cpp
struct ElasticSpaces
{
    FESpace displacement;
    SolutionLayout layout;
    int displacement_block = -1;
};

struct FluidSpaces
{
    FESpace velocity;
    FESpace pressure;

    SolutionLayout layout;
    int velocity_block = -1;
    int pressure_block = -1;
    int pressure_mean_constraint_block = -1;
};

struct BilaplacianSpaces
{
    FESpace value;
    FESpace helper;

    SolutionLayout layout;
    int value_block = -1;
    int helper_block = -1;
};
```

Shared builder functions are still useful:

```cpp
GeometryMapping build_geometry_mapping(...);
FESpace build_lagrange_space(...);
FESpace build_spline_space(...);
FESpace build_pressure_space(...);
```

Builders provide leverage without creating a registry.

## SolutionBlock

`SolutionBlock` describes one contiguous segment in the solver vector.

```cpp
struct SolutionBlock
{
    int offset = 0;
    int dof = 0;
    bool is_time_integrated = false;
    bool is_algebraic = false;
};
```

`offset`
: First degree of freedom in the global solver vector.

`dof`
: Number of degrees of freedom in the block.

`is_time_integrated`
: True when the block is advanced by the time integrator. In current mixed forms this is true for velocity, displacement, or scalar value blocks and false for pressure/helper blocks.

`is_algebraic`
: True when the block is a constraint or multiplier row rather than a sampled FE-space field. Current real use case: the fluid average-pressure constraint row.

Use `is_time_integrated`, not `is_time_integrand`. An integrand is the expression inside an integral; this flag marks state advanced by a time integrator.

Use `is_algebraic`, not `is_algebratic`.

## SolutionLayout

`SolutionLayout` owns the structural mapping from block id to solver-vector segment.

It should stay minimal:

```cpp
class SolutionLayout
{
public:
    int add_block(
        int dof,
        bool is_time_integrated,
        bool is_algebraic = false);

    SolutionBlock get_block(int id) const;

    int total_dof() const;

private:
    std::vector<SolutionBlock> blocks_;
};
```

`add_block`
: Appends a block and assigns its `offset` from the current `total_dof()`. Returns the block id.

`get_block`
: Returns a block by value. Returning by value prevents callers from mutating stored offsets.

`total_dof`
: Computes the full solver-vector size. Do not cache this; the number of blocks is small.

Do not add vector block views, assertion helpers, resize helpers, public block iteration, or cached totals until the current VarForm code proves the need.

Caller-side validation should remain explicit:

```cpp
assert(x.size() == layout.total_dof());
```

## Fluid Average-Pressure Constraint

`FluidVarForm` currently has one algebraic block when pressure is not otherwise anchored:

```text
[ velocity dofs ]
[ pressure dofs ]
[ mean-pressure constraint row ]
```

The last block is not an FE space. It should not be sampled, interpolated, mass-assembled, or output as pressure. It exists to enforce a zero-mean pressure gauge and avoid pressure being determined only up to an additive constant.

## Atomic Refactor Plan

This refactor should land as one coherent change. The repo should not keep an intermediate model where some VarForms use direct `VarForm` FE-space fields and others use the new spaces model.

The public `VarForm` seam should remain stable during the refactor:

```cpp
init(...);
set_mesh(...);
solve(...);
compute_errors(...);
save_json(...);
export_data(...);
```

The internal change is:

```text
VarForm-owned FE-space fields
  -> VarForm-specific spaces structs
  -> FESpace + GeometryMapping + VarFormBoundaryState + SolutionLayout
```

Single-pass checklist:

1. Add the new data modules:
   - `GeometryMapping`
   - `FESpace`
   - `SolutionBlock`
   - `SolutionLayout`
   - `VarFormBoundaryState`
2. Add concrete spaces structs:
   - `ScalarSpaces`
   - `ElasticSpaces`
   - `FluidSpaces`
   - `IncompressibleElasticSpaces`
   - `BilaplacianSpaces`
   - nonlinear elasticity can initially reuse `ElasticSpaces` plus its contact runtime state
3. Move current geometry state into `GeometryMapping`:
   - `geom_bases_`
   - `n_geom_bases`
   - `geom_mesh_nodes`
   - `polys`
   - `polys_3d`
4. Move current FE-space state into concrete `FESpace` members:
   - `bases`
   - `n_bases`
   - `disc_orders`
   - `disc_ordersq`
   - `mesh_nodes`
   - `ass_vals_cache`
   - pressure/helper space equivalents in mixed VarForms
5. Move current boundary vectors unchanged into `VarFormBoundaryState`.
   - Keep current names and semantics.
   - Do not redesign boundary behavior in this refactor.
6. Build `SolutionLayout` in each spaces struct and replace duplicated block helpers:
   - `primary_ndof()`
   - `pressure_block_size()`
   - `stacked_ndof()`
   - `split_solution(...)`
   - repeated stacked RHS bottom-block zeroing
7. Rewrite common VarForm helper code to take explicit data:
   - `const FESpace &`
   - `const GeometryMapping &`
   - `const VarFormBoundaryState &`
8. Remove the old direct FE-space and boundary fields from `VarForm`.
9. Update tests and `VarFormTestAccess` to inspect spaces instead of direct `VarForm` fields.

The final state should have no compatibility duplication. If a temporary forwarding helper is needed while editing, it should be deleted before the refactor is considered done.
