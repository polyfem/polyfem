# Optimization JSON spec migration guide

## New auto parameter mode

When there's only one variable to simulation in the config and and there's no slice composition map, you can enable auto mode.

```json
parameters = "auto"
```

instead of 

```json
parameters = [
    { "number": 143888 }
]
```

The optimization parameters dof and initial guess will be deduced automatically. Manual specification remain unchanged, this is a QOL feature.

## dirichlet variable to simulation is renamed

The new name is `dirichlet-boundary`, the aim is to reduce confusion with dirichlet node var2sim.

## Composite map is replaced by active selection

`/variable_to_simulation/*` json spec used to have `composite_map_type` and `composite_map_indices` field. Allowing user to select active adjoint optimization parameters in following ways:

- raw indices of optimization variable
- geometry nodes via various selection mode
- active dimensions (xyz)
- time slice indexes

However actual support is spotty in the code. A lot of var2sim class does not respect the selection. For example,

```json
// Does not work. The code ignore composite map internally.
{
  "type": "initial",
  "state": 0,
  "composition": [],
  "composite_map_type": "boundary",
  "surface_selection": [1],
}
```

```json
// Does not work. The code ignore composite map internally.
{
  "type": "friction",
  "state": 0,
  "composition": [],
  "composite_map_type": "indices",
  "composite_map_indices": [0],
}
```

The new spec design supports a much more limited set of active parameter selection methods augmented with strict validation.

### Migration example

Quick Before/After comparision.

**Legacy**

```json
{
  "type": "shape",
  "state": 0,
  "composition": [],
  "composite_map_type": "boundary_excluding_surface",
  "surface_selection": [3, 8],
  "active_dimensions": [0, 2]
}
```

Above config translates to:

- only optimize geometry nodes on the boundary
- exclude surfaces `3` and `8`
- only touch dimensions `0` and `2`

**New**

```json
{
  "type": "shape",
  "state": 0,
  "composition": [],
  "active_geometry_nodes": {
    "type": "boundary_excluding_surface",
    "selection": [3, 8]
  },
  "active_dimensions": [0, 2]
}
```

The "composite_map_type", "composite_map_indices" and "surface_selection" json fields are replaced by respective "active_xxx" field. Which we will discuss in the next section.

### Active selection spec

In total there are 5 kinds of active optimization parameter selection:

- `active_geometry_nodes`: specify vertex id. Support various selection modes.
   Supported by: Shape var2sim, dirichlet nodes var2sim
- `active_time_slices`: specify time slice indexes.
   Supported by: dirichlet boundary var2sim, pressure var2sim
- `active_dimensions`: specify cartesian dimension of geometry nodes.
   Supported by: Shape var2sim
- `active_dofs`: specify initial condition dof.
   Supported by: Initial condition var2sim
- `active_boundary_ids`: specify boundary id (selection id).
   Supported by: dirichlet boundary var2sim, pressure var2sim

All of them can accept either a json array of int (ex. `"active_geometry_nodes": [1, 2, 3]`) or a file containing such indices. By default, active selection is an empty json list, which implies all active.

**Example: boundary id and time slice selection**

```json
{
  "type": "dirichlet-boundary",
  "state": 0,
  "composition": [],
  "active_boundary_ids": [2, 5],
  "active_time_slices": [0, 2, 4]
}
```

### Advanced selection mode for `active_geometry_nodes`

In addition, `active_geometry_nodes` support advanced selection mode, which does vertex selection based on PolyFEM selection id.

**Overall structure**

```json
"active_geometry_nodes": { "type": <advanced selection type>, "selection": [ <indices> ] }
```

**Select all interior vertices from volume selection id**

```json
"active_geometry_nodes": { "type": "interior", "selection": [1, 2] }
```

**Select all boundary vertices from surface selection id**

```json
"active_geometry_nodes": { "type": "boundary", "selection": [3, 8] }
```

**Select all boundary vertices except surface selection id**

```json
"active_geometry_nodes": { "type": "boundary_excluding_surface", "selection": [3, 8] }
```


**Example: shape boundary selection**

```json
{
  "type": "shape",
  "state": 0,
  "composition": [],
  "active_geometry_nodes": {
    "type": "boundary",
    "selection": [1, 4]
  }
}
```

## Migration cheatsheet

- use auto parameters mode for simple case
- rename dirichlet var2sim to dirichlet-boundary var2sim. 
- migrate legacy `/variable_to_simulation/*/composite_map_type` modes to their new equivalents:
    - `none` → No migrate required.
    - `indices` → No replacement because this is too brittle.
    - `interior` → `active_geometry_nodes: { "type": "interior", "selection": [volume ids] }`
    - `boundary` → `active_geometry_nodes: { "type": "boundary", "selection": [surface ids] }`
    - `boundary_excluding_surface` → `active_geometry_nodes: { "type": "boundary_excluding_surface", "selection": [surface ids] }`
    - `time_step_indexing` → `active_time_slices: [..]`
