# Non-conforming mesh resources

A structured FEM mesh containing an `nc` group loads as `NCMesh2D` (triangles)
or `NCMesh3D` (tetrahedra), even when `State::load_mesh()` uses its default
arguments. No solver-specific mesh replacement is needed. Geometry selections,
transformations, arrays, and additional `n_refs` use the normal geometry loader.

An input bundle's `/config` can reference `"mesh": "/meshes/nc"` in its usual
geometry entry. Run the bundle with `PolyFEM_bin --hdf5 input.h5`; the NC payload
selects the runtime mesh implementation automatically.

All mesh output uses mesh schema version 2. Version 1 resources remain readable.
The ordinary `vertices`, `cells`, and label datasets describe the active mesh.
The `nc` subgroup has `schema_version=1` and contains full storage, including
inactive ancestors and ghost descendants:

| Dataset | Contents |
| --- | --- |
| `vertices` | All vertex positions |
| `cells`, `ordered_cells` | Geometric and runtime local vertex order |
| `cell_edges`, `cell_faces` | Full edge and face indices per element |
| `children` | Four child slots in 2D, eight in 3D; -1 for absent children |
| `element_state` | Level, parent, refined flag, ghost flag, body ID, geometry ID |
| `edges`, `faces` | Sorted vertex indices followed by the boundary ID |
| `midpoints` | Sorted edge endpoints and their midpoint vertex index |
| `node_ids` | Node labels for all stored vertices; -1 for unlabeled new nodes |
| `refinement_history` | Full element indices in operation order |

`cell_faces` and `faces` are omitted in 2D. The `label_flags` attribute records
which label categories are present (body=1, geometry=2, node=4, boundary=8).
All indices in the payload refer to full storage, not the active arrays.
Active indices are rebuilt in ascending full-storage order. Lookup maps,
incidence, adjacency, hanging relationships, and interpolation weights are
rebuilt rather than serialized.

Input groups and checkpoint groups use the same production mesh encoder.
Validation checks hierarchy reciprocity, acyclic references, midpoint
relationships, primitive connectivity, and consistency with the active mesh.
A flat list of nonmatching cells without an NC payload does not reconstruct
hanging relationships. For refinement, convert a conforming simplex mesh explicitly with
`mesh->to_nonconforming()`. `Mesh::create(data)` and all loading APIs infer the
runtime implementation solely from the stored data.

`tests/NCMeshTestSupport.hpp` provides small deterministic mesh generators with
configurable global/local/global refinement and a selection callback. Its 2D
example selects by barycenter; its 3D example selects elements with a vertex on
the negative side of a plane. `write_nc_bundle()` writes temporary input and
checkpoint mesh groups through the production encoder. These helpers do not
construct or mutate a solver state.

Run the focused tests with `unit_tests '[ncmesh]'` and the compatibility tests
with `unit_tests '[hdf5]'`.
