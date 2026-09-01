# Data integration tests

PolyFEM's integration tests read inputs from several pinned repositories. The
category manifests below cover PolyFEM entrypoints in `POLYFEM_DATA_DIR` (the
`polyfem-data` repository). `pref_test_list.txt` and
`polyspline_test_list.txt` cover `POLYFEM_PREF_DIR` and
`POLYFEM_POLYSPLINE_DIR`; optimization tests use `POLYFEM_DIFF_DIR` directly.

`all PolyFEM data JSON files are classified` is scoped to `POLYFEM_DATA_DIR`.
It requires every active PolyFEM entrypoint there to appear exactly once in:

- the existing run manifests for scenes with golden numerical values;
- `triangle.txt`, which runs only when `POLYFEM_WITH_TRIANGLE` is enabled;
- `slow.txt`, which runs in the nightly Linux Release/CPP and Release/TBB jobs; or
- `known_issues.txt` for scenes with a documented solver/runtime bug.

Configuration files reached through `common` are checked as dependencies, not
as standalone scenes. JSON inputs belonging to other tools are not PolyFEM
entrypoints. Files under `data/old-tolerances` are archival and excluded. A new
PolyFEM scene must be added to a run manifest; it cannot silently land without
coverage. Temporarily prefix a manifest entry with `*` to write golden `tests`
values into its JSON.
