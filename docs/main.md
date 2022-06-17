Polyfem as Libary
=================

**Polyfem** uses modern `cmake`, so it it should be enough to add this line
```cmake
add_subdirectory(<path-to-polyfem> polyfem)
```
in your cmake project, and then simply add
```cmake
target_link_library(<your_target> polyfem)
```
in your cmake script.
Polyfem will download the dependencies that it needs with the version that it needs.
If you dont need the viewer for your own project you can add
```cmake
SET(POLYFEM_NO_UI ON)
```


Interface
---------

The interface of polyfem is similar as [Python](polyfempy_doc.md). You should create a `polyfem::State` object and then call methods on it.
Most of the fields are public for convenience but we discourage use or access them.


This is the main interface of `polyfem::State`.


### Initialization

```c++
void init(const json &args)
void init(const std::string &json_path)
```

loads the settings from a json object or file

### Logging

```c++
void set_log_level(int log_level)
std::string get_log()
```
Sets the log level (1-6) and gets the log at the end


### Loading mesh

```c++
void load_mesh()
void load_mesh(const std::string &path)
void load_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
```
Different ways of loading a mesh

### Set boundary sides sets

```c++
void set_boundary_side_set(const std::function<int(const polyfem::RowVectorNd&)> &boundary_marker)
void set_boundary_side_set(const std::function<int(const polyfem::RowVectorNd&, bool)> &boundary_marker)
void set_boundary_side_set(const std::function<int(const std::vector<int>&, bool)> &boundary_marker)
```
All boundary side sets are assigned with a lambda function, the first 2 takes the barycenter of the face/edge, the last one the primite id. The second argument is a boolean that specifies if the sideset is boundary.


### Solving
```c++
void solve();
```

Note the solver internally calls
```c++
void build_basis();
void assemble_stiffness_mat();
void assemble_rhs();
void solve_problem();
```
You can use these instead of solve.

If you problem has a solution you can use
```c++
void compute_errors();
```
to obtain the error.


### Getters

```c++
const Eigen::MatrixXd &get_solution() const
const Eigen::MatrixXd &get_pressure() const
```
Gets the raw solution and pressure. The order of the coefficient is **unrelated** to the order of the vertices of the mesh.

```c++
void get_sampled_solution(Eigen::MatrixXd &points, Eigen::MatrixXi &tets, Eigen::MatrixXd &fun, bool boundary_only = false)
void get_stresses(Eigen::MatrixXd &fun, bool boundary_only = false)
void get_sampled_mises(Eigen::MatrixXd &fun, bool boundary_only = false)
void get_sampled_mises_avg(Eigen::MatrixXd &fun, Eigen::MatrixXd &tfun, bool boundary_only = false)
```
Gets the solution/stresses on the visualization mesh, use `vismesh_rel_area` to control density


### Exporting

Exports the solution to VTU for visualization

```c++
void get_sidesets(Eigen::MatrixXd &pts, Eigen::MatrixXi &faces, Eigen::MatrixXd &sidesets);

void export_data();

void save_vtu(const std::string &name);
void save_wire(const std::string &name, bool isolines = false);
```

