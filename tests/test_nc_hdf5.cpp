#include "NCMeshTestSupport.hpp"
#include "VarFormTestAccess.hpp"
#include <polyfem/State.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <h5pp/h5pp.h>
#include <chrono>

using namespace polyfem;
namespace
{
	struct Workspace
	{
		std::filesystem::path path = std::filesystem::temp_directory_path() / ("polyfem-nc-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
		Workspace() { std::filesystem::create_directories(path); }
		~Workspace() { std::filesystem::remove_all(path); }
	};

	void compare(const mesh::Mesh &a, const mesh::Mesh &b)
	{
		const auto x = a.to_mesh_data(), y = b.to_mesh_data();
		x.validate();
		y.validate();
		REQUIRE(x.nc);
		REQUIRE(y.nc);
		CHECK(x.vertices.isApprox(y.vertices));
		CHECK(x.elements == y.elements);
		CHECK(x.body_ids == y.body_ids);
		CHECK(x.geometry_ids == y.geometry_ids);
		CHECK(x.node_ids == y.node_ids);
		CHECK(x.boundary_elements == y.boundary_elements);
		CHECK(x.boundary_ids == y.boundary_ids);
		const auto &u = *x.nc, &v = *y.nc;
		CHECK(u.vertices.isApprox(v.vertices));
		CHECK(u.cells == v.cells);
		CHECK(u.ordered_cells == v.ordered_cells);
		CHECK(u.children == v.children);
		CHECK(u.element_state == v.element_state);
		CHECK(u.cell_edges == v.cell_edges);
		CHECK(u.cell_faces == v.cell_faces);
		CHECK(u.edges == v.edges);
		CHECK(u.faces == v.faces);
		CHECK(u.midpoints == v.midpoints);
		CHECK(u.refinement_history == v.refinement_history);
		CHECK(u.node_ids == v.node_ids);
		CHECK(u.label_flags == v.label_flags);
		for (int i = 0; i < a.n_vertices(); ++i)
		{
			if (a.dimension() == 2)
				CHECK(dynamic_cast<const mesh::NCMesh2D &>(a).leader_edge_of_vertex(i) == dynamic_cast<const mesh::NCMesh2D &>(b).leader_edge_of_vertex(i));
			else
			{
				CHECK(dynamic_cast<const mesh::NCMesh3D &>(a).leader_edge_of_vertex(i) == dynamic_cast<const mesh::NCMesh3D &>(b).leader_edge_of_vertex(i));
				CHECK(dynamic_cast<const mesh::NCMesh3D &>(a).leader_face_of_vertex(i) == dynamic_cast<const mesh::NCMesh3D &>(b).leader_face_of_vertex(i));
			}
		}
	}

	int coarsenable_child(const mesh::Mesh &m)
	{
		const auto data = m.to_mesh_data();
		const auto &n = *data.nc;
		for (int i = 0; i < n.children.rows(); ++i)
		{
			if (!n.element_state(i, 2) || n.element_state(i, 3))
				continue;
			bool leaves = true;
			for (int j = 0; j < n.children.cols(); ++j)
			{
				const int child = n.children(i, j);
				leaves &= child >= 0 && !n.element_state(child, 2) && !n.element_state(child, 3);
			}
			if (leaves)
				return n.children(i, 0);
		}
		return -1;
	}

	void coarsen(mesh::Mesh &m, int child)
	{
		if (auto *nc = dynamic_cast<mesh::NCMesh2D *>(&m))
			nc->coarsen_element(child);
		else
			dynamic_cast<mesh::NCMesh3D &>(m).coarsen_element(child);
		m.prepare_mesh();
	}

	void refine_full(mesh::Mesh &m, int parent)
	{
		if (auto *nc = dynamic_cast<mesh::NCMesh2D *>(&m))
			nc->refine_element(parent);
		else
			dynamic_cast<mesh::NCMesh3D &>(m).refine_element(parent);
		m.prepare_mesh();
	}

	json solve_config(int dimension, const std::filesystem::path &output)
	{
		const std::string exact = dimension == 2 ? "x*x+y*y" : "x*x+y*y+z*z";
		return json{
			{"materials", {{"type", "Laplacian"}}},
			{"geometry", {{{"mesh", "/meshes/nc"}}}},
			{"space", {{"discr_order", 2}, {"advanced", {{"bc_method", "lsq"}}}}},
			{"boundary_conditions", {{"dirichlet_boundary", {{{"id", "all"}, {"value", exact}}}}, {"rhs", dimension == 2 ? 4 : 6}}},
			{"output", {{"directory", output.string()}, {"reference", {{"solution", exact}, {"gradient", dimension == 2 ? json{"2*x", "2*y"} : json{"2*x", "2*y", "2*z"}}}}, {"paraview", {{"file_name", "nc.vtu"}, {"high_order_mesh", false}}}}},
			{"solver", {{"linear", {{"solver", "Eigen::SimplicialLDLT"}}}}}};
	}
} // namespace

TEST_CASE("NC HDF5 retains refinement lifecycle and ordering", "[ncmesh][hdf5]")
{
	State initialize_geogram;
	const int d = GENERATE(2, 3);
	const int local_refs = GENERATE(1, 2);
	const bool already_coarsened = GENERATE(false, true);
	Workspace workspace;
	auto original = tests::generate_nc_mesh(d, 0, local_refs);
	if (already_coarsened)
		coarsen(*original, coarsenable_child(*original));
	const auto path = workspace.path / "mesh.h5";
	tests::write_nc_bundle(path, *original, json::object());
	io::HDF5IO resources(path);
	auto loaded = mesh::MeshLoader(resources).load_fem("/meshes/nc");
	compare(*original, *loaded);
	io::CheckpointReader checkpoint(path);
	compare(*original, *checkpoint.read_mesh("/checkpoint/meshes/active"));
	compare(*loaded, *loaded->copy());
	tests::nc_refine_elements(*original, {0});
	tests::nc_refine_elements(*loaded, {0});
	compare(*original, *loaded);
	const int child = coarsenable_child(*original);
	REQUIRE(child >= 0);
	const int parent = original->to_mesh_data().nc->element_state(child, 1);
	coarsen(*original, child);
	coarsen(*loaded, child);
	compare(*original, *loaded);
	refine_full(*original, parent);
	refine_full(*loaded, parent);
	compare(*original, *loaded);
	const MatrixNd A = 2 * MatrixNd::Identity(d, d);
	const VectorNd b = VectorNd::Constant(d, 3);
	original->apply_affine_transformation(A, b);
	loaded->apply_affine_transformation(A, b);
	coarsen(*original, coarsenable_child(*original));
	coarsen(*loaded, coarsenable_child(*loaded));
	compare(*original, *loaded);
	original->normalize();
	loaded->normalize();
	compare(*original, *loaded);
	auto component = tests::generate_nc_mesh(d);
	component->apply_affine_transformation(MatrixNd::Identity(d, d), VectorNd::Constant(d, 4));
	original->append(*component);
	loaded->append(*component);
	compare(*original, *loaded);
	loaded->refine(1, .5);
	original->refine(1, .5);
	compare(*original, *loaded);
}

TEST_CASE("NC payload rejects corrupt hierarchy and topology", "[ncmesh][hdf5]")
{
	State initialize_geogram;
	const int d = GENERATE(2, 3);
	const int corruption = GENERATE(0, 1, 2, 3, 4, 5, 6);
	auto m = tests::generate_nc_mesh(d);
	auto data = m->to_mesh_data();
	auto &n = *data.nc;
	switch (corruption)
	{
	case 0:
		n.element_state(0, 1) = 0;
		break;
	case 1:
		n.children(0, 0) = n.cells.rows();
		break;
	case 2:
		n.midpoints(0, 2) = n.vertices.rows();
		break;
	case 3:
		n.cell_edges(0, 0) = n.edges.rows();
		break;
	case 4:
		data.vertices(0, 0) += .25;
		break;
	case 5:
		n.midpoints.conservativeResize(n.midpoints.rows() + 1, Eigen::NoChange);
		n.midpoints.bottomRows(1) = n.midpoints.topRows(1);
		break;
	case 6:
		std::swap(n.cell_edges(0, 0), n.cell_edges(0, 1));
		break;
	}
	CHECK_THROWS(mesh::Mesh::create(data));
}

TEST_CASE("NC input follows normal varform geometry loading and export", "[ncmesh][hdf5][varform]")
{
	State initialize_geogram;
	const int d = GENERATE(2, 3);
	Workspace workspace;
	auto original = tests::generate_nc_mesh(d);
	const auto path = workspace.path / "input.h5";
	auto config = solve_config(d, workspace.path);
	tests::write_nc_bundle(path, *original, config);
	auto input = io::load_hdf5_input(path);
	State state;
	state.init(input.config, *input.resources, true);
	state.load_mesh();
	Eigen::MatrixXd solution;
	state.solve(solution);
	const auto stats = state.variational_formulation->compute_errors(solution);
	CHECK(stats.l2_err < 1e-8);
	CHECK(stats.h1_semi_err < 1e-7);
	state.variational_formulation->export_data(solution);
	CHECK(std::filesystem::exists(workspace.path / "nc.vtu"));
	State memory;
	memory.init(config, *input.resources, true);
	memory.variational_formulation->set_mesh(std::move(original));
	Eigen::MatrixXd expected;
	memory.solve(expected);
	CHECK(solution.isApprox(expected, 1e-10));
	const auto actual_bases = test::VarFormTestAccess::debug_data(*state.variational_formulation).bases;
	const auto expected_bases = test::VarFormTestAccess::debug_data(*memory.variational_formulation).bases;
	REQUIRE(actual_bases->size() == expected_bases->size());
	bool has_hanging_weights = false;
	for (int e = 0; e < int(actual_bases->size()); ++e)
	{
		const auto &a = actual_bases->at(e).bases, &b = expected_bases->at(e).bases;
		REQUIRE(a.size() == b.size());
		for (int i = 0; i < int(a.size()); ++i)
		{
			const auto &u = a[i].global(), &v = b[i].global();
			REQUIRE(u.size() == v.size());
			has_hanging_weights |= u.size() > 1;
			for (int j = 0; j < int(u.size()); ++j)
			{
				CHECK(u[j].index == v[j].index);
				CHECK(std::abs(u[j].val - v[j].val) < 1e-12);
				CHECK(u[j].node.isApprox(v[j].node, 1e-12));
			}
		}
	}
	CHECK(has_hanging_weights);

	SECTION("Stored labels survive additional loader refinement")
	{
		config["geometry"][0]["n_refs"] = 1;
		State refined;
		refined.init(config, *input.resources, true);
		refined.load_mesh();
		Eigen::MatrixXd refined_solution;
		refined.solve(refined_solution);
		CHECK(refined.variational_formulation->compute_errors(refined_solution).l2_err < 1e-8);
	}
	SECTION("Geometry arrays preserve hierarchy")
	{
		config["geometry"][0]["type"] = "mesh_array";
		config["geometry"][0]["array"] = {{"size", d == 2 ? json{2, 1} : json{2, 1, 1}}, {"offset", 2}, {"relative", false}};
		State array;
		array.init(config, *input.resources, true);
		array.load_mesh();
		Eigen::MatrixXd array_solution;
		array.solve(array_solution);
		CHECK(array.variational_formulation->compute_errors(array_solution).l2_err < 1e-8);
	}
}

TEST_CASE("NC transient checkpoint resumes with stable basis ordering", "[ncmesh][hdf5][checkpoint][varform]")
{
	State initialize_geogram;
	const int d = GENERATE(2, 3);
	Workspace workspace;
	auto mesh = tests::generate_nc_mesh(d);
	auto config = solve_config(d, workspace.path);
	config["time"] = {{"dt", .1}, {"time_steps", 3}};
	config["output"]["advanced"]["save_time_sequence"] = false;
	config["output"]["checkpoint"]["path"] = (workspace.path / "checkpoint_{}.h5").string();
	const auto bundle = workspace.path / "input.h5";
	tests::write_nc_bundle(bundle, *mesh, config);
	auto input = io::load_hdf5_input(bundle);
	State uninterrupted;
	uninterrupted.init(config, *input.resources, true);
	uninterrupted.load_mesh();
	Eigen::MatrixXd expected;
	uninterrupted.solve(expected);
	const auto checkpoint_path = workspace.path / "checkpoint_1.h5";
	REQUIRE(std::filesystem::exists(checkpoint_path));
	io::CheckpointReader checkpoint(checkpoint_path);
	compare(*mesh, *checkpoint.read_mesh("/checkpoint/meshes/active"));
	State resumed;
	resumed.init(checkpoint, true);
	resumed.load_mesh();
	Eigen::MatrixXd actual;
	resumed.solve(actual);
	REQUIRE(actual.rows() == expected.rows());
	CHECK(actual.isApprox(expected, 1e-10));
}

TEST_CASE("NC loader promotes conforming simplex geometry components", "[ncmesh][hdf5][varform]")
{
	State initialize_geogram;
	const int d = GENERATE(2, 3);
	const bool nc_first = GENERATE(false, true);
	Workspace workspace;
	auto nc = tests::generate_nc_mesh(d), conforming = tests::nc_base_mesh(d);
	auto flat = conforming->Mesh::to_mesh_data();
	auto config = solve_config(d, workspace.path);
	json a = {{"mesh", "/meshes/nc"}};
	json b = {{"mesh", "/meshes/conforming"}, {"transformation", {{"translation", d == 2 ? json{2, 0} : json{2, 0, 0}}}}};
	config["geometry"] = nc_first ? json{a, b} : json{b, a};
	const auto path = workspace.path / "mixed.h5";
	{
		io::CheckpointMetadata metadata;
		metadata.dt = 1;
		io::CheckpointWriter writer(path, config, metadata);
		writer.write_mesh("/meshes/nc", *nc);
		writer.write_mesh("/meshes/conforming", flat);
		writer.finalize();
	}
	auto input = io::load_hdf5_input(path);
	CHECK(input.resources->read_integer_attribute("/meshes/conforming", "schema_version") == mesh::MESH_SCHEMA_VERSION);
	State state;
	state.init(config, *input.resources, true);
	state.load_mesh();
	Eigen::MatrixXd solution;
	state.solve(solution);
	CHECK(state.variational_formulation->compute_errors(solution).l2_err < 1e-8);
}

TEST_CASE("NC HDF5 loader rejects corrupted stored payloads", "[ncmesh][hdf5]")
{
	State initialize_geogram;
	const int d = GENERATE(2, 3);
	const int corruption = GENERATE(0, 1, 2, 3);
	Workspace workspace;
	auto mesh = tests::generate_nc_mesh(d);
	const auto path = workspace.path / "bad.h5";
	tests::write_nc_bundle(path, *mesh, json::object());
	{
		h5pp::File file(path, h5pp::FileAccess::READWRITE);
		switch (corruption)
		{
		case 0:
			file.writeAttribute(long(9), "/meshes/nc/nc", "schema_version");
			break;
		case 1:
		{
			auto state = mesh->to_mesh_data().nc->element_state;
			state(0, 1) = 0;
			file.writeDataset(state.cast<int64_t>().eval(), "/meshes/nc/nc/element_state");
			break;
		}
		case 2:
		{
			auto points = mesh->to_mesh_data().vertices;
			points(0, 0) += .2;
			file.writeDataset(points, "/meshes/nc/vertices");
			break;
		}
		case 3:
			file.writeAttribute(long(1), "/meshes/nc", "schema_version");
			break;
		}
	}
	io::HDF5IO resources(path);
	CHECK_THROWS(mesh::MeshLoader(resources).load_fem("/meshes/nc"));
}

TEST_CASE("Unrefined NC resources preserve empty refinement state", "[ncmesh][hdf5]")
{
	State initialize_geogram;
	const int d = GENERATE(2, 3);
	Workspace workspace;
	auto original = tests::generate_nc_mesh(d, 0, 0);
	original->set_geometry_ids({20, 21});
	original->prepare_mesh();
	const auto path = workspace.path / "roots.h5";
	tests::write_nc_bundle(path, *original, json::object());
	io::HDF5IO resources(path);
	auto loaded = mesh::MeshLoader(resources).load_fem("/meshes/nc");
	compare(*original, *loaded);
	original->refine(2, .5);
	loaded->refine(2, .5);
	original->prepare_mesh();
	loaded->prepare_mesh();
	compare(*original, *loaded);
}

TEST_CASE("NC conversion is explicit and creation follows mesh data", "[ncmesh][mesh_data]")
{
	State initialize_geogram;
	const int d = GENERATE(2, 3);
	auto base = tests::nc_base_mesh(d);
	auto conforming = mesh::Mesh::create(base->Mesh::to_mesh_data());
	REQUIRE(conforming->is_conforming());
	REQUIRE_FALSE(conforming->to_mesh_data().nc.has_value());
	auto nc = conforming->to_nonconforming();
	REQUIRE_FALSE(nc->is_conforming());
	CHECK(conforming->is_conforming());
	compare(*nc, *mesh::Mesh::create(nc->to_mesh_data()));
	compare(*nc, *nc->to_nonconforming());
	tests::nc_refine_elements(*nc, {0});
	compare(*nc, *mesh::Mesh::create(nc->to_mesh_data()));
}

TEST_CASE("NC conversion rejects non-simplex geometry", "[ncmesh][mesh_data]")
{
	State initialize_geogram;
	Eigen::MatrixXd vertices(4, 2);
	vertices << 0, 0, 1, 0, 1, 1, 0, 1;
	Eigen::MatrixXi cells(1, 4);
	cells << 0, 1, 2, 3;
	auto mesh = mesh::Mesh::create(mesh::MeshData(vertices, cells));
	CHECK_THROWS(mesh->to_nonconforming());
}
