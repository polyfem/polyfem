#include <polyfem/mesh/collision_proxy/CollisionProxy.hpp>
#include <polyfem/mesh/collision_proxy/UpsampleMesh.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/mesh/MeshUtils.hpp>

#include <polyfem/io/ResourceIO.hpp>

#include <polyfem/State.hpp>
#include <polyfem/legacy/State.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include "VarFormTestAccess.hpp"

#include <catch2/catch_all.hpp>

#include <igl/readPLY.h>
#include <igl/writePLY.h>
#include <igl/boundary_facets.h>
#include <igl/doublearea.h>
#include <igl/euler_characteristic.h>
#include <igl/facet_components.h>
#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/point_mesh_squared_distance.h>

#include <ipc/ipc.hpp>

namespace
{
	// windows in release generates this error when building p4 bases
	// Unhandled exception at 0x00007FF62FDD8DD7 in unit_tests.exe: 0xC00000FD: Stack overflow (parameters: 0x0000000000000001, 0x00000087AAE09000).
#if defined(WIN32) && defined(NDEBUG)
	std::shared_ptr<polyfem::State> get_state(const std::string mesh_path = "", const int discr_order = 3)
#else
	std::shared_ptr<polyfem::State> get_state(const std::string mesh_path = "", const int discr_order = 4)
#endif
	{
		polyfem::json in_args;
		in_args["/materials/type"_json_pointer] = "NeoHookean";
		in_args["/materials/E"_json_pointer] = 1e5;
		in_args["/materials/nu"_json_pointer] = 0.3;
		in_args["/materials/rho"_json_pointer] = 1e3;
		in_args["/space/discr_order"_json_pointer] = discr_order;
		if (mesh_path == "")
		{
			const std::string path = POLYFEM_DATA_DIR;
			// in_args["/geometry/0/mesh"_json_pointer] = path + "/contact/meshes/3D/simple/tet/tet-corner.msh";

#if defined(WIN32) && defined(NDEBUG)
			in_args["/geometry/0/mesh"_json_pointer] = path + "/contact/meshes/3D/simple/cube.msh";
#else
			in_args["/geometry/0/mesh"_json_pointer] = path + "/contact/meshes/3D/simple/sphere/coarse/P4.msh";
#endif
			// in_args["/geometry/0/mesh"_json_pointer] = path + "/contact/meshes/3D/creatures/armadillo/ArmadilloP4.msh";
			// in_args["/geometry/0/mesh"_json_pointer] = path + "/contact/meshes/3D/microstructure/P4.msh";
		}
		else
		{
			in_args["/geometry/0/mesh"_json_pointer] = mesh_path;
		}
		in_args["/time/time_steps"_json_pointer] = 1;
		in_args["/time/tend"_json_pointer] = 1;
		in_args["/output/log/level"_json_pointer] = "warning";

		std::shared_ptr<polyfem::State> state = std::make_shared<polyfem::State>();
		state->init(in_args, true);
		state->set_max_threads(1);

		state->load_mesh();

		polyfem::test::VarFormTestAccess::prepare(*state->variational_formulation);

		return state;
	}

	void build_mesh_matrices(const polyfem::test::VarFormDebugData &debug, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
	{
		REQUIRE(debug.mesh != nullptr);
		REQUIRE(debug.bases != nullptr);
		const size_t n_vertices = debug.n_bases - debug.n_obstacle_vertices;
		const int dim = debug.mesh->dimension();

		V.resize(n_vertices, dim);
		F.resize(debug.bases->size(), dim + 1);

		for (int i = 0; i < debug.bases->size(); i++)
		{
			const polyfem::basis::ElementBases &element = (*debug.bases)[i];
			for (int j = 0; j < element.bases.size(); j++)
			{
				const polyfem::basis::Basis &basis = element.bases[j];
				REQUIRE(basis.global().size() == 1);
				V.row(basis.global()[0].index) = basis.global()[0].node;
				if (j < F.cols())
					F(i, j) = basis.global()[0].index;
			}
		}
	}

	void check_closed_surface_mesh(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const int expected_euler_characteristic,
		const int expected_components)
	{
		REQUIRE(V.rows() > 0);
		REQUIRE(V.cols() == 3);
		REQUIRE(F.rows() > 0);
		REQUIRE(F.cols() == 3);
		REQUIRE(V.allFinite());
		REQUIRE(F.minCoeff() >= 0);
		REQUIRE(F.maxCoeff() < V.rows());

		Eigen::VectorXd double_area;
		igl::doublearea(V, F, double_area);
		CHECK(double_area.minCoeff() > 0);
		CHECK(igl::is_edge_manifold(F));
		CHECK(igl::is_vertex_manifold(F));

		Eigen::MatrixXi boundary_edges;
		igl::boundary_facets(F, boundary_edges);
		CHECK(boundary_edges.rows() == 0);
		CHECK(igl::euler_characteristic(F) == expected_euler_characteristic);

		Eigen::VectorXi components;
		CHECK(igl::facet_components(F, components) == expected_components);
	}
} // namespace

TEST_CASE("upsample mesh", "[upsample_mesh]")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	REQUIRE(igl::readPLY(std::string(POLYFEM_DATA_DIR) + "/octocat-coarse.ply", V, F));

	const double max_edge_length = 0.1 * polyfem::mesh::max_edge_length(V, F);

	Eigen::MatrixXd V_grid;
	Eigen::MatrixXi F_grid;
	polyfem::mesh::regular_grid_tessellation(V, F, max_edge_length, V_grid, F_grid);

	CHECK(V_grid.rows() == 126802);
	CHECK(F_grid.rows() == 253600);

	// REQUIRE(igl::writePLY("octocat-regular-tessellation.ply", V_grid, F_grid));

#ifdef POLYFEM_WITH_TRIANGLE
	Eigen::MatrixXd V_irregular;
	Eigen::MatrixXi F_irregular;
	polyfem::mesh::irregular_tessellation(V, F, max_edge_length, V_irregular, F_irregular);

	check_closed_surface_mesh(V_irregular, F_irregular, /*euler=*/2, /*components=*/1);

	Eigen::VectorXd input_double_area, output_double_area;
	igl::doublearea(V, F, input_double_area);
	igl::doublearea(V_irregular, F_irregular, output_double_area);
	CHECK(output_double_area.sum() == Catch::Approx(input_double_area.sum()).epsilon(1e-10));

	Eigen::VectorXd squared_distance;
	Eigen::VectorXi closest_face;
	Eigen::MatrixXd closest_point;
	igl::point_mesh_squared_distance(
		V_irregular, V, F, squared_distance, closest_face, closest_point);
	constexpr double stitch_tolerance = 1e-5;
	CHECK(squared_distance.maxCoeff() < stitch_tolerance * stitch_tolerance);

	const double p = 1.5 * max_edge_length;
	const double triangle_max_area = std::sqrt(p * std::pow(p - max_edge_length, 3));
	CHECK(0.5 * output_double_area.maxCoeff() <= triangle_max_area + 5e-7);

	// REQUIRE(igl::writePLY("octocat-irregular-tessellation.ply", V_irregular, F_irregular));
#endif
}

TEST_CASE("build collision proxy", "[build_collision_proxy]")
{
	using namespace polyfem::mesh;
	const CollisionProxyTessellation tessellation =
		GENERATE(CollisionProxyTessellation::REGULAR, CollisionProxyTessellation::IRREGULAR);

#ifndef POLYFEM_WITH_TRIANGLE
	if (tessellation == CollisionProxyTessellation::IRREGULAR)
		return;
#endif

	const auto state = get_state();
	const polyfem::test::VarFormDebugData debug =
		polyfem::test::VarFormTestAccess::debug_data(*state->variational_formulation);
	REQUIRE(debug.mesh != nullptr);
	REQUIRE(debug.bases != nullptr);
	REQUIRE(debug.geometry_bases != nullptr);
	REQUIRE(debug.total_local_boundary != nullptr);

	Eigen::MatrixXd proxy_vertices;
	Eigen::MatrixXi proxy_faces;
	std::vector<Eigen::Triplet<double>> displacement_map_entries;
	build_collision_proxy(
		*debug.bases, *debug.geometry_bases, *debug.total_local_boundary, debug.n_bases, debug.mesh->dimension(),
		/*max_edge_length=*/0.1, proxy_vertices, proxy_faces, displacement_map_entries, tessellation);

#if defined(WIN32) && defined(NDEBUG)
	CHECK(proxy_vertices.rows() == 488);
	CHECK(proxy_faces.rows() == 972);
#else
	if (tessellation == CollisionProxyTessellation::REGULAR)
	{
		CHECK(proxy_vertices.rows() == 1217);
		CHECK(proxy_faces.rows() == 2430);
		// REQUIRE(igl::writePLY("proxy-regular.ply", proxy_vertices, proxy_faces));
	}
#endif
	Eigen::MatrixXd V;
	Eigen::MatrixXi F, T;
	build_mesh_matrices(debug, V, T);
	igl::boundary_facets(T, F);
	check_closed_surface_mesh(proxy_vertices, proxy_faces, /*euler=*/2, /*components=*/1);

	Eigen::MatrixXd squished_V = V;
	squished_V.col(1) *= 0.1;

	const Eigen::MatrixXd U = squished_V - V;

	// REQUIRE(igl::writePLY("fem.ply", V, F));
	// REQUIRE(igl::writePLY("deformed_fem.ply", V + U, F));

	Eigen::SparseMatrix<double> W(proxy_vertices.rows(), V.rows());
	W.setFromTriplets(displacement_map_entries.begin(), displacement_map_entries.end());
	CHECK((W * Eigen::VectorXd::Ones(V.rows()) - Eigen::VectorXd::Ones(proxy_vertices.rows())).lpNorm<Eigen::Infinity>() < 1e-12);
	const Eigen::MatrixXd U_proxy = W * U;
	Eigen::MatrixXd expected_squished_proxy = proxy_vertices;
	expected_squished_proxy.col(1) *= 0.1;
	CHECK((proxy_vertices + U_proxy - expected_squished_proxy).cwiseAbs().maxCoeff() < 1e-12);

	// REQUIRE(igl::writePLY("deformed_proxy.ply", proxy_vertices + U_proxy, proxy_faces));
}

TEST_CASE("build collision proxy displacement map", "[build_collision_proxy]")
{
#if defined(WIN32) && defined(NDEBUG)
	const int discr_order = GENERATE(1, 2, 3);
#else
	const int discr_order = GENERATE(1, 2, 3, 4);
#endif

	const int n_nodes_per_element = (std::array<int, 4>{{4, 10, 20, 35}})[discr_order - 1];

	const std::string path = POLYFEM_DATA_DIR;
	std::string fe_mesh_path, proxy_mesh_path;
	SECTION("sphere-to-cube")
	{
		fe_mesh_path = path + "/contact/meshes/3D/simple/cube.msh";
		proxy_mesh_path = path + "/contact/meshes/3D/simple/sphere/sphere5K.msh";
	}
	// SECTION("cube-to-sphere")
	// {
	// 	fe_mesh_path = path + "/contact/meshes/3D/simple/sphere/sphere5K.msh";
	// 	proxy_mesh_path = path + "/contact/meshes/3D/simple/cube.msh";
	// }

	const auto state = get_state(fe_mesh_path, discr_order);
	const polyfem::test::VarFormDebugData debug =
		polyfem::test::VarFormTestAccess::debug_data(*state->variational_formulation);
	REQUIRE(debug.mesh != nullptr);
	REQUIRE(debug.bases != nullptr);
	REQUIRE(debug.geometry_bases != nullptr);
	REQUIRE(debug.total_local_boundary != nullptr);

	const polyfem::io::FileSystemIO resources(".");
	const polyfem::mesh::SurfaceMesh surface =
		polyfem::mesh::MeshLoader(resources).load_surface(proxy_mesh_path);
	const Eigen::MatrixXd &vertices = surface.vertices;

	std::vector<Eigen::Triplet<double>> displacement_map_entries;
	polyfem::mesh::build_collision_proxy_displacement_map(
		*debug.bases, *debug.geometry_bases, *debug.total_local_boundary,
		debug.n_bases, debug.mesh->dimension(), vertices,
		displacement_map_entries);

	CHECK(displacement_map_entries.size() == vertices.rows() * n_nodes_per_element);
}

TEST_CASE("spline contact builds a displacement map", "[build_collision_proxy]")
{
	polyfem::json in_args;
	in_args["/geometry/0/mesh"_json_pointer] = std::string(POLYFEM_DATA_DIR) + "/quad_test/hex.HYBRID";
	in_args["/materials/type"_json_pointer] = "NeoHookean";
	in_args["/materials/E"_json_pointer] = 1e5;
	in_args["/materials/nu"_json_pointer] = 0.3;
	in_args["/materials/rho"_json_pointer] = 1e3;
	in_args["/space/basis_type"_json_pointer] = "Spline";
	in_args["/contact/enabled"_json_pointer] = true;
	in_args["/time/time_steps"_json_pointer] = 1;
	in_args["/time/tend"_json_pointer] = 1;
	in_args["/output/log/level"_json_pointer] = "warning";

	polyfem::State state;
	state.init(in_args, true);
	state.set_max_threads(1);
	state.load_mesh();
	polyfem::test::VarFormTestAccess::prepare(*state.variational_formulation);

	const polyfem::test::VarFormDebugData debug =
		polyfem::test::VarFormTestAccess::debug_data(*state.variational_formulation);
	const polyfem::io::OutputSpace output_space = state.variational_formulation->output_space();
	REQUIRE(output_space.collision_mesh != nullptr);
	CHECK(output_space.collision_mesh->num_faces() > 0);
	CHECK_FALSE(ipc::has_intersections(
		*output_space.collision_mesh, output_space.collision_mesh->rest_positions()));

	const Eigen::MatrixXd mapped_displacements = output_space.collision_mesh->map_displacements(
		Eigen::MatrixXd::Zero(debug.n_bases, debug.mesh->dimension()));
	CHECK(mapped_displacements.rows() == output_space.collision_mesh->rest_positions().rows());
}

TEST_CASE("Bernstein 2D collision mesh evaluates Lagrange samples", "[collision_proxy][bernstein]")
{
	polyfem::json args = {
		{"geometry", {{{"mesh", ""}}}},
		{"materials", {{"type", "NeoHookean"}, {"E", 1e5}, {"nu", 0.3}, {"rho", 1e3}}},
		{"space", {{"discr_order", 3}, {"basis_type", "Bernstein"}}},
		{"contact", {{"enabled", true}, {"dhat", 0.01}}},
		{"output", {{"log", {{"level", "warning"}}}}}};
	const int sampling_order = GENERATE(-1, 0, 1, 6, 10);
	if (sampling_order >= 0)
		args["contact"]["collision_mesh"] = {{"enabled", true}, {"tessellation_type", "max_order"}, {"sampling_order", sampling_order}};
	const int segments = sampling_order > 0 ? sampling_order : 3;
	polyfem::State state;
	state.init(args, true);
	Eigen::MatrixXd V(3, 2);
	V << 0, 0, 1, 0, 0, 1;
	Eigen::MatrixXi F(1, 3);
	F << 0, 1, 2;
	state.set_mesh(V, F);
	polyfem::test::VarFormTestAccess::prepare(*state.variational_formulation);
	const auto debug = polyfem::test::VarFormTestAccess::debug_data(*state.variational_formulation);
	const auto output = state.variational_formulation->output_space();
	REQUIRE(output.collision_mesh != nullptr);
	const auto &collision = *output.collision_mesh;
	REQUIRE(collision.num_vertices() == 3 * segments);
	CHECK(collision.num_edges() == 3 * segments);
	CHECK(collision.num_faces() == 0);
	CHECK_FALSE(ipc::has_intersections(collision, collision.rest_positions()));

	// Perturb only the Bernstein coefficient at (1/3, 0). Along the bottom
	// edge its cubic basis is 3*t*(1-t)^2, not a nodal hat function.
	Eigen::MatrixXd u = Eigen::MatrixXd::Zero(debug.n_bases, 2);
	int control = -1;
	for (const auto &b : debug.bases->front().bases)
		if ((b.global().front().node.head<2>() - Eigen::RowVector2d(1.0 / 3, 0)).norm() < 1e-12)
			control = b.global().front().index;
	REQUIRE(control >= 0);
	u(control, 1) = 1;
	const Eigen::MatrixXd displaced = collision.displace_vertices(u);
	int bottom_samples = 0;
	for (int i = 0; i < collision.num_vertices(); ++i)
	{
		const double x = collision.rest_positions()(i, 0);
		const double y = collision.rest_positions()(i, 1);
		if (std::abs(y) < 1e-12)
		{
			++bottom_samples;
			CHECK(displaced(i, 0) == Catch::Approx(x).margin(1e-12));
			CHECK(displaced(i, 1) == Catch::Approx(3 * x * (1 - x) * (1 - x)).margin(1e-12));
		}
	}
	CHECK(bottom_samples == segments + 1);

	// Partition of unity: a uniform coefficient displacement translates
	// every sampled collision vertex by exactly the same amount.
	u.col(0).setConstant(0.2);
	u.col(1).setConstant(-0.3);
	const Eigen::MatrixXd translated = collision.displace_vertices(u);
	for (int i = 0; i < collision.num_vertices(); ++i)
		CHECK((translated.row(i) - collision.rest_positions().row(i) - Eigen::RowVector2d(0.2, -0.3)).norm() < 1e-12);
}

TEST_CASE("2D collision sampling supports weighted corners and empty bases", "[collision_proxy][sampling_regression]")
{
	polyfem::json args = {
		{"geometry", {{{"mesh", ""}}}},
		{"materials", {{"type", "NeoHookean"}, {"E", 1e5}, {"nu", 0.3}, {"rho", 1e3}}},
		{"space", {{"discr_order", 1}}},
		{"output", {{"log", {{"level", "warning"}}}}}};
	polyfem::State state;
	state.init(args, true);
	Eigen::MatrixXd V(3, 2);
	V << 0, 0, 1, 0, 0, 1;
	Eigen::MatrixXi F(1, 3);
	F << 0, 1, 2;
	state.set_mesh(V, F);
	polyfem::test::VarFormTestAccess::prepare(*state.variational_formulation);
	const auto debug = polyfem::test::VarFormTestAccess::debug_data(*state.variational_formulation);
	auto bases = *debug.bases;
	Eigen::MatrixXd vertices;
	Eigen::MatrixXi edges, faces;
	std::vector<Eigen::Triplet<double>> entries;

	SECTION("weighted corner")
	{
		// Replace one corner DOF by a weighted combination of two DOFs.
		auto &corner = bases.front().bases.front().global();
		corner = {bases.front().bases[1].global().front(), bases.front().bases[2].global().front()};
		corner[0].val = 0.25;
		corner[1].val = 0.75;
		polyfem::io::OutGeometryData::extract_boundary_mesh_2d_sampled(
			*debug.mesh, debug.n_bases, bases, *debug.geometry_bases, *debug.total_local_boundary,
			vertices, edges, faces, entries, 3);
		REQUIRE(vertices.rows() == 9);
		CHECK(edges.rows() == 9);
		Eigen::SparseMatrix<double> map(vertices.rows(), debug.n_bases);
		map.setFromTriplets(entries.begin(), entries.end());
		for (int i = 0; i < vertices.rows(); ++i)
		{
			CHECK(map.row(i).sum() == Catch::Approx(1));
			if (vertices.row(i).norm() < 1e-12)
			{
				CHECK(map.coeff(i, corner[0].index) == Catch::Approx(0.25));
				CHECK(map.coeff(i, corner[1].index) == Catch::Approx(0.75));
			}
		}
	}
	SECTION("empty boundary element")
	{
		const int order = GENERATE(0, 3);
		bases.front().bases.clear();
		REQUIRE_NOTHROW(polyfem::io::OutGeometryData::extract_boundary_mesh_2d_sampled(
			*debug.mesh, debug.n_bases, bases, *debug.geometry_bases, *debug.total_local_boundary,
			vertices, edges, faces, entries, order));
		CHECK(edges.rows() == 0);
		CHECK(entries.empty());
	}
}

TEST_CASE("2D polygon collision sampling uses standard extraction", "[collision_proxy][sampling_regression]")
{
	Eigen::MatrixXd V(5, 2);
	V << 0, 0, 1, 0, 2, 1, 1, 2, 0, 1;
	Eigen::MatrixXi F(1, 5);
	F << 0, 1, 2, 3, 4;
	const auto mesh = polyfem::mesh::Mesh::create(polyfem::mesh::MeshData(V, F));
	REQUIRE(mesh->has_poly());
	std::vector<polyfem::basis::ElementBases> bases(1);
	for (int i = 0; i < 5; ++i)
	{
		bases.front().bases.emplace_back();
		bases.front().bases.back().init(1, i, i, V.row(i));
	}
	bases.front().set_local_node_from_primitive_func([](const int edge, const polyfem::mesh::Mesh &mesh) {
		Eigen::VectorXi nodes(2);
		nodes << mesh.edge_vertex(edge, 0), mesh.edge_vertex(edge, 1);
		return nodes;
	});
	std::vector<polyfem::mesh::LocalBoundary> boundary;
	boundary.emplace_back(0, polyfem::mesh::BoundaryType::POLYGON);
	for (int e = 0; e < mesh->n_edges(); ++e)
		boundary.front().add_boundary_primitive(e, e);
	Eigen::MatrixXd vertices;
	Eigen::MatrixXi edges, faces;
	std::vector<Eigen::Triplet<double>> entries;
	const int order = GENERATE(0, 3);
	REQUIRE_NOTHROW(polyfem::io::OutGeometryData::extract_boundary_mesh_2d_sampled(
		*mesh, 5, bases, {}, boundary, vertices, edges, faces, entries, order));
	CHECK(vertices.isApprox(V));
	CHECK(edges.rows() == 5);
	CHECK(faces.size() == 0);
	CHECK(entries.empty());
}

TEST_CASE("Bernstein contact falls back on nonconforming volume meshes", "[collision_proxy][sampling_regression]")
{
	polyfem::json args = {
		{"geometry", {{{"mesh", ""}}}},
		{"materials", {{"type", "NeoHookean"}, {"E", 1e5}, {"nu", 0.3}, {"rho", 1e3}}},
		{"space", {{"discr_order", 1}, {"basis_type", "Bernstein"}}},
		{"contact", {{"enabled", true}, {"dhat", 0.01}}},
		{"output", {{"log", {{"level", "warning"}}}}}};
	polyfem::State state;
	state.init(args, true);
	Eigen::MatrixXd V(4, 3);
	V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
	Eigen::MatrixXi F(1, 4);
	F << 0, 1, 2, 3;
	state.set_mesh(V, F, true);
	REQUIRE_NOTHROW(polyfem::test::VarFormTestAccess::prepare(*state.variational_formulation));
	const auto output = state.variational_formulation->output_space();
	REQUIRE(output.collision_mesh != nullptr);
	CHECK(output.collision_mesh->num_vertices() == 4);
	CHECK(output.collision_mesh->num_faces() == 4);
}

TEST_CASE("2D spline max-order collision sampling preserves standard extraction", "[collision_proxy][sampling_regression]")
{
	polyfem::json args = {
		{"geometry", {{{"mesh", ""}}}},
		{"materials", {{"type", "NeoHookean"}, {"E", 1e5}, {"nu", 0.3}, {"rho", 1e3}}},
		{"space", {{"basis_type", "Spline"}}},
		{"contact", {{"enabled", true}, {"dhat", 0.01},
			{"collision_mesh", {{"enabled", true}, {"tessellation_type", "max_order"}}}}},
		{"output", {{"log", {{"level", "warning"}}}}}};
	polyfem::State state;
	state.init(args, true);
	Eigen::MatrixXd V(16, 2);
	for (int y = 0; y < 4; ++y)
		for (int x = 0; x < 4; ++x)
			V.row(4 * y + x) << x, y;
	Eigen::MatrixXi F(9, 4);
	for (int y = 0; y < 3; ++y)
		for (int x = 0; x < 3; ++x)
			F.row(3 * y + x) << 4 * y + x, 4 * y + x + 1, 4 * (y + 1) + x + 1, 4 * (y + 1) + x;
	state.set_mesh(V, F);
	polyfem::test::VarFormTestAccess::prepare(*state.variational_formulation);
	const auto output = state.variational_formulation->output_space();
	REQUIRE(output.collision_mesh != nullptr);
	const auto &collision = *output.collision_mesh;
	double perimeter = 0;
	for (int i = 0; i < collision.num_edges(); ++i)
		perimeter += (collision.rest_positions().row(collision.edges()(i, 0))
			- collision.rest_positions().row(collision.edges()(i, 1))).norm();
	const auto debug = polyfem::test::VarFormTestAccess::debug_data(*state.variational_formulation);
	Eigen::MatrixXd expected_vertices;
	Eigen::MatrixXi expected_edges, expected_faces;
	std::vector<Eigen::Triplet<double>> entries;
	polyfem::io::OutGeometryData::extract_boundary_mesh(
		*debug.mesh, debug.n_bases, *debug.bases, *debug.total_local_boundary,
		expected_vertices, expected_edges, expected_faces, entries);
	double expected_perimeter = 0;
	for (int i = 0; i < expected_edges.rows(); ++i)
		expected_perimeter += (expected_vertices.row(expected_edges(i, 0))
			- expected_vertices.row(expected_edges(i, 1))).norm();
	CHECK(collision.num_edges() == expected_edges.rows());
	CHECK(perimeter == Catch::Approx(expected_perimeter));
}

TEST_CASE("Bernstein 3D collision mesh evaluates a watertight sampled surface", "[collision_proxy][bernstein][3d]")
{
	const int sampling_order = GENERATE(-1, 0, 1, 6);
	const int order = sampling_order > 0 ? sampling_order : 3;
	polyfem::json args = {
		{"geometry", {{{"mesh", ""}}}},
		{"materials", {{"type", "NeoHookean"}, {"E", 1e5}, {"nu", 0.3}, {"rho", 1e3}}},
		{"space", {{"discr_order", 3}, {"basis_type", "Bernstein"}}},
		{"contact", {{"enabled", true}, {"dhat", 0.01}}},
		{"output", {{"log", {{"level", "warning"}}}}}};
	if (sampling_order >= 0)
		args["contact"]["collision_mesh"] = {{"enabled", true}, {"tessellation_type", "max_order"}, {"sampling_order", sampling_order}};
	polyfem::State state;
	state.init(args, true);
	state.set_max_threads(1);
	Eigen::MatrixXd V(4, 3);
	V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
	Eigen::MatrixXi T(1, 4);
	T << 0, 1, 2, 3;
	state.set_mesh(V, T);
	polyfem::test::VarFormTestAccess::prepare(*state.variational_formulation);
	const auto debug = polyfem::test::VarFormTestAccess::debug_data(*state.variational_formulation);
	const auto output = state.variational_formulation->output_space();
	REQUIRE(output.collision_mesh != nullptr);
	const auto &collision = *output.collision_mesh;
	CHECK(collision.num_vertices() == 2 * order * order + 2);
	CHECK(collision.num_faces() == 4 * order * order);
	check_closed_surface_mesh(collision.rest_positions(), collision.faces(), 2, 1);
	CHECK_FALSE(ipc::has_intersections(collision, collision.rest_positions()));

	// A single cubic Bernstein edge coefficient has the analytic value
	// 3*x*(1-x-y-z)^2 on the reference tetrahedron.
	Eigen::MatrixXd u = Eigen::MatrixXd::Zero(debug.n_bases, 3);
	int control = -1;
	for (const auto &b : debug.bases->front().bases)
		if ((b.global().front().node - Eigen::RowVector3d(1.0 / 3, 0, 0)).norm() < 1e-12)
			control = b.global().front().index;
	REQUIRE(control >= 0);
	u(control, 2) = 1;
	const Eigen::MatrixXd displaced = collision.displace_vertices(u);
	for (int i = 0; i < collision.num_vertices(); ++i)
	{
		const Eigen::RowVector3d p = collision.rest_positions().row(i);
		const double l0 = 1 - p.sum();
		CHECK((displaced.row(i) - p - Eigen::RowVector3d(0, 0, 3 * p(0) * l0 * l0)).norm() < 1e-12);
	}
	u.rowwise() = Eigen::RowVector3d(0.2, -0.3, 0.4);
	const Eigen::MatrixXd translated = collision.displace_vertices(u);
	for (int i = 0; i < collision.num_vertices(); ++i)
		CHECK((translated.row(i) - collision.rest_positions().row(i) - Eigen::RowVector3d(0.2, -0.3, 0.4)).norm() < 1e-12);

	// Give the independent geometry a curved mapping x -> x + 0.2*x*y.
	// FE displacements must still be evaluated in the original reference domain.
	auto geometry = *debug.geometry_bases;
	const auto original = geometry.front();
	geometry.front().set_bases_func([original](const Eigen::MatrixXd &points, std::vector<polyfem::assembler::AssemblyValues> &values) {
		original.evaluate_bases(points, values);
		Eigen::MatrixXd physical_points;
		original.eval_geom_mapping(points, physical_points);
		for (size_t k = 0; k < original.bases.size(); ++k)
		{
			const auto &p = original.bases[k].global().front().node;
			const double sign = p.norm() < 1e-12 ? -1 : (p - Eigen::RowVector3d(1, 0, 0)).norm() < 1e-12 ? 1 : 0;
			values[k].val.col(0).array() += sign * 0.2 * physical_points.col(0).array() * physical_points.col(1).array();
		}
	});
	Eigen::MatrixXd sampled;
	Eigen::MatrixXi edges, faces;
	std::vector<Eigen::Triplet<double>> entries;
	polyfem::io::OutGeometryData::extract_boundary_mesh_sampled(
		*debug.mesh, debug.n_bases, *debug.bases, *debug.total_local_boundary,
		sampled, edges, faces, entries, order, &geometry);
	REQUIRE(sampled.rows() == collision.num_vertices());
	for (int i = 0; i < sampled.rows(); ++i)
	{
		Eigen::RowVector3d expected = collision.rest_positions().row(i);
		expected(0) += 0.2 * expected(0) * expected(1);
		CHECK((sampled.row(i) - expected).norm() < 1e-12);
	}
}

TEST_CASE("collision mesh marks obstacle vertices", "[collision_mesh][obstacle]")
{
	const std::string path = std::string(POLYFEM_DATA_DIR) + "/contact/meshes/2D";
	polyfem::json in_args;
	in_args["geometry"] = polyfem::json::array({
		{{"mesh", path + "/simple/triangle/right.obj"},
		 {"transformation", {{"translation", {0, 0.6}}}}},
		{{"mesh", path + "/obstacles/line.obj"},
		 {"is_obstacle", true},
		 {"transformation", {{"translation", {0.5, -0.5}}}}},
	});
	in_args["/materials/type"_json_pointer] = "NeoHookean";
	in_args["/materials/E"_json_pointer] = 1e5;
	in_args["/materials/nu"_json_pointer] = 0.3;
	in_args["/materials/rho"_json_pointer] = 1e3;
	in_args["/contact/enabled"_json_pointer] = true;
	in_args["/time/time_steps"_json_pointer] = 1;
	in_args["/time/tend"_json_pointer] = 1;
	in_args["/output/log/level"_json_pointer] = "warning";

	// The obstacle's vertices are appended after the FE vertices.
	const auto check_obstacle_flags = [](const ipc::CollisionMesh &mesh, const int n_obstacle_vertices) {
		REQUIRE(n_obstacle_vertices > 0);
		int n_marked = 0;
		for (int i = 0; i < mesh.num_vertices(); i++)
		{
			const bool is_obstacle = mesh.to_full_vertex_id(i) >= mesh.full_num_vertices() - n_obstacle_vertices;
			CHECK(mesh.is_obstacle_vertex(i) == is_obstacle);
			n_marked += mesh.is_obstacle_vertex(i);
		}
		CHECK(n_marked == n_obstacle_vertices);
	};

	SECTION("VarForm")
	{
		polyfem::State state;
		state.init(in_args, true);
		state.set_max_threads(1);
		state.load_mesh();
		polyfem::test::VarFormTestAccess::prepare(*state.variational_formulation);

		const polyfem::io::OutputSpace output_space = state.variational_formulation->output_space();
		REQUIRE(output_space.collision_mesh != nullptr);
		REQUIRE(output_space.obstacle != nullptr);
		check_obstacle_flags(*output_space.collision_mesh, output_space.obstacle->n_vertices());
	}

	SECTION("legacy State")
	{
		polyfem::legacy::State state;
		state.init(in_args, true);
		state.set_max_threads(1);
		state.load_mesh();
		state.build_basis();
		check_obstacle_flags(state.collision_mesh, state.obstacle.n_vertices());
	}
}
