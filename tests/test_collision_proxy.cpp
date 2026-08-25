#include <polyfem/mesh/collision_proxy/CollisionProxy.hpp>
#include <polyfem/mesh/collision_proxy/UpsampleMesh.hpp>
#include <polyfem/mesh/MeshUtils.hpp>

#include <polyfem/State.hpp>
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

	Eigen::MatrixXd vertices;
	Eigen::VectorXi _;
	Eigen::MatrixXi __, faces;
	polyfem::mesh::read_surface_mesh(proxy_mesh_path, vertices, _, __, faces);

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
