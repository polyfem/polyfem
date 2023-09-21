#include <polyfem/mesh/collision_proxy/CollisionProxy.hpp>
#include <polyfem/mesh/collision_proxy/UpsampleMesh.hpp>
#include <polyfem/mesh/MeshUtils.hpp>

#include <polyfem/State.hpp>

#include <catch2/catch_all.hpp>

#include <igl/readPLY.h>
#include <igl/writePLY.h>
#include <igl/boundary_facets.h>

namespace
{
	std::shared_ptr<polyfem::State> get_state(const std::string mesh_path = "", const int discr_order = 4)
	{
		json in_args;
		in_args["/materials/type"_json_pointer] = "NeoHookean";
		in_args["/materials/E"_json_pointer] = 1e5;
		in_args["/materials/nu"_json_pointer] = 0.3;
		in_args["/materials/rho"_json_pointer] = 1e3;
		in_args["/space/discr_order"_json_pointer] = discr_order;
		if (mesh_path == "")
		{
			const std::string path = POLYFEM_DATA_DIR;
			// in_args["/geometry/0/mesh"_json_pointer] = path + "/contact/meshes/3D/simple/tet/tet-corner.msh";
			// in_args["/geometry/0/mesh"_json_pointer] = path + "/contact/meshes/3D/simple/cube.msh";
			in_args["/geometry/0/mesh"_json_pointer] = path + "/contact/meshes/3D/simple/sphere/coarse/P4.msh";
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

		state->build_basis();
		// state->assemble_rhs();
		// state->assemble_mass_mat();

		return state;
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

	CHECK(V_irregular.rows() == 12645);
	CHECK(F_irregular.rows() == 25286);

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

	Eigen::MatrixXd proxy_vertices;
	Eigen::MatrixXi proxy_faces;
	std::vector<Eigen::Triplet<double>> displacement_map_entries;
	build_collision_proxy(
		state->bases, state->geom_bases(), state->total_local_boundary, state->n_bases, state->mesh->dimension(),
		/*max_edge_length=*/0.1, proxy_vertices, proxy_faces, displacement_map_entries, tessellation);

	if (tessellation == CollisionProxyTessellation::REGULAR)
	{
		CHECK(proxy_vertices.rows() == 1217);
		CHECK(proxy_faces.rows() == 2430);
		// REQUIRE(igl::writePLY("proxy-regular.ply", proxy_vertices, proxy_faces));
	}
	else if (tessellation == CollisionProxyTessellation::IRREGULAR)
	{
		CHECK(proxy_vertices.rows() == 1801);
		CHECK(proxy_faces.rows() == 3598);
		// REQUIRE(igl::writePLY("proxy-irregular.ply", proxy_vertices, proxy_faces));
	}

	Eigen::MatrixXd V;
	Eigen::MatrixXi F, T;
	state->build_mesh_matrices(V, T);
	igl::boundary_facets(T, F);

	Eigen::MatrixXd squished_V = V;
	squished_V.col(1) *= 0.1;

	const Eigen::MatrixXd U = squished_V - V;

	// REQUIRE(igl::writePLY("fem.ply", V, F));
	// REQUIRE(igl::writePLY("deformed_fem.ply", V + U, F));

	Eigen::SparseMatrix<double> W(proxy_vertices.rows(), V.rows());
	W.setFromTriplets(displacement_map_entries.begin(), displacement_map_entries.end());
	const Eigen::MatrixXd U_proxy = W * U;

	// REQUIRE(igl::writePLY("deformed_proxy.ply", proxy_vertices + U_proxy, proxy_faces));
}

TEST_CASE("build collision proxy displacement map", "[build_collision_proxy]")
{
	const int discr_order = GENERATE(1, 2, 3, 4);
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

	Eigen::MatrixXd vertices;
	Eigen::VectorXi _;
	Eigen::MatrixXi __, faces;
	polyfem::mesh::read_surface_mesh(proxy_mesh_path, vertices, _, __, faces);

	std::vector<Eigen::Triplet<double>> displacement_map_entries;
	polyfem::mesh::build_collision_proxy_displacement_maps(
		state->bases, state->geom_bases(), state->total_local_boundary,
		state->n_bases, state->mesh->dimension(), vertices,
		displacement_map_entries);

	CHECK(displacement_map_entries.size() == vertices.rows() * n_nodes_per_element);
}