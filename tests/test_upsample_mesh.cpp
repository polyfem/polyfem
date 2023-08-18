#include <polyfem/mesh/collision_proxy/UpsampleMesh.hpp>

#include <catch2/catch.hpp>

#include <igl/readPLY.h>
#include <igl/writePLY.h>

TEST_CASE("upsample mesh", "[upsample_mesh]")
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	REQUIRE(igl::readPLY("/Users/zachary/Desktop/octocat-coarse.ply", V, F));

	const double max_edge_length = 0.1 * polyfem::mesh::max_edge_length(V, F);

	Eigen::MatrixXd V_grid;
	Eigen::MatrixXi F_grid;
	polyfem::mesh::regular_grid_tessilation(V, F, max_edge_length, V_grid, F_grid);

	REQUIRE(igl::writePLY("/Users/zachary/Desktop/octocat-regular-tessilation.ply", V_grid, F_grid));

#ifdef POLYFEM_WITH_TRIANGLE
	Eigen::MatrixXd V_irregular;
	Eigen::MatrixXi F_irregular;
	polyfem::mesh::irregular_tessilation(V, F, max_edge_length, V_irregular, F_irregular);

	REQUIRE(igl::writePLY("/Users/zachary/Desktop/octocat-irregular-tessilation.ply", V_irregular, F_irregular));
#endif
}