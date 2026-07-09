////////////////////////////////////////////////////////////////////////////////
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/State.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::mesh;

namespace
{
	std::unique_ptr<Mesh> create_test_triangle_mesh()
	{
		Eigen::MatrixXd vertices(3, 2);
		vertices << 0, 0,
			2, 0,
			0, 1;

		Eigen::MatrixXi cells(1, 3);
		cells << 0, 1, 2;

		return Mesh::create(vertices, cells);
	}

	std::unique_ptr<Mesh> create_test_tetra_mesh()
	{
		Eigen::MatrixXd vertices(4, 3);
		vertices << 0, 0, 0,
			1, 0, 0,
			0, 2, 0,
			0, 0, 3;

		Eigen::MatrixXi cells(1, 4);
		cells << 0, 1, 2, 3;

		return Mesh::create(vertices, cells);
	}
} // namespace

TEST_CASE("append_2d", "[mesh_test]")
{
	// Used to init geogram
	State state;

	const std::string path = POLYFEM_DATA_DIR;
	auto m1 = Mesh::create(POLYFEM_DATA_DIR + std::string("/contact/meshes/2D/arch/largeArch.01.obj"));
	const auto m2 = Mesh::create(POLYFEM_DATA_DIR + std::string("/contact/meshes/2D/arch/largeArch.02.obj"));

	m1->append(m2);
}

TEST_CASE("cmesh 2d selections and geometry queries", "[mesh_test]")
{
	auto mesh = create_test_triangle_mesh();

	REQUIRE(mesh->dimension() == 2);
	CHECK_FALSE(mesh->is_volume());
	CHECK(mesh->n_vertices() == 3);
	CHECK(mesh->n_elements() == 1);
	CHECK(mesh->n_edges() == 3);
	CHECK(mesh->n_boundary_elements() == 3);

	std::vector<int> vertices = mesh->element_vertices(0);
	CHECK(vertices == std::vector<int>({0, 1, 2}));

	Eigen::MatrixXd barycentric;
	mesh->barycentric_coords(Eigen::RowVector2d(0.5, 0.25), 0, barycentric);
	REQUIRE(barycentric.rows() == 1);
	REQUIRE(barycentric.cols() == 3);
	CHECK(barycentric.row(0).sum() == Catch::Approx(1.0).margin(1e-12));
	CHECK((barycentric.array() >= -1e-12).all());

	RowVectorNd min, max;
	mesh->bounding_box(min, max);
	CHECK(min[0] == Catch::Approx(0.0));
	CHECK(min[1] == Catch::Approx(0.0));
	CHECK(max[0] == Catch::Approx(2.0));
	CHECK(max[1] == Catch::Approx(1.0));

	mesh->set_boundary_ids({5, 6, 7});
	CHECK(mesh->has_boundary_ids());
	CHECK(mesh->get_boundary_id(0) == 5);
	CHECK(mesh->get_boundary_id(1) == 6);
	CHECK(mesh->get_boundary_id(2) == 7);

	mesh->compute_boundary_ids([](const size_t boundary_id, const std::vector<int> &vs, const RowVectorNd &, const bool is_boundary) {
		return is_boundary ? int(20 + boundary_id + vs.size()) : -1;
	});
	CHECK(mesh->get_boundary_id(0) == 22);
	CHECK(mesh->get_boundary_id(1) == 23);
	CHECK(mesh->get_boundary_id(2) == 24);

	mesh->set_body_ids({9});
	CHECK(mesh->has_body_ids());
	CHECK(mesh->get_body_id(0) == 9);

	mesh->compute_body_ids([](const size_t body_id, const std::vector<int> &vs, const RowVectorNd &barycenter) {
		return int(30 + body_id + vs.size() + (barycenter[0] > 0));
	});
	CHECK(mesh->get_body_id(0) == 34);

	mesh->compute_node_ids([](const size_t node_id, const RowVectorNd &p, const bool is_boundary) {
		return int(40 + node_id + (p[0] > 0) + (is_boundary ? 10 : 0));
	});
	CHECK(mesh->has_node_ids());
	CHECK(mesh->get_node_id(0) == 50);
	CHECK(mesh->get_node_id(1) == 52);
	CHECK(mesh->get_node_id(2) == 52);
}

TEST_CASE("cmesh 3d selections and geometry queries", "[mesh_test]")
{
	auto mesh = create_test_tetra_mesh();

	REQUIRE(mesh->dimension() == 3);
	CHECK(mesh->is_volume());
	CHECK(mesh->n_vertices() == 4);
	CHECK(mesh->n_elements() == 1);
	CHECK(mesh->n_faces() == 4);
	CHECK(mesh->n_boundary_elements() == 4);

	Eigen::MatrixXd barycentric;
	mesh->barycentric_coords(Eigen::RowVector3d(0.25, 0.5, 0.75), 0, barycentric);
	REQUIRE(barycentric.rows() == 1);
	REQUIRE(barycentric.cols() == 4);
	CHECK(barycentric.row(0).sum() == Catch::Approx(1.0).margin(1e-12));
	CHECK((barycentric.array() >= -1e-12).all());

	Eigen::MatrixXd barycenters;
	mesh->compute_element_barycenters(barycenters);
	REQUIRE(barycenters.rows() == 1);
	REQUIRE(barycenters.cols() == 3);
	CHECK(barycenters(0, 0) == Catch::Approx(0.25).margin(1e-12));
	CHECK(barycenters(0, 1) == Catch::Approx(0.5).margin(1e-12));
	CHECK(barycenters(0, 2) == Catch::Approx(0.75).margin(1e-12));

	RowVectorNd min, max;
	mesh->bounding_box(min, max);
	CHECK(min[0] == Catch::Approx(0.0));
	CHECK(min[1] == Catch::Approx(0.0));
	CHECK(min[2] == Catch::Approx(0.0));
	CHECK(max[0] == Catch::Approx(1.0));
	CHECK(max[1] == Catch::Approx(2.0));
	CHECK(max[2] == Catch::Approx(3.0));

	CHECK(mesh->tri_area(0) > 0);

	mesh->compute_boundary_ids([](const size_t face_id, const std::vector<int> &vs, const RowVectorNd &, const bool is_boundary) {
		return is_boundary ? int(50 + face_id + vs.size()) : -1;
	});
	CHECK(mesh->has_boundary_ids());
	for (int i = 0; i < mesh->n_boundary_elements(); ++i)
		CHECK(mesh->get_boundary_id(i) >= 53);

	mesh->set_body_ids({71});
	CHECK(mesh->get_body_id(0) == 71);

	mesh->compute_node_ids([](const size_t node_id, const RowVectorNd &, const bool is_boundary) {
		return int(80 + node_id + (is_boundary ? 10 : 0));
	});
	CHECK(mesh->get_node_id(0) == 90);
	CHECK(mesh->get_node_id(3) == 93);
}

TEST_CASE("obstacle meshes planes and displacement updates", "[mesh_test]")
{
	Obstacle obstacle;

	Eigen::MatrixXd vertices(2, 2);
	vertices << 0, 0,
		1, 0;
	Eigen::VectorXi codim_vertices(1);
	codim_vertices << 0;
	Eigen::MatrixXi codim_edges(1, 2);
	codim_edges << 0, 1;
	Eigen::MatrixXi faces(0, 3);

	json displacement;
	displacement["value"] = json::array({"x + t", "y + 2*t"});

	obstacle.append_mesh(vertices, codim_vertices, codim_edges, faces, displacement, "");
	CHECK(obstacle.dim() == 2);
	CHECK(obstacle.n_vertices() == 2);
	CHECK(obstacle.n_edges() == 1);
	CHECK(obstacle.codim_v().size() == 1);
	CHECK(obstacle.ndof() == 4);

	Eigen::MatrixXd sol = Eigen::MatrixXd::Constant(6, 1, -1);
	obstacle.update_displacement(0.5, sol);
	CHECK(sol(0, 0) == Catch::Approx(-1.0));
	CHECK(sol(1, 0) == Catch::Approx(-1.0));
	CHECK(sol(2, 0) == Catch::Approx(0.5));
	CHECK(sol(3, 0) == Catch::Approx(1.0));
	CHECK(sol(4, 0) == Catch::Approx(1.5));
	CHECK(sol(5, 0) == Catch::Approx(1.0));

	obstacle.set_zero(sol);
	CHECK(sol.bottomRows(obstacle.ndof()).isZero(1e-12));

	obstacle.change_displacement(0, Eigen::RowVector3d(2, 3, 4), std::string());
	Eigen::MatrixXd matrix_sol = Eigen::MatrixXd::Zero(obstacle.n_vertices(), obstacle.dim());
	obstacle.update_displacement(0.0, matrix_sol);
	CHECK(matrix_sol(0, 0) == Catch::Approx(2.0));
	CHECK(matrix_sol(0, 1) == Catch::Approx(3.0));
	CHECK(matrix_sol(1, 0) == Catch::Approx(2.0));
	CHECK(matrix_sol(1, 1) == Catch::Approx(3.0));

	obstacle.append_plane(Eigen::Vector2d(0, 0), Eigen::Vector2d(3, 4));
	REQUIRE(obstacle.planes().size() == 1);
	CHECK(obstacle.planes()[0].normal().norm() == Catch::Approx(1.0).margin(1e-12));
	CHECK(obstacle.planes()[0].vis_v().cols() == 2);
	CHECK(obstacle.planes()[0].vis_e().rows() == 10);
}

TEST_CASE("obstacle mesh sequence interpolates nodal displacement", "[mesh_test]")
{
	Obstacle obstacle;

	Eigen::MatrixXd frame0(2, 2);
	frame0 << 0, 0,
		1, 0;
	Eigen::MatrixXd frame1(2, 2);
	frame1 << 1, 2,
		2, 2;

	Eigen::VectorXi codim_vertices(0);
	Eigen::MatrixXi codim_edges(1, 2);
	codim_edges << 0, 1;
	Eigen::MatrixXi faces(0, 3);

	obstacle.append_mesh_sequence({frame0, frame1}, codim_vertices, codim_edges, faces, 1);

	Eigen::MatrixXd sol = Eigen::MatrixXd::Zero(obstacle.ndof(), 1);
	obstacle.update_displacement(0.5, sol);
	CHECK(sol(0, 0) == Catch::Approx(0.5));
	CHECK(sol(1, 0) == Catch::Approx(1.0));
	CHECK(sol(2, 0) == Catch::Approx(0.5));
	CHECK(sol(3, 0) == Catch::Approx(1.0));
}
