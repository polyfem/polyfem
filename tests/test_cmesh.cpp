////////////////////////////////////////////////////////////////////////////////
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/State.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
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

	double geogram_facet_signed_area(const GEO::Mesh &mesh, const int facet)
	{
		double area = 0.0;
		for (GEO::index_t lv = 0; lv < mesh.facets.nb_vertices(facet); ++lv)
		{
			const GEO::vec3 p0 = mesh_vertex(mesh, mesh.facets.vertex(facet, lv));
			const GEO::vec3 p1 = mesh_vertex(mesh, mesh.facets.vertex(facet, (lv + 1) % mesh.facets.nb_vertices(facet)));
			area += p0[0] * p1[1] - p1[0] * p0[1];
		}
		return 0.5 * area;
	}
} // namespace

TEST_CASE("mesh utilities geogram conversion and topology helpers", "[mesh_test][mesh_utils]")
{
	State state;

	Eigen::MatrixXd vertices2d(4, 2);
	vertices2d << 0, 0,
		1, 0,
		1, 1,
		0, 1;
	Eigen::MatrixXi quad(1, 4);
	quad << 0, 1, 2, 3;

	GEO::Mesh geogram_quad;
	to_geogram_mesh(vertices2d, quad, geogram_quad);
	REQUIRE(geogram_quad.vertices.nb() == 4);
	REQUIRE(geogram_quad.facets.nb() == 1);
	CHECK(is_planar(geogram_quad));

	const GEO::vec3 vertex = mesh_vertex(geogram_quad, 2);
	CHECK(vertex[0] == Catch::Approx(1.0));
	CHECK(vertex[1] == Catch::Approx(1.0));
	CHECK(vertex[2] == Catch::Approx(0.0));

	const GEO::vec3 barycenter = facet_barycenter(geogram_quad, 0);
	CHECK(barycenter[0] == Catch::Approx(0.5));
	CHECK(barycenter[1] == Catch::Approx(0.5));
	CHECK(barycenter[2] == Catch::Approx(0.0));

	const GEO::index_t new_vertex = mesh_create_vertex(geogram_quad, GEO::vec3(2.0, 3.0, 4.0));
	REQUIRE(new_vertex == 4);
	const GEO::vec3 created = mesh_vertex(geogram_quad, new_vertex);
	CHECK(created[0] == Catch::Approx(2.0));
	CHECK(created[1] == Catch::Approx(3.0));
	CHECK(created[2] == Catch::Approx(4.0));

	GEO::Attribute<bool> boundary_vertex(geogram_quad.vertices.attributes(), "boundary_vertex");
	for (GEO::index_t v = 0; v < geogram_quad.vertices.nb(); ++v)
		boundary_vertex[v] = true;
	std::vector<ElementType> element_tags;
	compute_element_tags(geogram_quad, element_tags);
	REQUIRE(element_tags.size() == 1);
	CHECK(element_tags[0] == ElementType::REGULAR_BOUNDARY_CUBE);

	Eigen::MatrixXd tri_vertices(3, 3);
	tri_vertices << 0, 0, 0.25,
		1, 0, 0.25,
		0, 1, 0.25;
	Eigen::MatrixXi tri(1, 3);
	tri << 0, 1, 2;
	GEO::Mesh geogram_tri;
	to_geogram_mesh(tri_vertices, tri, geogram_tri);
	CHECK(is_planar(geogram_tri));
	compute_element_tags(geogram_tri, element_tags);
	REQUIRE(element_tags.size() == 1);
	CHECK(element_tags[0] == ElementType::SIMPLEX);

	tri_vertices(2, 2) = 2.0;
	to_geogram_mesh(tri_vertices, tri, geogram_tri);
	CHECK_FALSE(is_planar(geogram_tri));

	Eigen::MatrixXi unsupported_faces(1, 5);
	unsupported_faces << 0, 1, 2, 3, 4;
	REQUIRE_THROWS_AS(to_geogram_mesh(vertices2d, unsupported_faces, geogram_tri), std::runtime_error);

	Eigen::MatrixXd reorder_vertices(4, 2);
	reorder_vertices << 0, 0,
		10, 0,
		20, 0,
		30, 0;
	Eigen::MatrixXi reorder_faces(1, 4);
	reorder_faces << 0, 1, 2, 3;
	Eigen::VectorXi colors(4);
	colors << 1, 0, 1, 0;
	Eigen::VectorXi ranges;
	reorder_mesh(reorder_vertices, reorder_faces, colors, ranges);
	REQUIRE(ranges.size() == 3);
	CHECK(ranges(0) == 0);
	CHECK(ranges(1) == 2);
	CHECK(ranges(2) == 4);
	CHECK(reorder_vertices(0, 0) == Catch::Approx(10.0));
	CHECK(reorder_vertices(1, 0) == Catch::Approx(30.0));
	CHECK(reorder_vertices(2, 0) == Catch::Approx(0.0));
	CHECK(reorder_vertices(3, 0) == Catch::Approx(20.0));
	CHECK(reorder_faces(0, 0) == 2);
	CHECK(reorder_faces(0, 1) == 0);
	CHECK(reorder_faces(0, 2) == 3);
	CHECK(reorder_faces(0, 3) == 1);
}

TEST_CASE("mesh utilities surfaces edges and orientation", "[mesh_test][mesh_utils]")
{
	State state;

	Eigen::MatrixXd tet_vertices(4, 3);
	tet_vertices << 0, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1;
	Eigen::MatrixXi outward_faces(4, 3);
	outward_faces << 0, 2, 1,
		0, 1, 3,
		0, 3, 2,
		1, 2, 3;

	CHECK(signed_volume(tet_vertices, outward_faces) == Catch::Approx(1.0 / 6.0).margin(1e-12));
	Eigen::MatrixXi inward_faces = outward_faces.rowwise().reverse().eval();
	CHECK(signed_volume(tet_vertices, inward_faces) == Catch::Approx(-1.0 / 6.0).margin(1e-12));
	orient_closed_surface(tet_vertices, inward_faces, true);
	CHECK(signed_volume(tet_vertices, inward_faces) == Catch::Approx(1.0 / 6.0).margin(1e-12));
	orient_closed_surface(tet_vertices, inward_faces, false);
	CHECK(signed_volume(tet_vertices, inward_faces) == Catch::Approx(-1.0 / 6.0).margin(1e-12));

	Eigen::MatrixXi tri_cells(2, 3);
	tri_cells << 0, 1, 2,
		2, 1, 3;
	CHECK(count_faces(2, tri_cells) == 5);

	Eigen::MatrixXi quad_cells(1, 4);
	quad_cells << 0, 1, 2, 3;
	CHECK(count_faces(2, quad_cells) == 4);

	Eigen::MatrixXi tet_cells(2, 4);
	tet_cells << 0, 1, 2, 3,
		1, 2, 3, 4;
	CHECK(count_faces(3, tet_cells) == 7);

	Eigen::MatrixXi hex_cells(1, 8);
	hex_cells << 0, 1, 2, 3, 4, 5, 6, 7;
	CHECK(count_faces(3, hex_cells) == 6);

	Eigen::MatrixXi unsupported_cells(1, 5);
	unsupported_cells << 0, 1, 2, 3, 4;
	REQUIRE_THROWS_AS(count_faces(3, unsupported_cells), std::runtime_error);

	Eigen::MatrixXd child_vertices(4, 2);
	child_vertices << 0, 0,
		0.5, 0,
		1, 0,
		1, 1;
	Eigen::MatrixXi child_edges(3, 2);
	child_edges << 0, 1,
		1, 2,
		2, 3;
	Eigen::MatrixXd parent_vertices(2, 2);
	parent_vertices << 0, 0,
		1, 0;
	Eigen::MatrixXi parent_edges(1, 2);
	parent_edges << 0, 1;
	Eigen::MatrixXi overlapping_edges;
	extract_parent_edges(child_vertices, child_edges, parent_vertices, parent_edges, overlapping_edges);
	REQUIRE(overlapping_edges.rows() == 2);
	CHECK(overlapping_edges.row(0).isApprox(Eigen::RowVector2i(0, 1)));
	CHECK(overlapping_edges.row(1).isApprox(Eigen::RowVector2i(1, 2)));

	Eigen::MatrixXd two_tet_vertices(5, 3);
	two_tet_vertices << 0, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 1;
	Eigen::MatrixXi two_tets(2, 4);
	two_tets << 0, 1, 2, 3,
		1, 2, 3, 4;
	Eigen::MatrixXd surface_vertices;
	Eigen::MatrixXi surface_tris;
	extract_triangle_surface_from_tets(two_tet_vertices, two_tets, surface_vertices, surface_tris);
	CHECK(surface_vertices.rows() == 5);
	CHECK(surface_vertices.cols() == 3);
	CHECK(surface_tris.rows() == 6);
	CHECK(surface_tris.cols() == 3);
	CHECK(surface_tris.maxCoeff() < surface_vertices.rows());
	CHECK(surface_tris.minCoeff() >= 0);

	GEO::Mesh geogram_tri;
	Eigen::MatrixXd vertices2d(4, 2);
	vertices2d << 0, 0,
		1, 0,
		1, 1,
		0, 1;
	Eigen::MatrixXi triangles(2, 3);
	triangles << 0, 1, 2,
		0, 2, 3;
	to_geogram_mesh(vertices2d, triangles, geogram_tri);
	CHECK(geogram_tri.edges.nb() == 0);
	generate_edges(geogram_tri);
	CHECK(geogram_tri.edges.nb() == 5);
	generate_edges(geogram_tri);
	CHECK(geogram_tri.edges.nb() == 5);

	GEO::Mesh clockwise_quad;
	Eigen::MatrixXi clockwise_face(1, 4);
	clockwise_face << 0, 3, 2, 1;
	to_geogram_mesh(vertices2d, clockwise_face, clockwise_quad);
	CHECK(geogram_facet_signed_area(clockwise_quad, 0) < 0);
	orient_normals_2d(clockwise_quad);
	CHECK(geogram_facet_signed_area(clockwise_quad, 0) > 0);
}

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

	obstacle.change_displacement(
		0,
		[](double x, double y, double z, double t) {
			Eigen::MatrixXd value(1, 2);
			value << x + t, y + 2 * t;
			return value;
		},
		std::string());
	matrix_sol.setZero();
	obstacle.update_displacement(0.5, matrix_sol);
	CHECK(matrix_sol(0, 0) == Catch::Approx(0.5));
	CHECK(matrix_sol(0, 1) == Catch::Approx(1.0));
	CHECK(matrix_sol(1, 0) == Catch::Approx(1.5));
	CHECK(matrix_sol(1, 1) == Catch::Approx(1.0));

	obstacle.change_displacement(0, json::array({"x", "y"}), "", "");
	matrix_sol.setZero();
	obstacle.update_displacement(0.5, matrix_sol);
	CHECK(matrix_sol(0, 0) == Catch::Approx(0.0));
	CHECK(matrix_sol(1, 0) == Catch::Approx(1.0));

	obstacle.append_plane(Eigen::Vector2d(0, 0), Eigen::Vector2d(3, 4));
	REQUIRE(obstacle.planes().size() == 1);
	CHECK(obstacle.planes()[0].normal().norm() == Catch::Approx(1.0).margin(1e-12));
	CHECK(obstacle.planes()[0].vis_v().cols() == 2);
	CHECK(obstacle.planes()[0].vis_e().rows() == 10);
}

TEST_CASE("obstacle triangular meshes interpolation paths and 3d planes", "[mesh_test]")
{
	Obstacle empty_obstacle;
	Eigen::MatrixXd no_vertices(0, 2);
	Eigen::VectorXi no_codim_vertices(0);
	Eigen::MatrixXi no_edges(0, 2);
	Eigen::MatrixXi no_faces(0, 3);
	json empty_displacement;
	empty_displacement["value"] = json::array({0, 0});
	empty_obstacle.append_mesh(no_vertices, no_codim_vertices, no_edges, no_faces, empty_displacement, "");
	empty_obstacle.append_mesh_sequence({}, no_codim_vertices, no_edges, no_faces, 24);
	CHECK(empty_obstacle.n_vertices() == 0);
	CHECK(empty_obstacle.dim() == 0);

	Obstacle invalid_obstacle;
	Eigen::MatrixXd vertices2d(3, 2);
	vertices2d << 0, 0,
		1, 0,
		0, 1;
	Eigen::MatrixXi invalid_faces(1, 4);
	invalid_faces << 0, 1, 2, 0;
	REQUIRE_THROWS(invalid_obstacle.append_mesh(vertices2d, no_codim_vertices, no_edges, invalid_faces, empty_displacement, ""));

	Obstacle obstacle;
	Eigen::MatrixXd vertices(3, 3);
	vertices << 0, 0, 0,
		1, 0, 0,
		0, 1, 0;
	Eigen::MatrixXi faces(1, 3);
	faces << 0, 1, 2;
	json displacement;
	displacement["value"] = json::array({"x + t", "y + t", "z + t"});
	displacement["interpolation"] = {{"type", "linear"}};
	obstacle.append_mesh(vertices, Eigen::VectorXi(0), Eigen::MatrixXi(0, 2), faces, displacement, "");
	CHECK(obstacle.dim() == 3);
	CHECK(obstacle.n_vertices() == 3);
	CHECK(obstacle.n_faces() == 1);
	CHECK(obstacle.n_edges() == 3);
	CHECK(obstacle.get_face_connectivity().rows() == 1);

	Eigen::MatrixXd sol = Eigen::MatrixXd::Zero(obstacle.n_vertices(), obstacle.dim());
	obstacle.update_displacement(0.5, sol);
	CHECK(sol(0, 0) == Catch::Approx(0.25));
	CHECK(sol(1, 0) == Catch::Approx(0.75));
	CHECK(sol(2, 1) == Catch::Approx(0.75));

	Obstacle array_interp_obstacle;
	json array_displacement;
	array_displacement["value"] = json::array({"x + 1", "y + 1"});
	array_displacement["interpolation"] = json::array({json{{"type", "linear"}}, json{{"type", "none"}}});
	array_interp_obstacle.append_mesh(vertices2d, no_codim_vertices, no_edges, no_faces, array_displacement, "");
	Eigen::MatrixXd array_sol = Eigen::MatrixXd::Zero(array_interp_obstacle.n_vertices(), array_interp_obstacle.dim());
	array_interp_obstacle.update_displacement(0.5, array_sol);
	CHECK(array_sol(0, 0) == Catch::Approx(0.5));
	CHECK(array_sol(0, 1) == Catch::Approx(1.0));

	Obstacle plane_only;
	plane_only.append_plane(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d::UnitY());
	plane_only.append_plane(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d::UnitZ());
	REQUIRE(plane_only.planes().size() == 2);
	for (const auto &plane : plane_only.planes())
	{
		CHECK(plane.normal().norm() == Catch::Approx(1.0).margin(1e-12));
		CHECK(plane.vis_v().cols() == 3);
		CHECK(plane.vis_f().cols() == 3);
		CHECK(plane.vis_f().rows() == 200);
		CHECK(plane.vis_e().cols() == 2);
	}
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

	sol.setZero();
	obstacle.update_displacement(2.0, sol);
	CHECK(sol(0, 0) == Catch::Approx(1.0));
	CHECK(sol(1, 0) == Catch::Approx(2.0));
	CHECK(sol(2, 0) == Catch::Approx(1.0));
	CHECK(sol(3, 0) == Catch::Approx(2.0));
}
