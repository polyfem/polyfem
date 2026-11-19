#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshData.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/PointTriangleDistance.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh2D/PolygonMesh.hpp>
#include <polyfem/mesh/mesh2D/PolygonUtils.hpp>
#include <polyfem/io/ResourceIO.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <Eigen/Dense>

#include <limits>
#include <vector>

using namespace polyfem;
using namespace polyfem::mesh;

TEST_CASE("surface offsets are stable when incident normals cancel", "[mesh_geometry]")
{
	const Eigen::RowVector3d interior(0.25, 0.25, -1);
	const double distance = 0.1;
	const Eigen::RowVector3d expected = distance * (-interior).normalized();
	for (const double perturbation : {-1e-16, 0.0, 1e-16})
	{
		Eigen::MatrixXd vertices(5, 3);
		vertices << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, perturbation, 1, 0, 0;
		Eigen::MatrixXi faces(2, 3);
		faces << 0, 1, 2, 0, 3, 4;
		Eigen::MatrixXd offset;
		offset_surface_vertices(vertices, faces, interior, distance, offset);
		CHECK(offset.allFinite());
		CHECK((offset.row(0) - expected).norm() < 1e-14);
		CHECK((offset.row(1) - vertices.row(1) - Eigen::RowVector3d(0, 0, distance)).norm() < 1e-14);
	}
}

TEST_CASE("visibility kernel center is independent of redundant corners", "[mesh_geometry]")
{
	Eigen::MatrixXd polygon(5, 2);
	polygon << 0, 0, 3, 0, 2, 1, 1, 1, 0, 1;
	const Eigen::RowVector3d expected(19.0 / 15, 7.0 / 15, 0);
	for (int start = 0; start < polygon.rows(); ++start)
	{
		Eigen::MatrixXd reordered(polygon.rows(), 2);
		for (int i = 0; i < polygon.rows(); ++i)
			reordered.row(i) = polygon.row((start + i) % polygon.rows());
		Eigen::RowVector3d center;
		REQUIRE(is_star_shaped(reordered, center));
		CHECK((center - expected).norm() < 1e-12);
	}
}

TEST_CASE("polygon sampling is stable under collinear subdivisions", "[mesh_geometry]")
{
	Eigen::MatrixXd square(4, 2);
	square << 0, 0, 1, 0, 1, 1, 0, 1;
	Eigen::MatrixXd expected;
	sample_polygon(square, 11, expected);
	for (const double perturbation : {-1e-16, 0.0, 1e-16})
	{
		Eigen::MatrixXd subdivided(8, 2);
		subdivided << 0, 0, 0.25, perturbation, 0.75, perturbation, 1, 0,
			1, 1, 0, 1, perturbation, 0.75, perturbation, 0.25;
		for (int start = 0; start < subdivided.rows(); ++start)
		{
			const Eigen::RowVector2d translation(10 * start, 20 * start);
			Eigen::MatrixXd reordered(subdivided.rows(), 2);
			for (int i = 0; i < subdivided.rows(); ++i)
				reordered.row(i) = subdivided.row((start + i) % subdivided.rows()) + translation;
			Eigen::MatrixXd samples;
			sample_polygon(reordered, 11, samples);
			CHECK(((samples.rowwise() - translation) - expected).cwiseAbs().maxCoeff() < 1e-13);
		}
	}
}

TEST_CASE("point-triangle distance of degenerate triangles", "[mesh_geometry]")
{
	const double p[3] = {0.5, 1, 0};
	// Collinear vertices: the distance to the segment.
	{
		const double a[3] = {0, 0, 0}, b[3] = {1, 0, 0}, c[3] = {2, 0, 0};
		CHECK(point_triangle_squared_distance(p, a, b, c) == Catch::Approx(1.0));
		const double before[3] = {-1, 1, 0}, after[3] = {3, 1, 0};
		CHECK(point_triangle_squared_distance(before, a, b, c) == Catch::Approx(2.0));
		CHECK(point_triangle_squared_distance(after, a, b, c) == Catch::Approx(2.0));
	}
	// All vertices equal: the distance to the point.
	{
		const double a[3] = {0, 0, 0};
		CHECK(point_triangle_squared_distance(p, a, a, a) == Catch::Approx(1.25));
	}
	// Points in the plane of a triangle are at distance zero (never negative).
	{
		const double a[3] = {0.1, 0.2, 0.3}, b[3] = {1.7, 0.4, 0.9}, c[3] = {0.3, 1.9, 1.3};
		for (int i = 1; i < 10; ++i)
			for (int j = 1; i + j < 10; ++j)
			{
				const double u = i / 10.0, v = j / 10.0;
				double q[3];
				for (int d = 0; d < 3; ++d)
					q[d] = a[d] + u * (b[d] - a[d]) + v * (c[d] - a[d]);
				const double d2 = point_triangle_squared_distance(q, a, b, c);
				CHECK(d2 >= 0);
				CHECK(d2 < 1e-12);
			}
	}
}

TEST_CASE("signed distances to an empty surface", "[mesh_geometry]")
{
	const Eigen::MatrixXd V(0, 3);
	const Eigen::MatrixXi F(0, 3);
	Eigen::MatrixXd P(2, 3);
	P << 0, 0, 0, 1, 2, 3;
	Eigen::VectorXd D;
	signed_squared_distances(V, F, P, D);
	REQUIRE(D.size() == 2);
	for (int i = 0; i < 2; ++i)
		CHECK(std::abs(D(i)) == std::numeric_limits<double>::max());
}

TEST_CASE("parent edges ignore collinear edges outside the parent", "[mesh_geometry]")
{
	Eigen::MatrixXd child_vertices(3, 2);
	child_vertices << 0, 0, 1, 0, 1.5, 0;
	Eigen::MatrixXi child_edges(2, 2);
	child_edges << 0, 1, 1, 2;
	Eigen::MatrixXd parent_vertices(2, 2);
	parent_vertices << 0, 0, 1, 0;
	Eigen::MatrixXi parent_edges(1, 2);
	parent_edges << 0, 1;
	Eigen::MatrixXi overlapping_edges;
	extract_parent_edges(child_vertices, child_edges, parent_vertices, parent_edges, overlapping_edges);
	REQUIRE(overlapping_edges.rows() == 1);
	CHECK(overlapping_edges.row(0) == Eigen::RowVector2i(0, 1));
}

TEST_CASE("polygon mesh connected components", "[mesh_geometry]")
{
	PolygonMesh mesh;
	// Two squares split into two triangles each, sharing nothing.
	for (const double x0 : {0.0, 2.0})
	{
		mesh.create_vertex(PolygonMesh::Point(x0, 0, 0));
		mesh.create_vertex(PolygonMesh::Point(x0 + 1, 0, 0));
		mesh.create_vertex(PolygonMesh::Point(x0 + 1, 1, 0));
		mesh.create_vertex(PolygonMesh::Point(x0, 1, 0));
	}
	mesh.create_polygon({0, 1, 2});
	mesh.create_polygon({0, 2, 3});
	mesh.create_polygon({4, 5, 6});
	mesh.create_polygon({4, 6, 7});
	mesh.connect_facets();
	std::vector<int> component;
	CHECK(mesh.connected_components(component) == 2);
	CHECK(component == std::vector<int>{0, 0, 1, 1});
}

TEST_CASE("P3 triangle meshes keep their face nodes", "[mesh_geometry]")
{
	const io::FileSystemIO resources(POLYFEM_DATA_DIR);
	const auto mesh = MeshLoader(resources).load_fem("circle_p3.msh");
	REQUIRE(mesh->orders().size() > 0);
	CHECK(mesh->orders().maxCoeff() == 3);
	const auto &mesh2d = dynamic_cast<const Mesh2D &>(*mesh);
	int n_curved = 0;
	for (int f = 0; f < mesh->n_faces(); ++f)
	{
		if (mesh->orders()(f) != 3)
			continue;
		const auto [node, id] = mesh2d.face_node(mesh2d.get_index_from_face(f), 1, 1, 1);
		CHECK(id >= 0);
		CHECK(node.size() == 2);
		++n_curved;
	}
	CHECK(n_curved > 0);
}

TEST_CASE("volume meshes reject non-conforming quad/triangle interfaces", "[mesh_geometry]")
{
	// A hex standing on two tets that split its bottom face along a diagonal.
	Eigen::MatrixXd vertices(9, 3);
	vertices << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
		0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
		0.5, 0.5, -1;
	Eigen::MatrixXi cells(3, 8);
	cells << 0, 1, 2, 3, 4, 5, 6, 7,
		0, 1, 2, 8, -1, -1, -1, -1,
		0, 2, 3, 8, -1, -1, -1, -1;
	CHECK_THROWS_WITH(Mesh::create(MeshData(vertices, cells)), Catch::Matchers::ContainsSubstring("non-conforming quad/triangle"));
}
