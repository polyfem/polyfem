////////////////////////////////////////////////////////////////////////////////
#include <numeric>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/mesh/MeshReader.hpp>
#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/State.hpp>
#include <polyfem/io/InputLoader.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/io/OutData.hpp>
#include "VarFormTestAccess.hpp"
#include <filesystem>

#include <igl/PI.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::mesh;

namespace
{
	int edge_id(const Mesh &mesh, const int v0, const int v1)
	{
		for (int e = 0; e < mesh.n_edges(); ++e)
			if ((mesh.edge_vertex(e, 0) == v0 && mesh.edge_vertex(e, 1) == v1)
				|| (mesh.edge_vertex(e, 0) == v1 && mesh.edge_vertex(e, 1) == v0))
				return e;
		return -1;
	}

	int face_id(const Mesh &mesh, std::vector<int> vertices)
	{
		std::sort(vertices.begin(), vertices.end());
		for (int f = 0; f < mesh.n_faces(); ++f)
		{
			std::vector<int> face_vertices(mesh.n_face_vertices(f));
			for (int lv = 0; lv < face_vertices.size(); ++lv)
				face_vertices[lv] = mesh.face_vertex(f, lv);
			std::sort(face_vertices.begin(), face_vertices.end());
			if (face_vertices == vertices)
				return f;
		}
		return -1;
	}

	std::unique_ptr<Mesh> create_test_triangle_mesh()
	{
		Eigen::MatrixXd vertices(3, 2);
		vertices << 0, 0,
			2, 0,
			0, 1;

		Eigen::MatrixXi cells(1, 3);
		cells << 0, 1, 2;

		return Mesh::create(MeshData(vertices, cells));
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

		return Mesh::create(MeshData(vertices, cells));
	}

	PolygonMesh make_polygon_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
	{
		PolygonMesh mesh;
		mesh.create_vertices(V.rows());
		for (int i = 0; i < V.rows(); ++i)
			mesh.point(i) = PolygonMesh::Point(V(i, 0), V(i, 1), V.cols() >= 3 ? V(i, 2) : 0);
		for (int f = 0; f < F.rows(); ++f)
		{
			std::vector<int> facet(F.cols());
			for (int lv = 0; lv < F.cols(); ++lv)
				facet[lv] = F(f, lv);
			mesh.create_polygon(facet);
		}
		return mesh;
	}

	double polygon_signed_area(const PolygonMesh &mesh, const int facet)
	{
		double area = 0.0;
		for (int lv = 0; lv < mesh.n_facet_vertices(facet); ++lv)
		{
			const PolygonMesh::Point p0 = mesh_vertex(mesh, mesh.facet_vertex(facet, lv));
			const PolygonMesh::Point p1 = mesh_vertex(mesh, mesh.facet_vertex(facet, (lv + 1) % mesh.n_facet_vertices(facet)));
			area += p0[0] * p1[1] - p1[0] * p0[1];
		}
		return 0.5 * area;
	}

	/// Minimum interior dihedral angle, in degrees, over all tetrahedra of a mesh.
	double min_tet_dihedral_angle(const Mesh &mesh)
	{
		static const int faces[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
		double min_angle = 180;
		for (int c = 0; c < mesh.n_cells(); ++c)
		{
			Eigen::Matrix<double, 4, 3> X;
			for (int lv = 0; lv < 4; ++lv)
				X.row(lv) = mesh.point(mesh.cell_vertex(c, lv));
			const Eigen::RowVector3d centroid = X.colwise().mean();

			std::array<Eigen::RowVector3d, 4> normals; // outward unit normals
			for (int f = 0; f < 4; ++f)
			{
				const Eigen::RowVector3d a = X.row(faces[f][0]);
				normals[f] = (X.row(faces[f][1]) - a).cross(X.row(faces[f][2]) - a).normalized();
				if (normals[f].dot(a - centroid) < 0)
					normals[f] *= -1;
			}

			// the interior dihedral angle is pi minus the angle between the outward normals
			for (int i = 0; i < 4; ++i)
				for (int j = i + 1; j < 4; ++j)
					min_angle = std::min(min_angle, 180 - std::acos(std::clamp(normals[i].dot(normals[j]), -1.0, 1.0)) * 180 / igl::PI);
		}
		return min_angle;
	}

	/// Flips negatively oriented tets so that every input cell has positive volume.
	Eigen::MatrixXi orient_tets(const Eigen::MatrixXd &V, Eigen::MatrixXi T)
	{
		for (int t = 0; t < T.rows(); ++t)
		{
			const Eigen::RowVector3d a = V.row(T(t, 0)), b = V.row(T(t, 1)), c = V.row(T(t, 2)), d = V.row(T(t, 3));
			if ((b - a).dot((c - a).cross(d - a)) < 0)
				std::swap(T(t, 1), T(t, 2));
		}
		return T;
	}

	struct TetMesh
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi T;
	};

	TetMesh regular_tet()
	{
		TetMesh m;
		m.V.resize(4, 3);
		m.V << 1, 1, 1,
			1, -1, -1,
			-1, 1, -1,
			-1, -1, 1;
		m.T.resize(1, 4);
		m.T << 0, 1, 2, 3;
		m.T = orient_tets(m.V, m.T);
		return m;
	}

	TetMesh generic_tet()
	{
		TetMesh m;
		m.V.resize(4, 3);
		m.V << 0, 0, 0,
			1, 0.1, 0.05,
			0.3, 0.9, 0.2,
			0.25, 0.3, 0.8;
		m.T.resize(1, 4);
		m.T << 0, 1, 2, 3;
		m.T = orient_tets(m.V, m.T);
		return m;
	}

	/// Unit cube split into 6 Kuhn tets around the (0,0,0)-(1,1,1) diagonal.
	TetMesh kuhn_cube()
	{
		TetMesh m;
		m.V.resize(8, 3);
		for (int i = 0; i < 8; ++i)
			m.V.row(i) << (i & 1), ((i >> 1) & 1), ((i >> 2) & 1);
		m.T.resize(6, 4);
		m.T << 0, 1, 3, 7,
			0, 1, 5, 7,
			0, 2, 3, 7,
			0, 2, 6, 7,
			0, 4, 5, 7,
			0, 4, 6, 7;
		m.T = orient_tets(m.V, m.T);
		return m;
	}

	struct TetRefinementCase
	{
		std::string name;
		TetMesh mesh;
		double min_angle; ///< lower bound on the minimum dihedral angle (degrees) at every level
		int n_levels;
	};

	/// Red refinement with a well-chosen inner diagonal produces at most 3 tet shapes,
	/// so the minimum dihedral angle settles after the first level. The bounds are just
	/// below those settled angles.
	std::vector<TetRefinementCase> tet_refinement_cases()
	{
		return {
			{"regular tet", regular_tet(), 54.7, 4},                    // 54.7356 = acos(1/sqrt(3))
			{"generic tet", generic_tet(), 46.0, 4},                    // 46.0339
			{"unit cube split into 6 Kuhn tets", kuhn_cube(), 35.2, 3}, // 35.2644
		};
	}

	/// Uniformly refines a tet mesh one level at a time and returns the minimum
	/// dihedral angle after each level (entry 0 is the input mesh).
	std::vector<double> uniform_refinement_min_angles(const TetMesh &input, const bool non_conforming, const int n_levels)
	{
		const auto n_boundary_faces = [](const Mesh &mesh) {
			int n = 0;
			for (int f = 0; f < mesh.n_faces(); ++f)
				n += mesh.is_boundary_face(f);
			return n;
		};

		auto mesh = Mesh::create(MeshData(input.V, input.T), non_conforming);
		REQUIRE(mesh != nullptr);
		const int n_input_boundary_faces = n_boundary_faces(*mesh);
		std::vector<double> min_angles = {min_tet_dihedral_angle(*mesh)};
		for (int level = 1; level <= n_levels; ++level)
		{
			mesh->refine(1, 0.5);
			mesh->prepare_mesh();
			REQUIRE(mesh->n_cells() == input.T.rows() << (3 * level));
			// the refined mesh is conforming: only the input boundary faces are boundary, each split in 4
			REQUIRE(n_boundary_faces(*mesh) == n_input_boundary_faces << (2 * level));
			min_angles.push_back(min_tet_dihedral_angle(*mesh));
		}
		return min_angles;
	}

	/// Checks that the minimum dihedral angle stays above a bound at every refinement level.
	void check_min_angles(const std::vector<double> &min_angles, const double bound)
	{
		for (int level = 1; level < min_angles.size(); ++level)
		{
			INFO("refinement level " << level);
			CHECK(min_angles[level] > bound);
		}
	}
} // namespace

TEST_CASE("HYBRID reader validates face vertex indices", "[mesh_test][mesh_data]")
{
	for (const int vertex : {-1, 0, 4, 100})
	{
		CAPTURE(vertex);
		std::istringstream input(
			"4 4 3\n"
			"0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
			"3 "
			+ std::to_string(vertex) + " 1 2\n"
									   "3 0 1 3\n3 0 2 3\n3 1 2 3\n"
									   "4 0 1 2 3\n4 1 1 1 1\n0\n");
		if (vertex == 0)
		{
			const auto data = MeshReader::read_hybrid(input, "test.hybrid");
			CHECK(data.vertices.rows() == 4);
			CHECK(data.cell_kernel_points.isApprox(Eigen::MatrixXd::Constant(1, 3, 0.25)));
		}
		else
		{
			CHECK_THROWS_WITH(MeshReader::read_hybrid(input, "test.hybrid"),
							  "Invalid HYBRID face vertex " + std::to_string(vertex) + " in test.hybrid: expected an index in [0, 4).");
		}
	}
}

TEST_CASE("Gmsh-ordered hex import preserves cube edges", "[mesh_test][mesh_data]")
{
	Eigen::MatrixXd vertices(8, 3);
	vertices << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
		0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
	Eigen::MatrixXi cells(1, 8);
	cells << 0, 1, 2, 3, 4, 5, 6, 7;

	const auto mesh = Mesh::create(MeshData(vertices, cells));
	REQUIRE(mesh != nullptr);
	CHECK(mesh->n_elements() == 1);
	CHECK(mesh->n_faces() == 6);
	REQUIRE(mesh->n_edges() == 12);
	for (int e = 0; e < mesh->n_edges(); ++e)
	{
		const RowVectorNd edge = mesh->point(mesh->edge_vertex(e, 0)) - mesh->point(mesh->edge_vertex(e, 1));
		CHECK(edge.norm() == Catch::Approx(1.0));
	}
}

TEST_CASE("MeshData imports volume and side selections", "[mesh_test][mesh_data]")
{
	SECTION("2D conforming and nonconforming meshes")
	{
		Eigen::MatrixXd vertices(4, 2);
		vertices << 0, 0,
			1, 0,
			1, 1,
			0, 1;
		Eigen::MatrixXi cells(2, 3);
		cells << 0, 1, 2,
			0, 2, 3;

		for (const bool non_conforming : {false, true})
		{
			CAPTURE(non_conforming);
			MeshData data(vertices, cells);
			data.body_ids = {21, 22};
			data.geometry_ids = {51, 52};
			data.node_ids = {61, 62, 63, 64};
			data.boundary_elements = {{0, 1}, {1, 2}, {0, 2}};
			data.boundary_ids = {11, 12, 14};
			const auto mesh = Mesh::create(data, non_conforming);

			REQUIRE(mesh->has_body_ids());
			CHECK(mesh->get_body_id(0) == 21);
			CHECK(mesh->get_body_id(1) == 22);
			REQUIRE(mesh->has_boundary_ids());
			CHECK(mesh->get_boundary_id(edge_id(*mesh, 0, 1)) == 11);
			CHECK(mesh->get_boundary_id(edge_id(*mesh, 1, 2)) == 12);
			CHECK(mesh->get_boundary_id(edge_id(*mesh, 0, 2)) == 14);

			const MeshData snapshot = mesh->to_mesh_data();
			const auto restored = Mesh::create(snapshot, non_conforming);
			CHECK(snapshot.vertices.isApprox(vertices));
			CHECK(snapshot.body_ids == std::vector<int>{21, 22});
			CHECK(snapshot.geometry_ids == std::vector<int>{51, 52});
			CHECK(snapshot.node_ids == std::vector<int>{61, 62, 63, 64});
			CHECK(restored->get_geometry_id(0) == 51);
			CHECK(restored->get_geometry_id(1) == 52);
			CHECK(restored->get_node_id(3) == 64);
			CHECK(restored->get_boundary_id(edge_id(*restored, 0, 1)) == 11);
			CHECK(restored->get_boundary_id(edge_id(*restored, 0, 2)) == 14);
		}
	}

	SECTION("3D conforming and nonconforming meshes")
	{
		Eigen::MatrixXd vertices(5, 3);
		vertices << 0, 0, 0,
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
			0, 0, -1;
		Eigen::MatrixXi cells(2, 4);
		cells << 0, 1, 2, 3,
			0, 2, 1, 4;

		for (const bool non_conforming : {false, true})
		{
			CAPTURE(non_conforming);
			MeshData data(vertices, cells);
			data.body_ids = {41, 42};
			data.geometry_ids = {71, 72};
			data.node_ids = {81, 82, 83, 84, 85};
			data.boundary_elements = {{0, 1, 3}, {0, 1, 2}};
			data.boundary_ids = {31, 33};
			const auto mesh = Mesh::create(data, non_conforming);

			REQUIRE(mesh->has_body_ids());
			CHECK(mesh->get_body_id(0) == 41);
			CHECK(mesh->get_body_id(1) == 42);
			REQUIRE(mesh->has_boundary_ids());
			const int exterior_face = face_id(*mesh, {0, 1, 3});
			const int interface_face = face_id(*mesh, {0, 1, 2});
			REQUIRE(exterior_face >= 0);
			REQUIRE(interface_face >= 0);
			CHECK(mesh->get_boundary_id(exterior_face) == 31);
			CHECK_FALSE(mesh->is_boundary_face(interface_face));
			CHECK(mesh->get_boundary_id(interface_face) == 33);

			const MeshData snapshot = mesh->to_mesh_data();
			const auto restored = Mesh::create(snapshot, non_conforming);
			CHECK(snapshot.vertices.isApprox(vertices));
			CHECK(snapshot.body_ids == std::vector<int>{41, 42});
			CHECK(snapshot.geometry_ids == std::vector<int>{71, 72});
			CHECK(snapshot.node_ids == std::vector<int>{81, 82, 83, 84, 85});
			CHECK(restored->get_geometry_id(0) == 71);
			CHECK(restored->get_geometry_id(1) == 72);
			CHECK(restored->get_node_id(4) == 85);
			CHECK(restored->get_boundary_id(face_id(*restored, {0, 1, 3})) == 31);
			CHECK(restored->get_boundary_id(face_id(*restored, {0, 1, 2})) == 33);
		}
	}
}

TEST_CASE("Gmsh physical sides are imported as mesh selections", "[mesh_test][gmsh]")
{
	SECTION("MSH 2.2 lines, including a higher-order line and an internal edge")
	{
		const io::FileSystemIO resources(POLYFEM_DATA_DIR);
		const auto mesh = MeshLoader(resources).load_fem("gmsh_physical_sides_2d_v22.msh");
		REQUIRE(mesh != nullptr);
		REQUIRE(mesh->has_boundary_ids());
		REQUIRE(mesh->has_body_ids());
		CHECK(mesh->get_body_id(0) == 21);
		CHECK(mesh->get_body_id(1) == 22);

		const int bottom_edge = edge_id(*mesh, 0, 1);
		const int right_edge = edge_id(*mesh, 1, 2);
		const int top_edge = edge_id(*mesh, 2, 3);
		const int interface_edge = edge_id(*mesh, 0, 2);
		REQUIRE(bottom_edge >= 0);
		REQUIRE(right_edge >= 0);
		REQUIRE(top_edge >= 0);
		REQUIRE(interface_edge >= 0);
		CHECK(mesh->get_boundary_id(bottom_edge) == 11);
		CHECK(mesh->get_boundary_id(right_edge) == 12);
		CHECK(mesh->get_boundary_id(top_edge) == 13);
		CHECK(mesh->get_boundary_id(interface_edge) == 14);
		const int untagged_edge = edge_id(*mesh, 0, 3);
		REQUIRE(untagged_edge >= 0);
		CHECK(mesh->get_boundary_id(untagged_edge) == std::numeric_limits<int>::max());
	}

	SECTION("MSH 4.1 surfaces, including an internal face")
	{
		const io::FileSystemIO resources(POLYFEM_DATA_DIR);
		const auto mesh = MeshLoader(resources).load_fem("gmsh_physical_sides_3d_v41.msh");
		REQUIRE(mesh != nullptr);
		REQUIRE(mesh->has_boundary_ids());
		REQUIRE(mesh->has_body_ids());
		CHECK(mesh->get_body_id(0) == 41);
		CHECK(mesh->get_body_id(1) == 42);

		const int side_a = face_id(*mesh, {0, 1, 3});
		const int side_b = face_id(*mesh, {1, 2, 3});
		REQUIRE(side_a >= 0);
		REQUIRE(side_b >= 0);
		CHECK(mesh->get_boundary_id(side_a) == 31);
		CHECK(mesh->get_boundary_id(side_b) == 32);
		const int interface_face = face_id(*mesh, {0, 1, 2});
		REQUIRE(interface_face >= 0);
		CHECK_FALSE(mesh->is_boundary_face(interface_face));
		CHECK(mesh->get_boundary_id(interface_face) == 33);

		const int untagged_face = face_id(*mesh, {0, 2, 3});
		REQUIRE(untagged_face >= 0);
		CHECK(mesh->get_boundary_id(untagged_face) == std::numeric_limits<int>::max());
	}
}

TEST_CASE("CMesh3D preserves cell-local vertex ordering", "[mesh_test][gmsh]")
{
	const io::FileSystemIO resources(POLYFEM_DATA_DIR);
	const auto mesh = MeshLoader(resources).load_fem("standard/reordered-local-vertices.msh");
	REQUIRE(mesh != nullptr);
	REQUIRE(mesh->n_cell_vertices(0) == 4);

	const std::array<int, 4> expected_order = {{3, 1, 2, 0}};
	for (int lv = 0; lv < int(expected_order.size()); ++lv)
		CHECK(mesh->cell_vertex(0, lv) == expected_order[lv]);
}

TEST_CASE("CMesh3D uniform tet refinement does not degenerate", "[mesh_test][refinement]")
{
	// CMesh3D cuts the inner octahedron along its shortest diagonal
	for (const auto &c : tet_refinement_cases())
	{
		DYNAMIC_SECTION(c.name)
		{
			check_min_angles(uniform_refinement_min_angles(c.mesh, false, c.n_levels), c.min_angle);
		}
	}
}

TEST_CASE("NCMesh3D tet refinement does not degenerate", "[mesh_test][refinement][ncmesh]")
{
	// NCMesh3D uses Bey's red refinement
	for (const auto &c : tet_refinement_cases())
	{
		DYNAMIC_SECTION("uniform " << c.name)
		{
			check_min_angles(uniform_refinement_min_angles(c.mesh, true, c.n_levels), c.min_angle);
		}

		DYNAMIC_SECTION("adaptive " << c.name)
		{
			auto mesh = Mesh::create(MeshData(c.mesh.V, c.mesh.T), true);
			auto &ncmesh = dynamic_cast<NCMesh3D &>(*mesh);
			for (int step = 1; step <= c.n_levels; ++step)
			{
				// refine every other cell, mixing cells of different levels with hanging nodes
				std::vector<int> ids;
				for (int e = 0; e < ncmesh.n_cells(); e += 2)
					ids.push_back(e);
				ncmesh.refine_elements(ids);
				ncmesh.prepare_mesh();

				INFO("refinement step " << step);
				CHECK(min_tet_dihedral_angle(ncmesh) > c.min_angle);
			}
		}
	}
}

TEST_CASE("mesh utilities polygon mesh and topology helpers", "[mesh_test][mesh_utils]")
{
	Eigen::MatrixXd vertices2d(4, 2);
	vertices2d << 0, 0,
		1, 0,
		1, 1,
		0, 1;
	Eigen::MatrixXi quad(1, 4);
	quad << 0, 1, 2, 3;

	PolygonMesh polygon_quad = make_polygon_mesh(vertices2d, quad);
	REQUIRE(polygon_quad.n_vertices() == 4);
	REQUIRE(polygon_quad.n_facets() == 1);
	CHECK(is_planar(vertices2d));

	const PolygonMesh::Point vertex = mesh_vertex(polygon_quad, 2);
	CHECK(vertex[0] == Catch::Approx(1.0));
	CHECK(vertex[1] == Catch::Approx(1.0));
	CHECK(vertex[2] == Catch::Approx(0.0));

	const PolygonMesh::Point barycenter = facet_barycenter(polygon_quad, 0);
	CHECK(barycenter[0] == Catch::Approx(0.5));
	CHECK(barycenter[1] == Catch::Approx(0.5));
	CHECK(barycenter[2] == Catch::Approx(0.0));

	const int new_vertex = mesh_create_vertex(polygon_quad, PolygonMesh::Point(2.0, 3.0, 4.0));
	REQUIRE(new_vertex == 4);
	const PolygonMesh::Point created = mesh_vertex(polygon_quad, new_vertex);
	CHECK(created[0] == Catch::Approx(2.0));
	CHECK(created[1] == Catch::Approx(3.0));
	CHECK(created[2] == Catch::Approx(4.0));

	for (int v = 0; v < polygon_quad.n_vertices(); ++v)
		polygon_quad.boundary_vertex[v] = true;
	std::vector<ElementType> element_tags;
	compute_element_tags(polygon_quad, element_tags);
	REQUIRE(element_tags.size() == 1);
	CHECK(element_tags[0] == ElementType::REGULAR_BOUNDARY_CUBE);

	Eigen::MatrixXd tri_vertices(3, 3);
	tri_vertices << 0, 0, 0.25,
		1, 0, 0.25,
		0, 1, 0.25;
	Eigen::MatrixXi tri(1, 3);
	tri << 0, 1, 2;
	PolygonMesh polygon_tri = make_polygon_mesh(tri_vertices, tri);
	CHECK(is_planar(tri_vertices));
	compute_element_tags(polygon_tri, element_tags);
	REQUIRE(element_tags.size() == 1);
	CHECK(element_tags[0] == ElementType::SIMPLEX);

	tri_vertices(2, 2) = 2.0;
	CHECK_FALSE(is_planar(tri_vertices));

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
	// Two hexes sharing the face (1, 2, 6, 5) of the first one.
	Eigen::MatrixXi two_hex_cells(2, 8);
	two_hex_cells << 0, 1, 2, 3, 4, 5, 6, 7,
		1, 8, 9, 2, 5, 10, 11, 6;
	CHECK(count_faces(3, two_hex_cells) == 11);

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

	Eigen::MatrixXd vertices2d(4, 2);
	vertices2d << 0, 0,
		1, 0,
		1, 1,
		0, 1;
	Eigen::MatrixXi triangles(2, 3);
	triangles << 0, 1, 2,
		0, 2, 3;
	PolygonMesh polygon_tri = make_polygon_mesh(vertices2d, triangles);
	CHECK(polygon_tri.n_edges() == 0);
	generate_edges(polygon_tri);
	CHECK(polygon_tri.n_edges() == 5);
	generate_edges(polygon_tri);
	CHECK(polygon_tri.n_edges() == 5);

	Eigen::MatrixXi clockwise_face(1, 4);
	clockwise_face << 0, 3, 2, 1;
	PolygonMesh clockwise_quad = make_polygon_mesh(vertices2d, clockwise_face);
	CHECK(polygon_signed_area(clockwise_quad, 0) < 0);
	orient_normals_2d(clockwise_quad);
	CHECK(polygon_signed_area(clockwise_quad, 0) > 0);
	CHECK(clockwise_quad.facet_vertex(0, 0) == 1);
	CHECK(clockwise_quad.facet_vertex(0, 1) == 2);
	CHECK(clockwise_quad.facet_vertex(0, 2) == 3);
	CHECK(clockwise_quad.facet_vertex(0, 3) == 0);
}

TEST_CASE("append_2d", "[mesh_test]")
{
	const std::string path = POLYFEM_DATA_DIR;
	const io::FileSystemIO resources(POLYFEM_DATA_DIR);
	auto m1 = MeshLoader(resources).load_fem("contact/meshes/2D/arch/largeArch.01.obj");
	const auto m2 = MeshLoader(resources).load_fem("contact/meshes/2D/arch/largeArch.02.obj");

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
	const io::FileSystemIO resources(".");
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

	obstacle.append_mesh(vertices, codim_vertices, codim_edges, faces, displacement, resources);
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

	obstacle.change_displacement(0, json::array({"x", "y"}), resources, "");
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
	const io::FileSystemIO resources(".");
	Obstacle empty_obstacle;
	Eigen::MatrixXd no_vertices(0, 2);
	Eigen::VectorXi no_codim_vertices(0);
	Eigen::MatrixXi no_edges(0, 2);
	Eigen::MatrixXi no_faces(0, 3);
	json empty_displacement;
	empty_displacement["value"] = json::array({0, 0});
	empty_obstacle.append_mesh(no_vertices, no_codim_vertices, no_edges, no_faces, empty_displacement, resources);
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
	REQUIRE_THROWS(invalid_obstacle.append_mesh(vertices2d, no_codim_vertices, no_edges, invalid_faces, empty_displacement, resources));

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
	obstacle.append_mesh(vertices, Eigen::VectorXi(0), Eigen::MatrixXi(0, 2), faces, displacement, resources);
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
	array_interp_obstacle.append_mesh(vertices2d, no_codim_vertices, no_edges, no_faces, array_displacement, resources);
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

TEST_CASE("node ids follow the FE node numbering", "[mesh_test]")
{
	Eigen::MatrixXd vertices(4, 2);
	vertices << 0, 0, 1, 0, 1, 1, 0, 1;
	Eigen::MatrixXi cells(2, 3);
	cells << 0, 1, 2, 0, 2, 3;
	auto mesh = Mesh::create(MeshData(vertices, cells));
	mesh->compute_node_ids([](const size_t v, const RowVectorNd &, bool) { return 10 + int(v); });

	// Higher-order numbering: vertex nodes interleaved with other nodes.
	Eigen::VectorXi in_node_to_node(4);
	in_node_to_node << 5, 0, 7, 2;
	mesh->update_nodes(in_node_to_node);
	CHECK(mesh->get_node_id(5) == 10);
	CHECK(mesh->get_node_id(0) == 11);
	CHECK(mesh->get_node_id(7) == 12);
	CHECK(mesh->get_node_id(2) == 13);
	for (const int other : {1, 3, 4, 6, 8, 100})
		CHECK(mesh->get_node_id(other) == -1);

	// Unreferenced input vertices map to -1.
	auto mesh2 = Mesh::create(MeshData(vertices, cells));
	mesh2->compute_node_ids([](const size_t v, const RowVectorNd &, bool) { return 10 + int(v); });
	Eigen::VectorXi partial(4);
	partial << 1, -1, 0, 2;
	mesh2->update_nodes(partial);
	CHECK(mesh2->get_node_id(0) == 12);
	CHECK(mesh2->get_node_id(1) == 10);
	CHECK(mesh2->get_node_id(2) == 13);
}

TEST_CASE("nodal boundary conditions with higher-order elements", "[mesh_test]")
{
	io::LoadedInput loaded = io::load_json_input(std::string(POLYFEM_DATA_DIR) + "/standard/laplace_nodal_neumann.json");
	json args = loaded.config;
	auto common_resources = utils::apply_common_params(args, *loaded.resources);
	const io::ResourceIO &resources = common_resources ? *common_resources : *loaded.resources;
	args["/geometry/n_refs"_json_pointer] = 0;
	args["/space/discr_order"_json_pointer] = 2;
	args["/output/log/level"_json_pointer] = "error";
	args.erase("tests");

	State state;
	state.init(args, resources, true);
	state.set_max_threads(1);
	state.load_mesh();
	Eigen::MatrixXd sol;
	REQUIRE_NOTHROW(state.solve(sol));
	CHECK(sol.size() > 0);
	CHECK(sol.allFinite());
}

TEST_CASE("obstacle mesh arrays replicate codimensional points", "[mesh_test]")
{
	namespace fs = std::filesystem;
	const fs::path dir = fs::temp_directory_path() / "polyfem-mesh-array-test";
	fs::create_directories(dir);
	{
		// One edge (v1-v2) and two free points (v3, v4): 4 vertices, 2 points.
		std::ofstream obj(dir / "edge_points.obj");
		obj << "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0.1 0.1 0\nl 1 2\n";
	}

	json in_args = R"({
		"materials": {"type": "NeoHookean", "E": 1e5, "nu": 0.3, "rho": 1e3},
		"contact": {"enabled": true, "dhat": 1e-3},
		"time": {"tend": 1, "time_steps": 1},
		"geometry": [
			{"mesh": "", "transformation": {"translation": [5, 5]}},
			{"mesh": "", "type": "mesh_array", "is_obstacle": true,
			 "array": {"offset": 1, "size": [3, 1]}}
		],
		"output": {"log": {"level": "error"}}
	})"_json;
	in_args["geometry"][0]["mesh"] = std::string(POLYFEM_DATA_DIR) + "/contact/meshes/2D/simple/square.obj";
	in_args["geometry"][1]["mesh"] = (dir / "edge_points.obj").string();

	State state;
	state.init(in_args, true);
	state.load_mesh();
	test::VarFormTestAccess::prepare(*state.variational_formulation);
	const io::OutputSpace output = state.variational_formulation->output_space();
	REQUIRE(output.obstacle != nullptr);
	const mesh::Obstacle &obstacle = *output.obstacle;
	CHECK(obstacle.n_vertices() == 12);
	const Eigen::VectorXi &points = obstacle.codim_v();
	REQUIRE(points.size() == 6);
	std::vector<int> sorted(points.data(), points.data() + points.size());
	std::sort(sorted.begin(), sorted.end());
	CHECK(sorted == std::vector<int>{2, 3, 6, 7, 10, 11});

	// A size that does not match the dimension is rejected.
	in_args["geometry"][1]["array"]["size"] = {3};
	State bad;
	bad.init(in_args, true);
	CHECK_THROWS(bad.load_mesh());
	fs::remove_all(dir);
}

TEST_CASE("hex meshes round-trip through MeshData", "[mesh_test][mesh_data]")
{
	Eigen::MatrixXd vertices(12, 3);
	vertices << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
		0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
		2, 0, 0, 2, 1, 0, 2, 0, 1, 2, 1, 1;
	Eigen::MatrixXi cells(2, 8);
	cells << 0, 1, 2, 3, 4, 5, 6, 7,
		1, 8, 9, 2, 5, 10, 11, 6;

	auto check_unit_edges = [](const Mesh &mesh) {
		REQUIRE(mesh.n_elements() == 2);
		CHECK(mesh.n_edges() == 20);
		for (int e = 0; e < mesh.n_edges(); ++e)
		{
			const RowVectorNd edge = mesh.point(mesh.edge_vertex(e, 0)) - mesh.point(mesh.edge_vertex(e, 1));
			CHECK(edge.norm() == Catch::Approx(1.0));
		}
	};

	const auto mesh = Mesh::create(MeshData(vertices, cells));
	check_unit_edges(*mesh);
	const MeshData data = mesh->to_mesh_data();
	CHECK(data.elements == cells);
	const auto restored = Mesh::create(data);
	check_unit_edges(*restored);
}

TEST_CASE("affine transformations move element kernels", "[mesh_test]")
{
	Eigen::MatrixXd vertices(8, 3);
	vertices << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
		0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
	Eigen::MatrixXi cells(1, 8);
	cells << 0, 1, 2, 3, 4, 5, 6, 7;
	auto mesh = Mesh::create(MeshData(vertices, cells));
	auto &mesh3d = dynamic_cast<Mesh3D &>(*mesh);

	MatrixNd A(3, 3);
	A << 2, 0, 0, 0, 3, 0, 0, 0, 4;
	VectorNd b(3);
	b << 10, 20, 30;
	mesh->apply_affine_transformation(A, b);

	const RowVectorNd expected = (A * Eigen::Vector3d(0.5, 0.5, 0.5) + b).transpose();
	CHECK((mesh3d.kernel(0) - expected).norm() < 1e-12);
	CHECK((mesh3d.cell_barycenter(0) - expected).norm() < 1e-12);
}

TEST_CASE("appending a curved mesh to a linear one keeps its nodes aligned", "[mesh_test]")
{
	const std::string dir = std::string(POLYFEM_DATA_DIR) + "/contact/meshes/3D/simple/sphere/coarse/";
	const io::FileSystemIO resources(dir);
	// A linear mesh built from plain vertices and cells has no curved-node slots.
	const MeshData p1 = MeshLoader(resources).load_fem("P1.msh")->to_mesh_data();
	auto merged = Mesh::create(MeshData(p1.vertices, p1.elements));
	const auto linear = Mesh::create(MeshData(p1.vertices, p1.elements));
	const auto curved = MeshLoader(resources).load_fem("P2.msh");
	REQUIRE(curved->orders().maxCoeff() == 2);
	merged->append(curved);

	const auto &merged3d = dynamic_cast<const Mesh3D &>(*merged);
	const auto &curved3d = dynamic_cast<const Mesh3D &>(*curved);
	const auto &linear3d = dynamic_cast<const Mesh3D &>(*linear);
	REQUIRE(merged->n_cells() == linear->n_cells() + curved->n_cells());
	const int n_shift = linear->n_cells();
	int n_curved = 0;
	for (int c = 0; c < merged->n_cells(); ++c)
	{
		auto index = merged3d.get_index_from_element(c);
		const bool second = c >= n_shift;
		auto reference = second ? curved3d.get_index_from_element(c - n_shift) : linear3d.get_index_from_element(c);
		const Mesh3D &source = second ? curved3d : linear3d;
		for (int le = 0; le < 3; ++le)
		{
			const RowVectorNd node = merged3d.edge_node(index, 1, 0).first;
			const RowVectorNd expected = source.edge_node(reference, 1, 0).first;
			CHECK((node - expected).norm() < 1e-12);
			const RowVectorNd midpoint = (merged->point(index.vertex) + merged->point(merged3d.switch_vertex(index).vertex)) / 2;
			n_curved += (node - midpoint).norm() > 1e-10;
			index = merged3d.next_around_face(index);
			reference = source.next_around_face(reference);
		}
	}
	CHECK(n_curved > 0);
}

TEST_CASE("appending node ids gives the receiving mesh the default ids", "[mesh_test]")
{
	Eigen::MatrixXd vertices(3, 2);
	vertices << 0, 0, 1, 0, 0, 1;
	Eigen::MatrixXi cells(1, 3);
	cells << 0, 1, 2;
	auto first = Mesh::create(MeshData(vertices, cells));
	auto second = Mesh::create(MeshData(vertices, cells));
	second->compute_node_ids([](const size_t, const RowVectorNd &, bool) { return 5; });
	first->append(second);
	for (int v = 0; v < 6; ++v)
		CHECK(first->get_node_id(v) == (v < 3 ? -1 : 5));
}

TEST_CASE("refining mixed triangle and quad meshes keeps the triangles", "[mesh_test]")
{
	Eigen::MatrixXd vertices(5, 2);
	vertices << 0, 0, 1, 0, 1, 1, 0, 1, 2, 0;
	Eigen::MatrixXi cells(2, 4);
	cells << 0, 1, 2, 3,
		1, 4, 2, -1;

	for (const double t : {0.0, 0.5})
	{
		CAPTURE(t);
		auto mesh = Mesh::create(MeshData(vertices, cells));
		REQUIRE(mesh->n_elements() == 2);
		mesh->refine(1, t);

		CHECK(mesh->n_elements() == 8);
		int n_triangles = 0, n_boundary = 0;
		double area = 0;
		for (int e = 0; e < mesh->n_elements(); ++e)
		{
			const int n = mesh->n_face_vertices(e);
			n_triangles += n == 3;
			for (int j = 0; j < n; ++j)
			{
				const RowVectorNd a = mesh->point(mesh->face_vertex(e, j));
				const RowVectorNd b = mesh->point(mesh->face_vertex(e, (j + 1) % n));
				area += 0.5 * (a[0] * b[1] - a[1] * b[0]);
			}
		}
		for (int e = 0; e < mesh->n_edges(); ++e)
			n_boundary += mesh->is_boundary_edge(e);
		CHECK(n_triangles == 4);
		CHECK(area == Catch::Approx(1.5));
		CHECK(n_boundary == 10);
	}
}

TEST_CASE("unsupported higher-order elements are rejected", "[mesh_test]")
{
	for (const int dim : {2, 3})
	{
		CAPTURE(dim);
		const int n_corners = dim + 1;
		Eigen::MatrixXd vertices = Eigen::MatrixXd::Zero(n_corners, dim);
		for (int d = 0; d < dim; ++d)
			vertices(d + 1, d) = 1;
		Eigen::MatrixXi cells(1, n_corners);
		for (int i = 0; i < n_corners; ++i)
			cells(0, i) = i;

		// One node more than the linear element.
		MeshData data(vertices, cells);
		data.higher_order_nodes.resize(n_corners + 1, dim);
		data.higher_order_nodes.topRows(n_corners) = vertices;
		data.higher_order_nodes.row(n_corners) = vertices.colwise().mean();
		data.higher_order_connectivity = {std::vector<int>(n_corners + 1)};
		std::iota(data.higher_order_connectivity[0].begin(), data.higher_order_connectivity[0].end(), 0);
		REQUIRE_THROWS_WITH(Mesh::create(data), Catch::Matchers::ContainsSubstring("Unsupported higher-order"));

		// The linear connectivity is accepted and gives linear orders.
		data.higher_order_nodes = vertices;
		data.higher_order_connectivity[0].pop_back();
		const auto mesh = Mesh::create(data);
		CHECK((mesh->orders().size() == 0 || mesh->orders().maxCoeff() == 1));
	}
}

TEST_CASE("mesh nodes place lazily created nodes at their primitive", "[mesh_test]")
{
	Eigen::MatrixXd vertices(4, 2);
	vertices << 0, 0, 1, 0, 1, 1, 0, 1;
	Eigen::MatrixXi cells(2, 3);
	cells << 0, 1, 2, 0, 2, 3;
	const auto mesh = Mesh::create(MeshData(vertices, cells));

	// Several node slots per edge and face, as for mixed-order spaces.
	MeshNodes nodes(*mesh, false, true, 2, 3);
	for (int e = mesh->n_edges() - 1; e >= 0; --e)
	{
		const int node = nodes.node_id_from_edge(e);
		CHECK((nodes.node_position(node) - mesh->edge_barycenter(e)).norm() < 1e-14);
		CHECK(nodes.edge_from_node_id(node) == e);
		CHECK(nodes.primitive_to_node()[nodes.primitive_from_edge(e)] == node);
	}
	for (int f = 0; f < mesh->n_faces(); ++f)
	{
		const int node = nodes.node_id_from_face(f);
		CHECK((nodes.node_position(node) - mesh->face_barycenter(f)).norm() < 1e-14);
		CHECK(nodes.face_from_node_id(node) == f);
		CHECK(nodes.primitive_to_node()[nodes.primitive_from_face(f)] == node);
	}
	CHECK(nodes.num_edge_nodes() == mesh->n_edges());
	CHECK(nodes.num_face_nodes() == mesh->n_faces());
}

TEST_CASE("element boxes of meshes with negative coordinates", "[mesh_test]")
{
	for (const int dim : {2, 3})
	{
		CAPTURE(dim);
		Eigen::MatrixXd vertices = Eigen::MatrixXd::Zero(dim + 1, dim);
		for (int d = 0; d < dim; ++d)
			vertices(d + 1, d) = 1;
		vertices.array() -= 5;
		Eigen::MatrixXi cells(1, dim + 1);
		for (int i = 0; i <= dim; ++i)
			cells(0, i) = i;
		const auto mesh = Mesh::create(MeshData(vertices, cells));

		std::vector<std::array<Eigen::Vector3d, 2>> boxes;
		mesh->elements_boxes(boxes);
		REQUIRE(boxes.size() == 1);
		for (int d = 0; d < dim; ++d)
		{
			CHECK(boxes[0][0][d] == -5);
			CHECK(boxes[0][1][d] == -4);
		}
	}
}
