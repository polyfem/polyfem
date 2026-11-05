////////////////////////////////////////////////////////////////////////////////
#include <polyfem/State.hpp>
#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <iostream>
#include <fstream>
#include <cmath>

#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include <polyfem/utils/MaybeParallelFor.hpp>

#include <catch2/catch_test_macros.hpp>

#include "VarFormTestAccess.hpp"

#include <polyfem/mesh/MeshData.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/io/ResourceIO.hpp>

#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// FIXME: these tests are not working because of the way the mesh is loaded in State. We should refactor State to allow loading a mesh from memory, instead of only from file.
/*
using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::mesh;

TEST_CASE("ncmesh2d", "[ncmesh]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"materials": {"type": "Laplacian"},

			"geometry": [{
				"mesh": "",
				"enabled": true,
				"type": "mesh",
				"surface_selection": 7
			}],

			"space":{
				"discr_order": 5,
				"advanced": {
					"isoparametric": false,
					"bc_method": "sample"
				}
			},

			"boundary_conditions": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": "x^5+y^5"
				}],
				"rhs": "20*x^3+20*y^3"
			},

			"output": {
				"reference": {
					"solution": "x^5+y^5",
					"gradient": ["5*x^4","5*y^4"]
				},
				"paraview": {
					"high_order_mesh": false
				}
			}
		})"_json;
	in_args["geometry"][0]["mesh"] = path + "/contact/meshes/2D/simple/circle/circle36.obj";

	State state;
	state.set_max_threads(1);
	state.init_logger("", spdlog::level::off, spdlog::level::off, false);
	state.init(in_args, true);

	state.load_mesh(true);
	state.update_mesh([](mesh::Mesh &mesh) {
		auto *ncmesh = dynamic_cast<NCMesh2D *>(&mesh);
		REQUIRE(ncmesh != nullptr);

		for (int n = 0; n < 2; n++)
		{
			ncmesh->prepare_mesh();
			std::vector<int> ref_ids(ncmesh->n_faces() / 2);
			for (int i = 0; i < ref_ids.size(); i++)
				ref_ids[i] = 2 * i;

			ncmesh->refine_elements(ref_ids);
		}
		ncmesh->prepare_mesh();
	});

	Eigen::MatrixXd sol;

	state.solve(sol);
	const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);

	// state.save_vtu("debug.vtu", 1.);

	REQUIRE(fabs(stats.h1_semi_err) < 1e-9);
	REQUIRE(fabs(stats.l2_err) < 1e-10);
}

TEST_CASE("ncmesh3d", "[ncmesh]")
{
	const std::string path = POLYFEM_DATA_DIR;
	json in_args = R"(
		{
			"materials": {"type": "Laplacian"},

			"geometry": [{
				"mesh": "",
				"enabled": true,
				"type": "mesh",
				"surface_selection": 7
			}],

			"space":{
				"discr_order": 2,
				"advanced": {
					"isoparametric": false,
					"bc_method": "sample"
				}
			},

			"boundary_conditions": {
				"dirichlet_boundary": [{
					"id": "all",
					"value": "x^2+y^2+z^2"
				}],
				"rhs": 6
			},

			"output": {
				"reference": {
					"solution": "x^2+y^2+z^2",
					"gradient": ["2*x","2*y","2*z"]
				},
				"paraview": {
					"high_order_mesh": false
				}
			},

			"solver": {
				"linear": {
					"solver": "Eigen::SimplicialLDLT"
				}
			}
		}
	)"_json;
	in_args["geometry"][0]["mesh"] = path + "/contact/meshes/3D/simple/bar/bar-186.msh";

	State state;
	state.init_logger("", spdlog::level::off, spdlog::level::off, false);
	state.init(in_args, true);

	state.load_mesh(true);
	state.update_mesh([](mesh::Mesh &mesh) {
		auto *ncmesh = dynamic_cast<NCMesh3D *>(&mesh);
		REQUIRE(ncmesh != nullptr);

		for (int n = 0; n < 2; n++)
		{
			ncmesh->prepare_mesh();
			std::vector<int> ref_ids(int(ncmesh->n_cells() / 2.01));
			for (int i = 0; i < ref_ids.size(); i++)
				ref_ids[i] = i * 2;

			ncmesh->refine_elements(ref_ids);
		}
		ncmesh->prepare_mesh();
	});

	Eigen::MatrixXd sol;

	state.solve(sol);
	const io::OutStatsData stats = state.variational_formulation->compute_errors(sol);

	// state.save_vtu("debug.vtu", 1.);

	REQUIRE(fabs(stats.h1_semi_err) < 1e-7);
	REQUIRE(fabs(stats.l2_err) < 1e-8);
}
*/

namespace
{
	using namespace polyfem;
	using namespace polyfem::mesh;

	/// Everything a refined mesh exposes through its public queries.
	struct QueriedState
	{
		int n_vertices, n_edges, n_faces, n_cells;
		std::vector<int> edge_vertices, boundary_ids, body_ids;
		std::vector<double> points;
		bool operator==(const QueriedState &o) const
		{
			return n_vertices == o.n_vertices && n_edges == o.n_edges && n_faces == o.n_faces && n_cells == o.n_cells
				   && edge_vertices == o.edge_vertices && boundary_ids == o.boundary_ids && body_ids == o.body_ids
				   && points == o.points;
		}
	};

	QueriedState query(Mesh &mesh)
	{
		QueriedState q;
		q.n_vertices = mesh.n_vertices();
		q.n_edges = mesh.n_edges();
		q.n_faces = mesh.n_faces();
		q.n_cells = mesh.n_cells();
		for (int v = 0; v < mesh.n_vertices(); ++v)
			for (int d = 0; d < mesh.dimension(); ++d)
				q.points.push_back(mesh.point(v)[d]);
		for (int e = 0; e < mesh.n_edges(); ++e)
		{
			q.edge_vertices.push_back(mesh.edge_vertex(e, 0));
			q.edge_vertices.push_back(mesh.edge_vertex(e, 1));
		}
		mesh.set_body_ids(std::vector<int>(mesh.n_elements(), 3));
		for (int e = 0; e < mesh.n_elements(); ++e)
			q.body_ids.push_back(mesh.get_body_id(e));
		mesh.compute_boundary_ids([](const size_t, const std::vector<int> &, const RowVectorNd &p, bool boundary) {
			return boundary ? (p[0] < 0.5 ? 1 : 2) : -1;
		});
		for (int b = 0; b < mesh.n_boundary_elements(); ++b)
			q.boundary_ids.push_back(mesh.get_boundary_id(b));
		return q;
	}

	MeshData nc_test_mesh(const int dim)
	{
		if (dim == 2)
		{
			Eigen::MatrixXd V(4, 2);
			V << 0, 0, 1, 0, 1, 1, 0, 1;
			Eigen::MatrixXi F(2, 3);
			F << 0, 1, 2, 0, 2, 3;
			return MeshData(V, F);
		}
		Eigen::MatrixXd V(5, 3);
		V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1;
		Eigen::MatrixXi T(2, 4);
		T << 0, 1, 2, 3, 1, 2, 3, 4;
		return MeshData(V, T);
	}
} // namespace

TEST_CASE("non-conforming refinement leaves the mesh ready for queries", "[ncmesh][mesh_test]")
{
	for (const int dim : {2, 3})
	{
		CAPTURE(dim);

		// refine(): queried right away vs. after an explicit prepare_mesh().
		auto refined = Mesh::create(nc_test_mesh(dim), true);
		auto expected = Mesh::create(nc_test_mesh(dim), true);
		REQUIRE(refined);
		REQUIRE(expected);
		refined->refine(2, 0.5);
		expected->refine(2, 0.5);
		expected->prepare_mesh();
		CHECK(refined->n_elements() == (dim == 2 ? 2 * 16 : 2 * 64));
		CHECK(query(*refined) == query(*expected));

		// refine_elements(): same.
		auto partially = Mesh::create(nc_test_mesh(dim), true);
		auto partially_expected = Mesh::create(nc_test_mesh(dim), true);
		if (dim == 2)
		{
			dynamic_cast<NCMesh2D &>(*partially).refine_elements({0});
			dynamic_cast<NCMesh2D &>(*partially_expected).refine_elements({0});
		}
		else
		{
			dynamic_cast<NCMesh3D &>(*partially).refine_elements({0});
			dynamic_cast<NCMesh3D &>(*partially_expected).refine_elements({0});
		}
		partially_expected->prepare_mesh();
		CHECK(query(*partially) == query(*partially_expected));

		// The low-level refine_element() leaves the id maps stale: queries throw
		// instead of reading out of bounds.
		auto stale = Mesh::create(nc_test_mesh(dim), true);
		if (dim == 2)
			dynamic_cast<NCMesh2D &>(*stale).refine_element(0);
		else
			dynamic_cast<NCMesh3D &>(*stale).refine_element(0);
		CHECK_THROWS(stale->edge_vertex(0, 0));
		stale->prepare_mesh();
		CHECK_NOTHROW(stale->edge_vertex(0, 0));
	}
}

TEST_CASE("non-conforming meshes accept n_refs with selections", "[ncmesh]")
{
	json in_args = R"(
		{
			"materials": {"type": "Laplacian"},
			"geometry": [{
				"mesh": "",
				"n_refs": 1,
				"surface_selection": [
					{"id": 4, "box": [[-2, -2], [0, 2]]},
					{"id": 5, "box": [[0, -2], [2, 2]]}
				]
			}],
			"boundary_conditions": {
				"dirichlet_boundary": [{"id": "all", "value": 0}]
			}
		})"_json;
	in_args["geometry"][0]["mesh"] = std::string(POLYFEM_DATA_DIR) + "/contact/meshes/2D/simple/circle/circle36.obj";

	State state;
	state.init_logger("", spdlog::level::err, spdlog::level::off, false);
	state.init(in_args, true);
	state.load_mesh(true);
	test::VarFormTestAccess::prepare(*state.variational_formulation);

	const auto debug = test::VarFormTestAccess::debug_data(*state.variational_formulation);
	REQUIRE(debug.mesh != nullptr);
	const Mesh &mesh = *debug.mesh;
	REQUIRE(!mesh.is_conforming());
	CHECK(mesh.n_elements() == 4 * 36);

	int n_checked = 0;
	for (int e = 0; e < mesh.n_edges(); ++e)
	{
		if (!mesh.is_boundary_edge(e))
			continue;
		const double x = mesh.edge_barycenter(e)[0];
		if (std::abs(x) < 1e-8)
			continue;
		CHECK(mesh.get_boundary_id(e) == (x < 0 ? 4 : 5));
		++n_checked;
	}
	CHECK(n_checked > 0);
}

TEST_CASE("non-conforming tet meshes answer element queries like conforming ones", "[ncmesh][mesh_test]")
{
	const io::FileSystemIO resources(POLYFEM_DATA_DIR);
	const auto conforming = MeshLoader(resources).load_fem("contact/meshes/3D/simple/sphere/sphere1K.msh", false);
	const auto nonconforming = MeshLoader(resources).load_fem("contact/meshes/3D/simple/sphere/sphere1K.msh", true);
	REQUIRE(conforming->n_elements() == nonconforming->n_elements());
	REQUIRE(conforming->n_vertices() == nonconforming->n_vertices());

	int n_interior = 0;
	for (int c = 0; c < conforming->n_elements(); ++c)
	{
		CAPTURE(c);
		CHECK(nonconforming->is_boundary_element(c) == conforming->is_boundary_element(c));
		n_interior += !conforming->is_boundary_element(c);
		const RowVectorNd expected = conforming->cell_barycenter(c);
		const RowVectorNd actual = nonconforming->cell_barycenter(c);
		CHECK((expected - actual).norm() < 1e-12);
	}
	CHECK(n_interior > 0);
}

TEST_CASE("non-conforming 2D navigation after refinement", "[ncmesh][mesh_test]")
{
	auto mesh = Mesh::create(nc_test_mesh(2), true);
	auto &nc = dynamic_cast<NCMesh2D &>(*mesh);
	nc.refine_elements({0}); // hanging nodes between the two original triangles
	nc.refine(1, 0.5);

	int n_interior = 0;
	for (int f = 0; f < nc.n_faces(); ++f)
	{
		for (int lv = 0; lv < nc.n_face_vertices(f); ++lv)
		{
			const auto idx = nc.get_index_from_face(f, lv);
			const auto other = nc.switch_face(idx);
			if (other.face < 0)
				continue;
			++n_interior;
			CAPTURE(f, lv, other.face);
			REQUIRE(other.face < nc.n_faces());
			// The neighbor across the edge contains that edge, and switching back returns f.
			bool has_edge = false;
			for (int le = 0; le < 3; ++le)
				has_edge |= nc.face_edge(other.face, le) == idx.edge;
			CHECK(has_edge);
			CHECK(nc.switch_face(other).face == f);
		}
	}
	CHECK(n_interior > 0);
}

TEST_CASE("non-conforming refinement after coarsening", "[ncmesh][mesh_test]")
{
	for (const int dim : {2, 3})
	{
		CAPTURE(dim);
		auto twice = Mesh::create(nc_test_mesh(dim), true);
		auto once = Mesh::create(nc_test_mesh(dim), true);
		if (dim == 2)
		{
			auto &nc = dynamic_cast<NCMesh2D &>(*twice);
			nc.refine_element(0);
			nc.coarsen_element(2); // first child of element 0 (full id)
			nc.refine_element(0);
			nc.prepare_mesh();
			dynamic_cast<NCMesh2D &>(*once).refine_elements({0});
			CHECK_THROWS(nc.coarsen_element(1)); // element 1 has no parent
		}
		else
		{
			auto &nc = dynamic_cast<NCMesh3D &>(*twice);
			nc.refine_element(0);
			nc.coarsen_element(2); // first child of element 0 (full id)
			nc.refine_element(0);
			nc.prepare_mesh();
			dynamic_cast<NCMesh3D &>(*once).refine_elements({0});
			CHECK_THROWS(nc.coarsen_element(1)); // element 1 has no parent
		}
		CHECK(query(*twice) == query(*once));
	}
}

TEST_CASE("non-conforming append merges structure and labels", "[ncmesh][mesh_test]")
{
	for (const int dim : {2, 3})
	{
		CAPTURE(dim);
		auto first = Mesh::create(nc_test_mesh(dim), true);
		MeshData shifted = nc_test_mesh(dim);
		shifted.vertices.col(0).array() += 5;
		auto second = Mesh::create(shifted, true);

		// First mesh: refined with hanging nodes, labeled.
		if (dim == 2)
			dynamic_cast<NCMesh2D &>(*first).refine_elements({0});
		else
			dynamic_cast<NCMesh3D &>(*first).refine_elements({0});
		first->set_body_ids(std::vector<int>(first->n_elements(), 7));
		first->compute_boundary_ids([](const size_t, const std::vector<int> &, const RowVectorNd &, bool boundary) { return boundary ? 3 : -1; });
		const int n_first_elements = first->n_elements(), n_first_vertices = first->n_vertices();
		const int n_first_boundary = first->n_boundary_elements();
		std::vector<int> first_boundary_ids;
		for (int b = 0; b < n_first_boundary; ++b)
			first_boundary_ids.push_back(first->get_boundary_id(b));

		// Second mesh: body ids only.
		second->set_body_ids(std::vector<int>(second->n_elements(), 9));
		const int n_second_elements = second->n_elements(), n_second_vertices = second->n_vertices();

		first->append(second);
		REQUIRE(first->n_elements() == n_first_elements + n_second_elements);
		REQUIRE(first->n_vertices() == n_first_vertices + n_second_vertices);
		for (int e = 0; e < first->n_elements(); ++e)
			CHECK(first->get_body_id(e) == (e < n_first_elements ? 7 : 9));
		CHECK(first->get_body_ids().size() == first->n_elements());

		// Boundary ids: kept for the first mesh, defaults for the second one.
		int n_checked = 0;
		for (int b = 0; b < first->n_boundary_elements(); ++b)
		{
			const RowVectorNd center = dim == 2 ? first->edge_barycenter(b) : first->face_barycenter(b);
			const bool boundary = dim == 2 ? first->is_boundary_edge(b) : first->is_boundary_face(b);
			if (center[0] < 2)
				CHECK(first->get_boundary_id(b) == (boundary ? 3 : -1));
			else
				CHECK(first->get_boundary_id(b) == (boundary ? std::numeric_limits<int>::max() : -1));
			++n_checked;
		}
		CHECK(n_checked == first->n_boundary_elements());
		CHECK(std::vector<int>(first_boundary_ids.begin(), first_boundary_ids.end()).size() == n_first_boundary);

		// The second mesh's elements only touch its own vertices.
		for (int e = n_first_elements; e < first->n_elements(); ++e)
			for (const int v : first->element_vertices(e))
				CHECK(first->point(v)[0] >= 5 - 1e-12);
	}
}

TEST_CASE("stored labels follow unreferenced input vertices", "[ncmesh][mesh_test]")
{
	// Vertex 1 is not used by any element.
	Eigen::MatrixXd vertices(4, 2);
	vertices << 0, 0, 9, 9, 1, 0, 0, 1;
	Eigen::MatrixXi cells(1, 3);
	cells << 0, 2, 3;
	MeshData data(vertices, cells);
	data.node_ids = {10, 11, 12, 13};
	// The last side uses the unreferenced vertex: it is not an edge of the mesh.
	data.boundary_elements = {{0, 2}, {2, 3}, {3, 0}, {1, 2}};
	data.boundary_ids = {5, 6, 7, 9};

	for (const bool non_conforming : {false, true})
	{
		CAPTURE(non_conforming);
		const auto mesh = Mesh::create(data, non_conforming);
		for (int v = 0; v < mesh->n_vertices(); ++v)
		{
			const RowVectorNd p = mesh->point(v);
			int input = -1;
			for (int i = 0; i < vertices.rows(); ++i)
				if (vertices.row(i) == p)
					input = i;
			REQUIRE(input >= 0);
			CHECK(mesh->get_node_id(v) == data.node_ids[input]);
		}
		for (int e = 0; e < mesh->n_edges(); ++e)
		{
			const RowVectorNd c = mesh->edge_barycenter(e);
			const int expected = c[1] == 0 ? 5 : (c[0] == 0 ? 7 : 6);
			CHECK(mesh->get_boundary_id(e) == expected);
		}
	}
}

TEST_CASE("stale non-conforming meshes refuse every query", "[ncmesh][mesh_test]")
{
	for (const int dim : {2, 3})
	{
		CAPTURE(dim);
		auto mesh = Mesh::create(nc_test_mesh(dim), true);
		if (dim == 2)
			dynamic_cast<NCMesh2D &>(*mesh).refine_element(0);
		else
			dynamic_cast<NCMesh3D &>(*mesh).refine_element(0);
		// refine_element is low level: the id maps are stale until prepare_mesh().
		CHECK_THROWS(mesh->point(0));
		CHECK_THROWS(mesh->edge_vertex(0, 0));
		CHECK_THROWS(mesh->get_body_id(0));
		if (dim == 3)
			CHECK_THROWS(mesh->face_vertex(0, 0));
		mesh->prepare_mesh();
		CHECK_NOTHROW(mesh->point(0));
	}
}

TEST_CASE("non-conforming 3D refinement keeps the label arrays consistent", "[ncmesh][mesh_test]")
{
	auto mesh = Mesh::create(nc_test_mesh(3), true);
	mesh->set_body_ids({7, 8});
	mesh->compute_boundary_ids([](const size_t, const std::vector<int> &, const RowVectorNd &, bool boundary) { return boundary ? 3 : -1; });
	mesh->refine(1, 0.5);
	REQUIRE(mesh->n_cells() == 16);
	REQUIRE(mesh->get_body_ids().size() == mesh->n_cells());
	for (int c = 0; c < mesh->n_cells(); ++c)
		CHECK(mesh->get_body_ids()[c] == mesh->get_body_id(c));
	const MeshData data = mesh->to_mesh_data();
	REQUIRE(data.boundary_ids.size() == mesh->n_faces());
	for (int f = 0; f < mesh->n_faces(); ++f)
	{
		CHECK(mesh->get_boundary_id(f) == (mesh->is_boundary_face(f) ? 3 : -1));
		CHECK(data.boundary_ids[f] == mesh->get_boundary_id(f));
	}
}

TEST_CASE("non-conforming append keeps the refinement hierarchy of the appended mesh", "[ncmesh][mesh_test]")
{
	for (const int dim : {2, 3})
	{
		CAPTURE(dim);
		auto first = Mesh::create(nc_test_mesh(dim), true);
		first->set_body_ids(std::vector<int>(first->n_elements(), 7));
		// The appended mesh is refined (hanging nodes) and has no body ids.
		MeshData shifted = nc_test_mesh(dim);
		shifted.vertices.col(0).array() += 5;
		auto second = Mesh::create(shifted, true);
		if (dim == 2)
			dynamic_cast<NCMesh2D &>(*second).refine_elements({0});
		else
			dynamic_cast<NCMesh3D &>(*second).refine_elements({0});
		const int n_first = first->n_elements(), n_second = second->n_elements();

		first->append(second);
		REQUIRE(first->n_elements() == n_first + n_second);
		for (int e = 0; e < first->n_elements(); ++e)
			CHECK(first->get_body_id(e) == (e < n_first ? 7 : 0));

		// The children of the appended mesh still know their parent: coarsening
		// one of them restores the unrefined appended mesh.
		const int n_children = dim == 2 ? 4 : 8;
		// Full ids: the 2 elements of the first mesh, the 2 roots of the appended
		// one, then the children of its first root.
		const int first_child = 4;
		if (dim == 2)
		{
			auto &nc = dynamic_cast<NCMesh2D &>(*first);
			nc.coarsen_element(first_child);
			nc.prepare_mesh();
		}
		else
		{
			auto &nc = dynamic_cast<NCMesh3D &>(*first);
			nc.coarsen_element(first_child);
			nc.prepare_mesh();
		}
		CHECK(first->n_elements() == n_first + n_second - n_children + 1);
		// Refining again uses the appended midpoints.
		const int n_vertices = first->n_vertices();
		if (dim == 2)
			dynamic_cast<NCMesh2D &>(*first).refine_elements({n_first});
		else
			dynamic_cast<NCMesh3D &>(*first).refine_elements({n_first});
		CHECK(first->n_elements() == n_first + n_second);
		CHECK(first->n_vertices() == n_vertices + (dim == 2 ? 3 : 6));
	}
}
