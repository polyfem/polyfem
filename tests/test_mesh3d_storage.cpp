#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3DStorage.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
#include <polyfem/mesh/MeshData.hpp>
#include <polyfem/io/ResourceIO.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>

using namespace polyfem;
using namespace polyfem::mesh;

TEST_CASE("jagged arrays store rows contiguously", "[mesh_test][mesh3d_storage]")
{
	JaggedArray<uint32_t> rows;
	CHECK(rows.size() == 0);
	rows.push_back({1, 2, 3});
	rows.push_back_filled(0, 0);
	rows.push_back(std::vector<int>{4, 5});
	REQUIRE(rows.size() == 3);
	CHECK(rows.n_entries() == 5);
	CHECK(rows.row_size(1) == 0);
	CHECK(rows[0].to_vector() == std::vector<uint32_t>{1, 2, 3});
	CHECK(rows[2].to_vector() == std::vector<uint32_t>{4, 5});

	rows[0][1] = 7;
	std::reverse(rows[2].begin(), rows[2].end());
	CHECK(rows[0].to_vector() == std::vector<uint32_t>{1, 7, 3});
	CHECK(rows[2].to_vector() == std::vector<uint32_t>{5, 4});

	JaggedArray<uint32_t> other;
	other.push_back({0, 1});
	rows.append(other, 10);
	REQUIRE(rows.size() == 4);
	CHECK(rows[3].to_vector() == std::vector<uint32_t>{10, 11});

	rows.assign_sizes({2, 0, 1}, 9);
	REQUIRE(rows.size() == 3);
	CHECK(rows[0].to_vector() == std::vector<uint32_t>{9, 9});
	CHECK(rows[1].empty());
	CHECK(rows[2].to_vector() == std::vector<uint32_t>{9});
}

namespace
{
	void check_storage_invariants(Mesh &mesh)
	{
		const Mesh3DStorage &M = dynamic_cast<CMesh3D &>(mesh).mesh_storge();
		REQUIRE(M.n_cells() == mesh.n_cells());
		CHECK(M.memory() > 0);

		const auto sorted = [](const auto &row) { return std::is_sorted(row.begin(), row.end()); };

		// Edges sorted by (min, max).
		for (int e = 0; e < M.n_edges(); ++e)
		{
			CHECK(M.edge_vs[e][0] < M.edge_vs[e][1]);
			if (e > 0)
				CHECK(M.edge_vs[e - 1] < M.edge_vs[e]);
			CHECK(sorted(M.edge_fs[e]));
			CHECK(sorted(M.edge_hs[e]));
		}
		// Face edges aligned with the face vertices; incident cells and slots.
		for (int f = 0; f < M.n_faces(); ++f)
		{
			const auto vs = M.face_vs[f];
			REQUIRE(M.face_es.row_size(f) == vs.size());
			for (int j = 0; j < vs.size(); ++j)
			{
				const auto &ev = M.edge_vs[M.face_es[f][j]];
				const uint32_t a = vs[j], b = vs[(j + 1) % vs.size()];
				CHECK(((ev[0] == a && ev[1] == b) || (ev[0] == b && ev[1] == a)));
			}
			REQUIRE(M.face_hs.row_size(f) >= 1);
			REQUIRE(M.face_hs.row_size(f) <= 2);
			CHECK(bool(M.face_boundary[f]) == (M.face_hs.row_size(f) == 1));
			CHECK(sorted(M.face_hs[f]));
			for (int k = 0; k < M.face_hs.row_size(f); ++k)
			{
				CHECK(M.cell_fs[M.face_hs[f][k]][M.face_hs_slot[f][k]] == f);
				CHECK(M.FH(k, f) == int(M.face_hs[f][k]));
			}
		}
		// Vertex incidence: sorted, and vertex_vs[v][k] is the other end of vertex_es[v][k].
		for (int v = 0; v < M.n_vertices(); ++v)
		{
			CHECK(sorted(M.vertex_es[v]));
			CHECK(sorted(M.vertex_fs[v]));
			CHECK(sorted(M.vertex_hs[v]));
			REQUIRE(M.vertex_vs.row_size(v) == M.vertex_es.row_size(v));
			for (int k = 0; k < M.vertex_es.row_size(v); ++k)
			{
				const auto &ev = M.edge_vs[M.vertex_es[v][k]];
				CHECK(M.vertex_vs[v][k] == (ev[0] == v ? ev[1] : ev[0]));
			}
		}
		// Cells: the faces' vertices, sorted edges, and every vertex knows the cell.
		for (int c = 0; c < M.n_cells(); ++c)
		{
			CHECK(sorted(M.cell_es[c]));
			REQUIRE(M.cell_fs_flag.row_size(c) == M.cell_fs.row_size(c));
			std::vector<uint32_t> face_vertices;
			for (const auto f : M.cell_fs[c])
				face_vertices.insert(face_vertices.end(), M.face_vs[f].begin(), M.face_vs[f].end());
			std::sort(face_vertices.begin(), face_vertices.end());
			face_vertices.erase(std::unique(face_vertices.begin(), face_vertices.end()), face_vertices.end());
			std::vector<uint32_t> cell_vertices = M.cell_vs[c].to_vector();
			std::sort(cell_vertices.begin(), cell_vertices.end());
			CHECK(cell_vertices == face_vertices);
			for (const auto v : M.cell_vs[c])
				CHECK(std::binary_search(M.vertex_hs[v].begin(), M.vertex_hs[v].end(), uint32_t(c)));
		}
	}

	/// Every flag of every cell switches back, and the lookups by vertices find it.
	void check_navigation(const Mesh &mesh)
	{
		const auto &m = dynamic_cast<const Mesh3D &>(mesh);
		for (int c = 0; c < m.n_cells(); ++c)
		{
			CHECK(m.get_index_from_element(c).element == c);
			CHECK(m.n_cell_edges(c) > 0);
			for (int le = 0; le < m.n_cell_edges(c); ++le)
				CHECK(m.cell_edge(c, le) < m.n_edges());
			for (int lf = 0; lf < m.n_cell_faces(c); ++lf)
			{
				const int f = m.cell_face(c, lf);
				for (int lv = 0; lv < m.n_face_vertices(f); ++lv)
				{
					const auto idx = m.get_index_from_element(c, lf, lv);
					REQUIRE(idx.element == c);
					REQUIRE(idx.face == f);
					const auto sv = m.switch_vertex(idx);
					CHECK(m.switch_vertex(sv).vertex == idx.vertex);
					CHECK(sv.edge == idx.edge);
					const auto se = m.switch_edge(idx);
					CHECK(m.switch_edge(se).edge == idx.edge);
					const auto sf = m.switch_face(idx);
					CHECK(sf.edge == idx.edge);
					CHECK(m.switch_face(sf).face == idx.face);
					const auto sc = m.switch_element(idx);
					CHECK((sc.element == -1) == m.is_boundary_face(f));
					if (sc.element >= 0)
						CHECK(m.switch_element(sc).element == c);

					const auto by_edge = m.get_index_from_element_edge(c, idx.vertex, sv.vertex);
					CHECK(by_edge.edge == idx.edge);
					CHECK(by_edge.vertex == idx.vertex);
					if (m.n_face_vertices(f) == 3)
					{
						const auto by_face = m.get_index_from_element_face(c, idx.vertex, sv.vertex, m.switch_vertex(se).vertex);
						CHECK(by_face.face == f);
					}
				}
				if (m.n_face_vertices(f) == 4)
					CHECK(m.quad_area(f) > 0);
			}
		}
		for (int e = 0; e < m.n_edges(); ++e)
		{
			const RowVectorNd bary = (m.point(m.edge_vertex(e, 0)) + m.point(m.edge_vertex(e, 1))) / 2;
			CHECK((m.edge_barycenter(e) - bary).norm() < 1e-12);
			std::vector<int> cells;
			m.get_edge_elements_neighs(e, cells);
			CHECK(!cells.empty());
		}
		for (int v = 0; v < m.n_vertices(); ++v)
		{
			std::vector<int> cells;
			m.get_vertex_elements_neighs(v, cells);
			CHECK(cells.size() == m.vertex_neighs(v).size());
		}
	}
} // namespace

TEST_CASE("3D mesh storage satisfies its documented invariants", "[mesh_test][mesh3d_storage]")
{
	const io::FileSystemIO resources(POLYFEM_DATA_DIR);
	for (const std::string path : {"contact/meshes/3D/simple/sphere/sphere1K.msh", "quad_test/hex.HYBRID", "contact/examples/3D/hex_tet/hexcube.msh", "contact/examples/3D/hex_tet/cube.msh"})
	{
		CAPTURE(path);
		auto mesh = MeshLoader(resources).load_fem(path, false);
		check_storage_invariants(*mesh);
		check_navigation(*mesh);

		// Red refinement (tets) or Catmull-Clark/polar refinement (the others).
		const int n_cells = mesh->n_cells();
		mesh->refine(1, 0.5);
		CHECK(mesh->n_cells() > n_cells);
		check_storage_invariants(*mesh);
		check_navigation(*mesh);
	}
}

TEST_CASE("3D mesh storage of prisms and pyramids", "[mesh_test][mesh3d_storage]")
{
	// A prism and, next to it, a pyramid.
	Eigen::MatrixXd vertices(11, 3);
	vertices << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
		3, 0, 0, 4, 0, 0, 4, 1, 0, 3, 1, 0, 3.5, 0.5, 1;
	Eigen::MatrixXi cells(2, 6);
	cells << 0, 1, 2, 3, 4, 5,
		6, 7, 8, 9, 10, -1;
	auto mesh = Mesh::create(MeshData(vertices, cells));
	REQUIRE(mesh->n_cells() == 2);
	CHECK(mesh->is_prism(0));
	CHECK(mesh->is_pyramid(1));
	check_storage_invariants(*mesh);
	check_navigation(*mesh);

	// Out-of-range local ids wrap around.
	const auto &m = dynamic_cast<const Mesh3D &>(*mesh);
	const auto idx = m.get_index_from_element(0, 1, 2);
	const auto wrapped = m.get_index_from_element(2, 1 + m.n_cell_faces(0), 2 + m.n_face_vertices(idx.face));
	CHECK(wrapped.face == idx.face);
	CHECK(wrapped.vertex == idx.vertex);

	// Polar refinement of the non-hex cells.
	mesh->refine(1, 0.5);
	CHECK(mesh->n_cells() > 2);
	check_storage_invariants(*mesh);
	check_navigation(*mesh);
}

TEST_CASE("3D mesh storage of hex grids and hex interfaces", "[mesh_test][mesh3d_storage]")
{
	// A 3 x 3 x 3 grid of hexes: the center one is a regular interior cube.
	const int n = 3;
	Eigen::MatrixXd vertices((n + 1) * (n + 1) * (n + 1), 3);
	const auto vid = [&](int i, int j, int k) { return i + (n + 1) * (j + (n + 1) * k); };
	for (int k = 0; k <= n; ++k)
		for (int j = 0; j <= n; ++j)
			for (int i = 0; i <= n; ++i)
				vertices.row(vid(i, j, k)) << i, j, k;
	Eigen::MatrixXi cells(n * n * n, 8);
	for (int k = 0, c = 0; k < n; ++k)
		for (int j = 0; j < n; ++j)
			for (int i = 0; i < n; ++i, ++c)
				cells.row(c) << vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i, j + 1, k),
					vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1);
	auto grid = Mesh::create(MeshData(vertices, cells));
	check_storage_invariants(*grid);
	check_navigation(*grid);
	CHECK(grid->elements_tag()[13] == ElementType::REGULAR_INTERIOR_CUBE);
	CHECK(grid->is_spline_compatible(13));

	// Pyramids on the top and bottom faces of a hex (every vertex of the hex
	// touches one): the hex is an interface cube. Appending a tet to the grid
	// (another type) gives a hybrid mesh.
	Eigen::MatrixXd hp_vertices(10, 3);
	hp_vertices << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0.5, 0.5, 2, 0.5, 0.5, -1;
	Eigen::MatrixXi hp_cells(3, 8);
	hp_cells << 0, 1, 2, 3, 4, 5, 6, 7,
		4, 5, 6, 7, 8, -1, -1, -1,
		0, 3, 2, 1, 9, -1, -1, -1;
	auto hex_pyramid = Mesh::create(MeshData(hp_vertices, hp_cells));
	CHECK(hex_pyramid->elements_tag()[0] == ElementType::INTERFACE_CUBE);
	CHECK(hex_pyramid->is_pyramid(1));
	CHECK(hex_pyramid->is_pyramid(2));
	check_storage_invariants(*hex_pyramid);

	Eigen::MatrixXd tet_vertices(4, 3);
	tet_vertices << 5, 0, 0, 6, 0, 0, 5, 1, 0, 5, 0, 1;
	Eigen::MatrixXi tet(1, 4);
	tet << 0, 1, 2, 3;
	grid->append(Mesh::create(MeshData(tet_vertices, tet)));
	CHECK(grid->n_cells() == n * n * n + 1);
	CHECK(grid->is_simplex(n * n * n));
	check_storage_invariants(*grid);
	check_navigation(*grid);
}
