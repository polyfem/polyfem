#include <polyfem/mesh/mesh3D/CMesh3D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3DStorage.hpp>
#include <polyfem/mesh/MeshLoader.hpp>
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

TEST_CASE("3D mesh storage satisfies its documented invariants", "[mesh_test][mesh3d_storage]")
{
	const io::FileSystemIO resources(POLYFEM_DATA_DIR);
	for (const std::string path : {"contact/meshes/3D/simple/sphere/sphere1K.msh", "quad_test/hex.HYBRID", "contact/examples/3D/hex_tet/hexcube.msh"})
	{
		CAPTURE(path);
		auto mesh = MeshLoader(resources).load_fem(path, false);
		const Mesh3DStorage &M = dynamic_cast<CMesh3D &>(*mesh).mesh_storge();
		REQUIRE(M.n_cells() == mesh->n_cells());

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
}
