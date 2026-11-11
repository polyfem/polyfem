#include "CellConnectivity.hpp"

#include <polyfem/utils/Logger.hpp>

#include <algorithm>
#include <cassert>
#include <utility>

namespace polyfem::mesh
{
	namespace
	{
		// Local facets of the standard cells. The tables are geogram's
		// MeshCellDescriptors (https://github.com/BrunoLevy/geogram,
		// src/lib/geogram/mesh/mesh.cpp, BSD-3-Clause license).
		const CellDescriptor tet_descriptor = {
			4,            // n_vertices
			4,            // n_facets
			{3, 3, 3, 3}, // n_facet_vertices
			{             // facet_vertex
			 {1, 3, 2},
			 {0, 2, 3},
			 {3, 1, 0},
			 {0, 1, 2}}};

		const CellDescriptor hex_descriptor = {
			8,                  // n_vertices
			6,                  // n_facets
			{4, 4, 4, 4, 4, 4}, // n_facet_vertices
			{                   // facet_vertex
			 {0, 2, 6, 4},
			 {3, 1, 5, 7},
			 {1, 0, 4, 5},
			 {2, 3, 7, 6},
			 {1, 3, 2, 0},
			 {4, 6, 7, 5}}};

		const CellDescriptor prism_descriptor = {
			6,               // n_vertices
			5,               // n_facets
			{3, 3, 4, 4, 4}, // n_facet_vertices
			{                // facet_vertex
			 {0, 1, 2},
			 {3, 5, 4},
			 {0, 3, 4, 1},
			 {0, 2, 5, 3},
			 {1, 4, 5, 2}}};

		const CellDescriptor pyramid_descriptor = {
			5,               // n_vertices
			5,               // n_facets
			{4, 3, 3, 3, 3}, // n_facet_vertices
			{                // facet_vertex
			 {0, 1, 2, 3},
			 {0, 4, 1},
			 {0, 3, 4},
			 {2, 4, 3},
			 {2, 1, 4}}};

		bool triangles_equal(int v1, int v2, int v3, int w1, int w2, int w3)
		{
			return (v1 == w1 && v2 == w2 && v3 == w3)
				   || (v1 == w2 && v2 == w3 && v3 == w1)
				   || (v1 == w3 && v2 == w1 && v3 == w2);
		}

		/// Local facet of tet t made of (v1, v2, v3) up to a circular permutation, or -1.
		int find_tet_facet(const std::array<int, 4> &t, int v1, int v2, int v3)
		{
			for (int lf = 0; lf < 4; ++lf)
			{
				const int *fv = tet_descriptor.facet_vertex[lf];
				if (triangles_equal(v1, v2, v3, t[fv[0]], t[fv[1]], t[fv[2]]))
					return lf;
			}
			return -1;
		}

		/// Whether facet f1 of c1 and facet f2 of c2 have the same vertices in opposite orientation.
		bool facets_match(const StandardCell &c1, int f1, const StandardCell &c2, int f2)
		{
			const int nbv = c1.n_facet_vertices(f1);
			if (c2.n_facet_vertices(f2) != nbv)
				return false;
			for (int offset = 0; offset < nbv; ++offset)
			{
				bool match = true;
				for (int v1 = 0; v1 < nbv; ++v1)
				{
					const int v2 = (nbv - v1 + offset) % nbv;
					if (c1.facet_vertex(f1, v1) != c2.facet_vertex(f2, v2))
					{
						match = false;
						break;
					}
				}
				if (match)
					return true;
			}
			return false;
		}

		/// Local facet of c1 glued to facet f2 of c2, or -1.
		int find_cell_facet(const StandardCell &c1, const StandardCell &c2, int f2)
		{
			for (int f1 = 0; f1 < c1.n_facets(); ++f1)
			{
				if (facets_match(c1, f1, c2, f2))
					return f1;
			}
			return -1;
		}

		int find_cell_vertex(const StandardCell &c, int v)
		{
			for (int lv = 0; lv < c.descriptor().n_vertices; ++lv)
			{
				if (c.vertices[lv] == v)
					return lv;
			}
			return -1;
		}

		/// Whether triangular facet lf1 of c1 covers half of quad facet lf2 of c2 (opposite orientation).
		bool triangular_facet_matches_quad_facet(const StandardCell &c1, int lf1, const StandardCell &c2, int lf2)
		{
			const int v1 = c1.facet_vertex(lf1, 0);
			const int v2 = c1.facet_vertex(lf1, 1);
			const int v3 = c1.facet_vertex(lf1, 2);
			const int w1 = c2.facet_vertex(lf2, 0);
			const int w2 = c2.facet_vertex(lf2, 1);
			const int w3 = c2.facet_vertex(lf2, 2);
			const int w4 = c2.facet_vertex(lf2, 3);
			return triangles_equal(v1, v2, v3, w4, w3, w2)
				   || triangles_equal(v1, v2, v3, w3, w2, w1)
				   || triangles_equal(v1, v2, v3, w2, w1, w4)
				   || triangles_equal(v1, v2, v3, w1, w4, w3);
		}

		void connect_tets(const std::vector<StandardCell> &cells, const int n_vertices, std::vector<std::array<int, 6>> &adjacent)
		{
			// Compact copy of the tets (4 vertices each) for cache-friendly access.
			std::vector<std::array<int, 4>> tets(cells.size());
			for (int t = 0; t < cells.size(); ++t)
				for (int lv = 0; lv < 4; ++lv)
					tets[t][lv] = cells[t].vertices[lv];

			// Chain tet corners around vertices.
			std::vector<int> next_tet_corner_around_vertex(4 * tets.size(), -1);
			std::vector<int> v2c(n_vertices, -1);
			for (int t = 0; t < tets.size(); ++t)
			{
				for (int lv = 0; lv < 4; ++lv)
				{
					const int v = tets[t][lv];
					next_tet_corner_around_vertex[4 * t + lv] = v2c[v];
					v2c[v] = 4 * t + lv;
				}
			}

			for (int t1 = 0; t1 < tets.size(); ++t1)
			{
				for (int lf1 = 0; lf1 < 4; ++lf1)
				{
					if (adjacent[t1][lf1] != -1)
						continue;
					const int *fv = tet_descriptor.facet_vertex[lf1];
					const int v1 = tets[t1][fv[0]];
					const int v2 = tets[t1][fv[1]];
					const int v3 = tets[t1][fv[2]];
					for (int c2 = v2c[v1]; c2 != -1; c2 = next_tet_corner_around_vertex[c2])
					{
						const int t2 = c2 / 4;
						const int lf2 = find_tet_facet(tets[t2], v3, v2, v1);
						if (lf2 != -1)
						{
							adjacent[t1][lf1] = t2;
							adjacent[t2][lf2] = t1;
							break;
						}
					}
				}
			}
		}
	} // namespace

	const CellDescriptor &cell_descriptor(const StandardCellType type)
	{
		switch (type)
		{
		case StandardCellType::TET:
			return tet_descriptor;
		case StandardCellType::HEX:
			return hex_descriptor;
		case StandardCellType::PRISM:
			return prism_descriptor;
		case StandardCellType::PYRAMID:
			return pyramid_descriptor;
		}
		assert(false);
		return tet_descriptor;
	}

	std::vector<std::array<int, 6>> connect_cells(const std::vector<StandardCell> &cells, const int n_vertices)
	{
		std::array<int, 6> unconnected;
		unconnected.fill(-1);
		std::vector<std::array<int, 6>> adjacent(cells.size(), unconnected);

		const bool all_tets = std::all_of(cells.begin(), cells.end(), [](const StandardCell &c) {
			return c.type == StandardCellType::TET;
		});
		if (all_tets)
		{
			connect_tets(cells, n_vertices, adjacent);
			return adjacent;
		}

		// Chain cells around vertices.
		std::vector<std::array<int, 8>> next_cell_around_vertex(cells.size());
		std::vector<int> v2cell(n_vertices, -1);
		for (int c = 0; c < cells.size(); ++c)
		{
			for (int lv = 0; lv < cells[c].descriptor().n_vertices; ++lv)
			{
				const int v = cells[c].vertices[lv];
				next_cell_around_vertex[c][lv] = v2cell[v];
				v2cell[v] = c;
			}
		}
		const auto next_cell = [&](const int c, const int v) {
			return next_cell_around_vertex[c][find_cell_vertex(cells[c], v)];
		};

		// Glue each unconnected facet (c1, lf1) to the first matching facet of
		// a cell incident to its first vertex.
		for (int c1 = 0; c1 < cells.size(); ++c1)
		{
			for (int lf1 = 0; lf1 < cells[c1].n_facets(); ++lf1)
			{
				if (adjacent[c1][lf1] != -1)
					continue;
				const int v1 = cells[c1].facet_vertex(lf1, 0);
				for (int c2 = v2cell[v1]; c2 != -1; c2 = next_cell(c2, v1))
				{
					const int lf2 = find_cell_facet(cells[c2], cells[c1], lf1);
					if (lf2 != -1)
					{
						adjacent[c1][lf1] = c2;
						adjacent[c2][lf2] = c1;
						break;
					}
				}
			}
		}

		// Reject non-conforming quad/triangle interfaces.
		std::vector<std::pair<int, int>> matches;
		for (int c1 = 0; c1 < cells.size(); ++c1)
		{
			if (cells[c1].type == StandardCellType::TET)
				continue;
			for (int lf1 = 0; lf1 < cells[c1].n_facets(); ++lf1)
			{
				if (cells[c1].n_facet_vertices(lf1) != 4 || adjacent[c1][lf1] != -1)
					continue;

				matches.clear();
				for (int lv1 = 0; lv1 < 4; ++lv1)
				{
					const int v1 = cells[c1].facet_vertex(lf1, lv1);
					for (int c2 = v2cell[v1]; c2 != -1; c2 = next_cell(c2, v1))
					{
						if (c2 == c1 || cells[c2].type == StandardCellType::HEX)
							continue;
						for (int lf2 = 0; lf2 < cells[c2].n_facets(); ++lf2)
						{
							if (cells[c2].n_facet_vertices(lf2) == 3
								&& triangular_facet_matches_quad_facet(cells[c2], lf2, cells[c1], lf1))
								matches.emplace_back(c2, lf2);
						}
					}
				}
				std::sort(matches.begin(), matches.end());
				matches.erase(std::unique(matches.begin(), matches.end()), matches.end());
				if (matches.size() >= 2)
					log_and_throw_error(
						"Cell {} facet {}: a quadrilateral facet is glued to {} triangular facets; "
						"non-conforming quad/triangle interfaces are not supported.",
						c1, lf1, matches.size());
			}
		}

		return adjacent;
	}
} // namespace polyfem::mesh
