#pragma once

#include <array>
#include <vector>

namespace polyfem::mesh
{
	/// Standard volume cell types with a fixed local facet layout.
	enum class StandardCellType
	{
		TET,
		HEX,
		PRISM,
		PYRAMID
	};

	/// Local facets of a standard cell: for each facet, its local vertices,
	/// ordered so that the facet normal points outside the cell.
	struct CellDescriptor
	{
		int n_vertices;
		int n_facets;
		int n_facet_vertices[6];
		int facet_vertex[6][4];
	};

	/// Local vertex and facet conventions of each cell type. These are the
	/// conventions of geogram's MeshCellDescriptors, which PolyFEM used to build
	/// its volume meshes (local vertex order of hexes, facet order and facet
	/// vertex order all follow them).
	const CellDescriptor &cell_descriptor(StandardCellType type);

	/// A standard cell referencing global vertex indices.
	struct StandardCell
	{
		explicit StandardCell(const StandardCellType type)
			: type(type), desc(&cell_descriptor(type))
		{
			vertices.fill(-1);
		}

		StandardCellType type;
		const CellDescriptor *desc;
		std::array<int, 8> vertices;

		const CellDescriptor &descriptor() const { return *desc; }
		int n_facets() const { return descriptor().n_facets; }
		int n_facet_vertices(const int lf) const { return descriptor().n_facet_vertices[lf]; }
		int facet_vertex(const int lf, const int lv) const { return vertices[descriptor().facet_vertex[lf][lv]]; }
	};

	///
	/// @brief      Computes cell-to-cell adjacency across facets.
	///
	/// Two cell facets are glued when they have the same vertices in opposite
	/// orientation. The traversal order (and therefore the choice made for
	/// non-manifold facets) reproduces geogram's MeshCells::connect(): tet meshes
	/// chain tet corners around vertices, other meshes chain cells around
	/// vertices, and candidates are visited from the most recently created cell
	/// to the oldest one.
	///
	/// A quadrilateral boundary facet glued to two or more triangular facets
	/// (a non-conforming quad/triangle interface) is rejected: geogram used to
	/// insert artificial "connector" cells there, which PolyFEM cannot use.
	///
	/// @param[in]  cells       Input cells
	/// @param[in]  n_vertices  Number of vertices referenced by the cells
	///
	/// @return     For each cell and local facet, the adjacent cell or -1.
	///
	std::vector<std::array<int, 6>> connect_cells(const std::vector<StandardCell> &cells, int n_vertices);
} // namespace polyfem::mesh
