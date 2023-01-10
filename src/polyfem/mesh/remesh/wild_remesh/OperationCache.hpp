#pragma once

#include <polyfem/mesh/remesh/WildRemesher.hpp>

#include <wmtk/utils/TupleUtils.hpp>

namespace polyfem::mesh
{
	class WildTriRemesher;
	class WildTetRemesher;

	enum class CollapseEdgeTo
	{
		V0,
		V1,
		MIDPOINT,
		ILLEGAL
	};

	class TriOperationCache
	{
	public:
		using VertexAttributes = WildRemesher<wmtk::TriMesh>::VertexAttributes;
		using EdgeAttributes = WildRemesher<wmtk::TriMesh>::BoundaryAttributes;
		using FaceAttributes = WildRemesher<wmtk::TriMesh>::ElementAttributes;
		using Tuple = wmtk::TriMesh::Tuple;

		/// @brief Construct a local mesh as an n-ring around a vertex.
		static TriOperationCache split_edge(WildTriRemesher &m, const Tuple &t);
		static TriOperationCache swap_edge(WildTriRemesher &m, const Tuple &t);
		static TriOperationCache collapse_edge(WildTriRemesher &m, const Tuple &t);

		const std::pair<size_t, VertexAttributes> &v0() const { return m_v0; }
		const std::pair<size_t, VertexAttributes> &v1() const { return m_v1; }
		const Remesher::EdgeMap<EdgeAttributes> &edges() const { return m_edges; }
		const std::vector<FaceAttributes> &faces() const { return m_faces; }

		double local_energy = std::numeric_limits<double>::quiet_NaN();
		CollapseEdgeTo collapse_to = CollapseEdgeTo::ILLEGAL;

	protected:
		std::pair<size_t, VertexAttributes> m_v0;
		std::pair<size_t, VertexAttributes> m_v1;
		Remesher::EdgeMap<EdgeAttributes> m_edges;
		std::vector<FaceAttributes> m_faces;
	};

	class TetOperationCache
	{
	public:
		using VertexAttributes = WildRemesher<wmtk::TetMesh>::VertexAttributes;
		using EdgeAttributes = WildRemesher<wmtk::TetMesh>::EdgeAttributes;
		using FaceAttributes = WildRemesher<wmtk::TetMesh>::BoundaryAttributes;
		using TetAttributes = WildRemesher<wmtk::TetMesh>::ElementAttributes;
		using Tuple = wmtk::TetMesh::Tuple;

		static TetOperationCache split_edge(WildTetRemesher &m, const Tuple &t);
		// static TetOperationCache swap_32(WildTetRemesher &m, const Tuple &t);
		// static TetOperationCache swap_23(WildTetRemesher &m, const Tuple &t);
		// static TetOperationCache swap_44(WildTetRemesher &m, const Tuple &t);
		// static TetOperationCache collapse_edge(WildTetRemesher &m, const Tuple &t);

		const std::pair<size_t, VertexAttributes> &v0() const { return m_v0; }
		const std::pair<size_t, VertexAttributes> &v1() const { return m_v1; }
		const std::optional<std::pair<size_t, VertexAttributes>> &v2() const { return m_v2; }
		const Remesher::EdgeMap<EdgeAttributes> &edges() const { return m_edges; }
		const Remesher::FaceMap<FaceAttributes> &faces() const { return m_faces; }
		const Remesher::TetMap<TetAttributes> &tets() const { return m_tets; }

		double local_energy = std::numeric_limits<double>::quiet_NaN();
		CollapseEdgeTo collapse_to = CollapseEdgeTo::ILLEGAL;

	protected:
		std::pair<size_t, VertexAttributes> m_v0;
		std::pair<size_t, VertexAttributes> m_v1;
		std::optional<std::pair<size_t, VertexAttributes>> m_v2;
		Remesher::EdgeMap<EdgeAttributes> m_edges;
		Remesher::FaceMap<FaceAttributes> m_faces;
		Remesher::TetMap<TetAttributes> m_tets;
	};
} // namespace polyfem::mesh