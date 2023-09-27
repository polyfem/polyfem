#pragma once

#include <polyfem/mesh/remesh/WildRemesher.hpp>

#include <wmtk/utils/TupleUtils.hpp>

namespace polyfem::mesh
{
	class TriOperationCache
	{
	public:
		using VertexAttributes = WildTriRemesher::VertexAttributes;
		using EdgeAttributes = WildTriRemesher::BoundaryAttributes;
		using FaceAttributes = WildTriRemesher::ElementAttributes;
		using Tuple = wmtk::TriMesh::Tuple;

		/// @brief Construct a local mesh as an n-ring around a vertex.
		static std::shared_ptr<TriOperationCache> split_edge(WildTriRemesher &m, const Tuple &t);
		static std::shared_ptr<TriOperationCache> swap_edge(WildTriRemesher &m, const Tuple &t);
		static std::shared_ptr<TriOperationCache> collapse_edge(WildTriRemesher &m, const Tuple &t);

		const std::pair<size_t, VertexAttributes> &v0() const { return m_v0; }
		const std::pair<size_t, VertexAttributes> &v1() const { return m_v1; }
		const Remesher::EdgeMap<EdgeAttributes> &edges() const { return m_edges; }
		const std::vector<FaceAttributes> &faces() const { return m_faces; }
		bool is_boundary_op() const { return m_is_boundary_op; }

		double local_energy = std::numeric_limits<double>::quiet_NaN();
		CollapseEdgeTo collapse_to = CollapseEdgeTo::ILLEGAL;

	protected:
		std::pair<size_t, VertexAttributes> m_v0;
		std::pair<size_t, VertexAttributes> m_v1;
		Remesher::EdgeMap<EdgeAttributes> m_edges;
		std::vector<FaceAttributes> m_faces;
		bool m_is_boundary_op = false;
	};

	class TetOperationCache
	{
	public:
		using VertexAttributes = WildTetRemesher::VertexAttributes;
		using EdgeAttributes = WildTetRemesher::EdgeAttributes;
		using FaceAttributes = WildTetRemesher::BoundaryAttributes;
		using TetAttributes = WildTetRemesher::ElementAttributes;
		using Tuple = wmtk::TetMesh::Tuple;

		static std::shared_ptr<TetOperationCache> split_edge(WildTetRemesher &m, const Tuple &t);
		static std::shared_ptr<TetOperationCache> swap_32(WildTetRemesher &m, const Tuple &t) { log_and_throw_error("TetOperationCache::swap_32 not implemented!"); }
		// static TetOperationCache swap_23(WildTetRemesher &m, const Tuple &t);
		// static TetOperationCache swap_44(WildTetRemesher &m, const Tuple &t);
		static std::shared_ptr<TetOperationCache> collapse_edge(WildTetRemesher &m, const Tuple &t);

		const std::pair<size_t, VertexAttributes> &v0() const { return m_v0; }
		const std::pair<size_t, VertexAttributes> &v1() const { return m_v1; }
		const std::optional<std::pair<size_t, VertexAttributes>> &v2() const { return m_v2; }
		const Remesher::EdgeMap<EdgeAttributes> &edges() const { return m_edges; }
		const Remesher::FaceMap<FaceAttributes> &faces() const { return m_faces; }
		const Remesher::TetMap<TetAttributes> &tets() const { return m_tets; }
		bool is_boundary_op() const { return m_is_boundary_op; }

		double local_energy = std::numeric_limits<double>::quiet_NaN();
		CollapseEdgeTo collapse_to = CollapseEdgeTo::ILLEGAL;

	protected:
		std::pair<size_t, VertexAttributes> m_v0;
		std::pair<size_t, VertexAttributes> m_v1;
		std::optional<std::pair<size_t, VertexAttributes>> m_v2;
		Remesher::EdgeMap<EdgeAttributes> m_edges;
		Remesher::FaceMap<FaceAttributes> m_faces;
		Remesher::TetMap<TetAttributes> m_tets;
		bool m_is_boundary_op = false;
	};
} // namespace polyfem::mesh