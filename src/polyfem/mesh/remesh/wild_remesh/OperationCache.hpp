#pragma once

#include <polyfem/mesh/remesh/WildRemeshing.hpp>

#include <wmtk/utils/TupleUtils.hpp>

namespace polyfem::mesh
{
	class WildRemeshing2D;
	class WildRemeshing3D;

	class OperationCache2D
	{
	public:
		static constexpr int DIM = 2;
		using VertexAttributes = WildRemeshing::VertexAttributes<DIM>;
		using EdgeAttributes = WildRemeshing::BoundaryAttributes;
		using FaceAttributes = WildRemeshing::ElementAttributes;
		using Tuple = wmtk::TriMesh::Tuple;

		/// @brief Construct a local mesh as an n-ring around a vertex.
		static OperationCache2D split(WildRemeshing2D &m, const Tuple &t);
		static OperationCache2D swap(WildRemeshing2D &m, const Tuple &t);
		static OperationCache2D collapse(WildRemeshing2D &m, const Tuple &t);

		const std::pair<size_t, VertexAttributes> &v0() const { return m_v0; }
		const std::pair<size_t, VertexAttributes> &v1() const { return m_v1; }
		const WildRemeshing::EdgeMap<EdgeAttributes> &edges() const { return m_edges; }
		const std::vector<FaceAttributes> &faces() const { return m_faces; }

	protected:
		std::pair<size_t, VertexAttributes> m_v0;
		std::pair<size_t, VertexAttributes> m_v1;
		WildRemeshing::EdgeMap<EdgeAttributes> m_edges;
		std::vector<FaceAttributes> m_faces;
	};

	class OperationCache3D
	{
	public:
		static constexpr int DIM = 3;
		using VertexAttributes = WildRemeshing::VertexAttributes<DIM>;
		using FaceAttributes = WildRemeshing::BoundaryAttributes;
		using TetAttributes = WildRemeshing::ElementAttributes;
		using Tuple = wmtk::TetMesh::Tuple;

		static OperationCache3D split(WildRemeshing3D &m, const Tuple &t);
		static OperationCache3D swap_32(WildRemeshing3D &m, const Tuple &t);
		static OperationCache3D swap_23(WildRemeshing3D &m, const Tuple &t);
		static OperationCache3D swap_44(WildRemeshing3D &m, const Tuple &t);
		static OperationCache3D collapse(WildRemeshing3D &m, const Tuple &t);

		const std::pair<size_t, VertexAttributes> &v0() const { return m_v0; }
		const std::pair<size_t, VertexAttributes> &v1() const { return m_v1; }
		const std::optional<std::pair<size_t, VertexAttributes>> &v2() const { return m_v2; }
		const WildRemeshing::FaceMap<FaceAttributes> &faces() const { return m_faces; }
		const std::vector<TetAttributes> &tets() const { return m_tets; }

	protected:
		std::pair<size_t, VertexAttributes> m_v0;
		std::pair<size_t, VertexAttributes> m_v1;
		std::optional<std::pair<size_t, VertexAttributes>> m_v2;
		WildRemeshing::FaceMap<FaceAttributes> m_faces;
		std::vector<TetAttributes> m_tets;
	};
} // namespace polyfem::mesh