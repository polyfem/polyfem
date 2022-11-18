#pragma once

#include <polyfem/mesh/remesh/WildRemesh2D.hpp>

namespace polyfem::mesh
{
	class LocalMesh
	{
		using Tuple = WildRemeshing2D::Tuple;

	public:
		LocalMesh(const WildRemeshing2D &m, const std::vector<Tuple> &triangle_tuples);

		/// @brief Construct a local mesh as an n-ring around a vertex.
		static LocalMesh n_ring(const WildRemeshing2D &m, const Tuple &center, int n);

		int num_triangles() const { return m_triangles.rows(); }
		int num_vertices() const { return m_global_to_local.size(); }

		const Eigen::MatrixXi &triangles() const { return m_triangles; }
		const Eigen::MatrixXd &rest_positions() const { return m_rest_positions; }
		const Eigen::MatrixXd &positions() const { return m_positions; }
		Eigen::MatrixXd displacements() const { return m_positions - m_rest_positions; }
		const Eigen::MatrixXd &prev_positions() const { return m_prev_positions; }
		const Eigen::MatrixXd &prev_velocities() const { return m_prev_velocities; }
		const Eigen::MatrixXd &prev_accelerations() const { return m_prev_accelerations; }
		const std::unordered_map<int, int> &global_to_local() const { return m_global_to_local; }
		const std::vector<int> &local_to_global() const { return m_local_to_global; }
		const std::vector<int> &body_ids() const { return m_body_ids; }
		const std::vector<std::unordered_set<int>> &vertex_boundary_ids() const { return m_vertex_boundary_ids; }
		const WildRemeshing2D::EdgeMap<bool> &is_edge_on_global_boundary() const { return m_is_edge_on_global_boundary; }
		bool is_edge_on_global_boundary(size_t v0, size_t v1) const;
		bool is_edge_on_global_boundary(const Eigen::Vector2i &e) const { return is_edge_on_global_boundary(e[0], e[1]); }

	protected:
		Eigen::MatrixXi m_triangles;
		Eigen::MatrixXd m_rest_positions;
		Eigen::MatrixXd m_positions;
		// TODO: replace this with a time integrator object
		Eigen::MatrixXd m_prev_positions;
		Eigen::MatrixXd m_prev_velocities;
		Eigen::MatrixXd m_prev_accelerations;
		std::unordered_map<int, int> m_global_to_local;
		std::vector<int> m_local_to_global;
		std::vector<int> m_body_ids;
		std::vector<std::unordered_set<int>> m_vertex_boundary_ids;
		WildRemeshing2D::EdgeMap<bool> m_is_edge_on_global_boundary;

		static constexpr int DIM = 2;
	};
} // namespace polyfem::mesh