#pragma once

#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>

namespace polyfem::mesh
{
	template <typename M>
	class LocalMesh
	{
	protected:
		using Tuple = typename M::Tuple;

	public:
		LocalMesh(
			const M &m,
			const std::vector<Tuple> &element_tuples,
			const bool include_global_boundary);

		/// @brief Construct a local mesh as an n-ring around a vertex.
		static std::vector<Tuple> n_ring(
			const M &m,
			const Tuple &center,
			const int n);

		/// @brief Construct a local mesh as an n-ring around a vertex.
		static std::vector<Tuple> flood_fill_n_ring(
			const M &m,
			const Tuple &center,
			const double area);

		static std::vector<Tuple> ball_selection(
			const M &m,
			const VectorNd &center,
			const double rel_radius);

		int num_elements() const { return m_elements.rows(); }
		int num_local_vertices() const { return m_num_local_vertices; }
		int num_vertices() const { return m_rest_positions.rows(); }

		const Eigen::MatrixXd &rest_positions() const { return m_rest_positions; }
		const Eigen::MatrixXd &positions() const { return m_positions; }
		Eigen::MatrixXd displacements() const { return m_positions - m_rest_positions; }

		const Eigen::MatrixXd &projection_quantities() const { return m_projection_quantities; }

		const Eigen::MatrixXi &elements() const { return m_elements; }
		const Eigen::MatrixXi &boundary_edges() const { return m_boundary_edges; }
		const Eigen::MatrixXi &boundary_faces() const { return m_boundary_faces; }

		const std::unordered_map<int, int> &global_to_local() const { return m_global_to_local; }
		const std::vector<int> &local_to_global() const { return m_local_to_global; }

		const std::vector<int> &fixed_vertices() const { return m_fixed_vertices; }
		const std::vector<int> &boundary_ids() const { return m_boundary_ids; }
		const std::vector<int> &body_ids() const { return m_body_ids; }

		void reorder_vertices(const Eigen::VectorXi &permutation);

		/// @brief Get a reference to the boundary facets (edges in 2D or faces in 3D).
		/// @todo Make this const.
		Eigen::MatrixXi &boundary_facets();
		const Eigen::MatrixXi &boundary_facets() const;

		void write_mesh(const std::string &path, const Eigen::MatrixXd &sol) const;

	protected:
		void remove_duplicate_fixed_vertices();
		void init_local_to_global();
		void init_vertex_attributes(const M &m);

		Eigen::MatrixXd m_rest_positions;
		Eigen::MatrixXd m_positions;

		Eigen::MatrixXd m_projection_quantities;

		Eigen::MatrixXi m_elements;
		Eigen::MatrixXi m_boundary_edges;
		Eigen::MatrixXi m_boundary_faces;

		int m_num_local_vertices;
		std::unordered_map<int, int> m_global_to_local;
		std::vector<int> m_local_to_global;

		std::vector<int> m_fixed_vertices;
		std::vector<int> m_boundary_ids; // only for boundary facets
		std::vector<int> m_body_ids;
	};
} // namespace polyfem::mesh