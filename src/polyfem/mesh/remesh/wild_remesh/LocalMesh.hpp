#pragma once

#include <polyfem/mesh/remesh/Remesher.hpp>
#include <polyfem/basis/ElementBases.hpp>
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
		static std::vector<Tuple> n_ring(
			const M &m,
			const std::vector<Tuple> &one_ring,
			const int n);

		/// @brief Construct a local mesh as an n-ring around a vertex.
		static std::vector<Tuple> flood_fill_n_ring(
			const M &m,
			const Tuple &center,
			const double area);

		static std::vector<Tuple> ball_selection(
			const M &m,
			const VectorNd &center,
			const double rel_radius,
			const int n_ring_size);

		/// Number of vertices in the local mesh (not including extra global boundary vertices).
		int num_local_vertices() const { return m_num_local_vertices; }
		/// Number of vertices in the local mesh.
		int num_vertices() const { return m_rest_positions.rows(); }
		/// Number of elements
		int num_elements() const { return m_elements.rows(); }

		/// Rest positions of the vertices.
		const Eigen::MatrixXd &rest_positions() const { return m_rest_positions; }
		/// Deformed positions of the vertices.
		const Eigen::MatrixXd &positions() const { return m_positions; }

		/// Displacements of the vertices.
		Eigen::MatrixXd displacements() const { return m_positions - m_rest_positions; }

		/// Projection quantaties for each vertex.
		const Eigen::MatrixXd &projection_quantities() const { return m_projection_quantities; }

		/// Elements in the local mesh.
		const Eigen::MatrixXi &elements() const { return m_elements; }
		/// Edged on the boundary.
		const Eigen::MatrixXi &boundary_edges() const { return m_boundary_edges; }
		/// Faces on the boundary (not set for TriMesh).
		const Eigen::MatrixXi &boundary_faces() const { return m_boundary_faces; }

		/// Map from global vertex index to local vertex index.
		const std::unordered_map<int, int> &global_to_local() const { return m_global_to_local; }

		/// Map from local vertex index to global vertex index.
		const std::vector<int> &local_to_global() const { return m_local_to_global; }

		/// Fixed vertices.
		const std::vector<int> &fixed_vertices() const { return m_fixed_vertices; }

		/// Map from boundary elements (all edges for TriMesh or faces in TetMesh) to their boundary id.
		const Remesher::BoundaryMap<int> &boundary_ids() const { return m_boundary_ids; }

		/// One body id per element.
		const std::vector<int> &body_ids() const { return m_body_ids; }

		/// @brief Get a reference to the boundary facets (edges in 2D or faces in 3D).
		/// @todo Make this const.
		Eigen::MatrixXi &boundary_facets();
		const Eigen::MatrixXi &boundary_facets() const;

		/// @brief Build the ElementBases for the local mesh.
		/// @note Reorders the vertices to match the order of the ElementBases.
		/// @param formulation Energy formulation.
		/// @return ElementBases for the local mesh.
		std::vector<polyfem::basis::ElementBases> build_bases(const std::string &formulation);

		/// @brief Reorder the vertices of the local mesh.
		/// @param permutation Map from old vertex index to new vertex index.
		void reorder_vertices(const Eigen::VectorXi &permutation);

		/// @brief Write the local mesh to a VTU file.
		/// @param path Filename to write to.
		/// @param sol Solution to write to the file.
		void write_mesh(const std::string &path, const Eigen::MatrixXd &sol) const;

	protected:
		void remove_duplicate_fixed_vertices();
		void init_local_to_global();
		void init_vertex_attributes(const M &m);

		/// Rest positions of the vertices.
		Eigen::MatrixXd m_rest_positions;

		/// Deformed positions of the vertices.
		Eigen::MatrixXd m_positions;

		/// Projection quantaties for each vertex.
		Eigen::MatrixXd m_projection_quantities;

		/// Elements in the local mesh.
		Eigen::MatrixXi m_elements;

		/// Edged on the boundary.
		Eigen::MatrixXi m_boundary_edges;

		/// Faces on the boundary (not set for TriMesh).
		Eigen::MatrixXi m_boundary_faces;

		/// Number of vertices in the local mesh (not including extra global boundary vertices).
		int m_num_local_vertices;

		/// Map from global vertex index to local vertex index.
		std::unordered_map<int, int> m_global_to_local;

		/// Map from local vertex index to global vertex index.
		std::vector<int> m_local_to_global;

		/// Fixed vertices.
		std::vector<int> m_fixed_vertices;

		/// Map from boundary elements (all edges for TriMesh or faces in TetMesh) to their boundary id.
		Remesher::BoundaryMap<int> m_boundary_ids;

		/// One body id per element.
		std::vector<int> m_body_ids;
	};
} // namespace polyfem::mesh