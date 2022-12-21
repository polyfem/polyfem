#pragma once

#include <polyfem/mesh/remesh/WildRemeshing.hpp>

namespace polyfem::mesh
{
	template <class WMTKMesh, int DIM>
	class WildRemeshingND : public WildRemeshing, public WMTKMesh
	{
	public:
		using Tuple = typename WMTKMesh::Tuple;
		using VertexAttributes = WildRemeshing::VertexAttributes<DIM>;
		using BoundaryAttributes = WildRemeshing::BoundaryAttributes;
		using ElementAttributes = WildRemeshing::ElementAttributes;

		/// @brief Construct a new WildRemeshingND object
		/// @param state Simulation current state
		WildRemeshingND(
			const State &state,
			const Eigen::MatrixXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy)
			: WildRemeshing(state, obstacle_displacements, obstacle_vals, current_time, starting_energy),
			  WMTKMesh()
		{
		}

		virtual ~WildRemeshingND(){};

		/// @brief Initialize the mesh
		/// @param rest_positions Rest positions of the mesh (|V| × 2)
		/// @param positions Current positions of the mesh (|V| × 2)
		/// @param triangles Triangles of the mesh (|T| × 3)
		/// @param projection_quantities Quantities to be projected to the new mesh (2 rows per vertex and 1 column per quantity)
		/// @param edge_to_boundary_id Map from edge to boundary id (of size |E|)
		/// @param body_ids Body ids of the mesh (of size |T|)
		virtual void init(
			const Eigen::MatrixXd &rest_positions,
			const Eigen::MatrixXd &positions,
			const Eigen::MatrixXi &triangles,
			const Eigen::MatrixXd &projection_quantities,
			const BoundaryMap<int> &boundary_to_id,
			const std::vector<int> &body_ids) override;

		// ---------------------------------------------------------------------
		// Getters

		std::vector<Tuple> get_elements() const;

		/// @brief Dimension of the mesh
		int dim() const override { return DIM; }
		/// @brief Exports rest positions of the stored mesh
		Eigen::MatrixXd rest_positions() const override;
		/// @brief Exports positions of the stored mesh
		Eigen::MatrixXd displacements() const override;
		/// @brief Exports displacements of the stored mesh
		Eigen::MatrixXd positions() const override;
		/// @brief Exports edges of the stored mesh
		Eigen::MatrixXi edges() const override;
		/// @brief Exports triangles of the stored mesh
		Eigen::MatrixXi faces() const;
		/// @brief Exports triangles of the stored mesh
		Eigen::MatrixXi elements() const override;
		/// @brief Exports projected quantities of the stored mesh
		Eigen::MatrixXd projected_quantities() const override;
		/// @brief Exports boundary ids of the stored mesh
		BoundaryMap<int> boundary_ids() const override;
		/// @brief Exports body ids of the stored mesh
		std::vector<int> body_ids() const override;
		/// @brief Get the boundary nodes of the stored mesh
		std::vector<int> boundary_nodes() const override;

		/// @brief Number of projection quantities (not including the position)
		int n_quantities() const override { return m_n_quantities; };

		// ---------------------------------------------------------------------
		// Setters

		/// @brief Set rest positions of the stored mesh
		void set_rest_positions(const Eigen::MatrixXd &positions) override;
		/// @brief Set deformed positions of the stored mesh
		void set_positions(const Eigen::MatrixXd &positions) override;
		/// @brief Set projected quantities of the stored mesh
		void set_projected_quantities(const Eigen::MatrixXd &projected_quantities) override;
		/// @brief Set if a vertex is fixed
		void set_fixed(const std::vector<bool> &fixed) override;
		/// @brief Set the boundary IDs of all edges
		void set_boundary_ids(const BoundaryMap<int> &boundary_to_id) override;
		/// @brief Set the body IDs of all triangles
		void set_body_ids(const std::vector<int> &body_ids) override;

		// ---------------------------------------------------------------------

		/// @brief Check if invariants are satisfied
		bool invariants(const std::vector<Tuple> &new_tris) override;

		/// @brief Check if a triangle is inverted
		virtual bool is_inverted(const Tuple &loc) const = 0;

		// ---------------------------------------------------------------------
		// Utils

		/// @brief Compute the length of an edge.
		double edge_length(const Tuple &e) const;

		/// @brief Compute the volume (area) of an tetrahedron (triangle) element.
		double element_volume(const Tuple &e) const;

		// ---------------------------------------------------------------------
		// Attributes

		wmtk::AttributeCollection<VertexAttributes> vertex_attrs;
		wmtk::AttributeCollection<BoundaryAttributes> boundary_attrs;
		wmtk::AttributeCollection<ElementAttributes> element_attrs;

	protected:
		int m_n_quantities;
		double total_volume;
	};

} // namespace polyfem::mesh
