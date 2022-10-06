#pragma once

#include <polyfem/State.hpp>

#include <wmtk/TriMesh.h>
#include <wmtk/ExecutionScheduler.hpp>

namespace polyfem::mesh
{
	class WildRemeshing2D : public wmtk::TriMesh
	{
	public:
		typedef wmtk::TriMesh super;

		WildRemeshing2D(const Obstacle &obstacle)
			: wmtk::TriMesh(), obstacle(obstacle) {}

		virtual ~WildRemeshing2D(){};

		static constexpr int DIM = 2;
		static constexpr wmtk::ExecutionPolicy EXECUTION_POLICY = wmtk::ExecutionPolicy::kSeq;

		// Initializes the mesh
		void create_mesh(
			const Eigen::MatrixXd &rest_positions,
			const Eigen::MatrixXd &positions,
			const Eigen::MatrixXd &velocities,
			const Eigen::MatrixXd &accelerations,
			const Eigen::MatrixXi &triangles);

		/// Exports rest positions of the stored mesh
		Eigen::MatrixXd rest_positions() const;
		/// Exports positions of the stored mesh
		Eigen::MatrixXd displacements() const;
		/// Exports displacements of the stored mesh
		Eigen::MatrixXd positions() const;
		/// Exports velocities of the stored mesh
		Eigen::MatrixXd velocities() const;
		/// Exports accelerations of the stored mesh
		Eigen::MatrixXd accelerations() const;
		/// Exports triangles of the stored mesh
		Eigen::MatrixXi triangles() const;

		/// Set positions of the stored mesh
		void set_positions(const Eigen::MatrixXd &positions);
		/// Set velocities of the stored mesh
		void set_velocities(const Eigen::MatrixXd &velocities);
		/// Set accelerations of the stored mesh
		void set_accelerations(const Eigen::MatrixXd &accelerations);

		/// Writes a triangle mesh in OBJ format
		void write_obj(const std::string &path, bool deformed) const;
		void write_rest_obj(const std::string &path) const { write_obj(path, false); }
		void write_deformed_obj(const std::string &path) const { write_obj(path, true); }

		/// Compute the global energy of the mesh
		double compute_global_energy() const;
		double compute_global_wicke_measure() const;

		// Check if a triangle is inverted
		bool is_inverted(const Tuple &loc) const;

		// Check if invariants
		bool invariants(const std::vector<Tuple> &new_tris) override;

		// Update the mesh position
		void update_positions();

		// Smoothing
		void smooth_all_vertices();
		bool smooth_before(const Tuple &t) override;
		bool smooth_after(const Tuple &t) override;

		// Edge splitting
		void split_all_edges();
		bool split_edge_before(const Tuple &t) override;
		bool split_edge_after(const Tuple &t) override;

		// Edge collapse
		void collapse_all_edges();
		bool collapse_edge_before(const Tuple &t) override;
		bool collapse_edge_after(const Tuple &t) override;

		std::vector<Tuple> new_edges_after(const std::vector<Tuple> &tris) const;

		struct VertexAttributes
		{
			Eigen::Vector2d rest_position;
			Eigen::Vector2d position;
			Eigen::Vector2d velocity;
			Eigen::Vector2d acceleration;
			size_t partition_id = 0; // Vertices marked as fixed cannot be modified by any local operation
			bool frozen = false;

			Eigen::Vector2d displacement() const { return position - rest_position; }
		};
		wmtk::AttributeCollection<VertexAttributes> vertex_attrs;

		struct FaceAttributes
		{
			// polyfem::basis::ElementBases bases;
			// polyfem::basis::ElementBases geom_bases;
			// int body_id = 0;
		};

		struct EdgeAttributes
		{
			int boundary_id = -1;
		};

	protected:
		/// Get the boundary nodes of the stored mesh
		std::vector<int> boundary_nodes() const;

		void cache_before();

		const Obstacle &obstacle;

		// int old_n_bases;
		// std::vector<polyfem::basis::ElementBases> old_bases;
		// const std::vector<polyfem::basis::ElementBases> old_geom_bases;
		Eigen::MatrixXd rest_positions_before;
		Eigen::MatrixXd positions_before;
		Eigen::MatrixXd velocities_before;
		Eigen::MatrixXd accelerations_before;
		Eigen::MatrixXi triangles_before;
		double energy_before;

		std::array<VertexAttributes, 2> cache;
	};

} // namespace polyfem::mesh
