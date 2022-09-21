#pragma once

#include <wmtk/TriMesh.h>

namespace polyfem::mesh
{
	class WildRemeshing2D : public wmtk::TriMesh
	{
	public:
		typedef wmtk::TriMesh super;

		WildRemeshing2D() {}

		virtual ~WildRemeshing2D(){};

		// Initializes the mesh
		void create_mesh(
			const Eigen::MatrixXd &rest_positions,
			const Eigen::MatrixXd &positions,
			const Eigen::MatrixXi &triangles);

		// Exports positions and triangles of the stored mesh
		void export_mesh(
			Eigen::MatrixXd &rest_positions,
			Eigen::MatrixXd &positions,
			Eigen::MatrixXi &triangles);

		// Writes a triangle mesh in OBJ format
		void write_rest_obj(const std::string &path);
		void write_deformed_obj(const std::string &path);

		// Computes the quality of a triangle
		double get_quality(const Tuple &loc) const;

		// Computes the average quality of a mesh
		Eigen::VectorXd get_quality_all_triangles();

		// Check if a triangle is inverted
		bool is_inverted(const Tuple &loc) const;

		bool invariants(const std::vector<Tuple> &new_tris) override;

		// Smoothing
		void smooth_all_vertices();
		bool smooth_before(const Tuple &t) override;
		bool smooth_after(const Tuple &t) override;

		// void freeze_vertex(TriMesh::Tuple &v);

		// void partition_mesh();

		// bool collapse_edge_before(const Tuple &t) override;
		// bool collapse_edge_after(const Tuple &t) override;
		// bool collapse_shortest(int target_vertex_count);
		// bool invariants(const std::vector<Tuple> &new_tris) override;

		// void build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &triangles);

		// bool is_vertex_frozen(const Tuple &v) const
		// {
		// 	return vertex_attrs[v.vid(*this)].frozen;
		// }

		struct VertexAttributes
		{

			Eigen::Vector2d rest_position;
			Eigen::Vector2d position;
			size_t partition_id = 0; // Vertices marked as fixed cannot be modified by any local operation
			bool frozen = false;
		};

		wmtk::AttributeCollection<VertexAttributes> vertex_attrs;

		// Energy Assigned to undefined energy
		// TODO: why not the max double?
		static constexpr double MAX_ENERGY = 1e50;

	protected:
		// std::vector<TriMesh::Tuple> new_edges_after(const std::vector<TriMesh::Tuple> &t) const;

		/// Cached edge vertices before collapse.
		// struct PositionInfoCache
		// {
		// 	VertexAttributes v1_attr;
		// 	VertexAttributes v2_attr;
		// };
		// tbb::enumerable_thread_specific<PositionInfoCache> position_cache;
	};

} // namespace polyfem::mesh
