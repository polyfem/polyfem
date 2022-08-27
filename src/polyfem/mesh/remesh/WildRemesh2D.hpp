#pragma once

#include <wmtk/ConcurrentTriMesh.h>

namespace polyfem::mesh
{
	class WildRemeshing2D : public wmtk::ConcurrentTriMesh
	{
	public:
		typedef wmtk::ConcurrentTriMesh super;

		WildRemeshing2D(
			Eigen::MatrixXd rest_positions,
			Eigen::MatrixXd displacements,
			Eigen::MatrixXd velocities,
			Eigen::MatrixXd accelerations,
			int num_threads = 1);

		virtual ~WildRemeshing2D(){};

		void create_mesh(
			size_t n_vertices,
			const std::vector<std::array<size_t, 3>> &tris);

		void freeze_vertex(TriMesh::Tuple &v);

		void partition_mesh();

		bool collapse_edge_before(const Tuple &t) override;
		bool collapse_edge_after(const Tuple &t) override;
		bool collapse_shortest(int target_vertex_count);
		bool invariants(const std::vector<Tuple> &new_tris) override;

		void build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &F);

		bool is_vertex_frozen(const Tuple &v) const
		{
			return vertex_attrs[v.vid(*this)].frozen;
		}

		struct VertexAttributes
		{
			Eigen::Vector2d rest_position;
			Eigen::Vector2d displacement;
			Eigen::Vector2d velocity;
			Eigen::Vector2d acceleration;
			size_t partition_id = 0;
			bool frozen = false;
		};

		wmtk::AttributeCollection<VertexAttributes> vertex_attrs;
		int retry_limit = 10;

	protected:
		std::vector<TriMesh::Tuple> new_edges_after(const std::vector<TriMesh::Tuple> &t) const;

		/// Cached edge vertices before collapse.
		struct PositionInfoCache
		{
			VertexAttributes v1_attr;
			VertexAttributes v2_attr;
		};
		tbb::enumerable_thread_specific<PositionInfoCache> position_cache;
	};

} // namespace polyfem::mesh
