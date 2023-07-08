#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh2D/Navigation.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <geogram/mesh/mesh.h>

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>

namespace polyfem
{
	namespace mesh
	{
		class CMesh2D : public Mesh2D
		{
		public:
			CMesh2D() = default;
			virtual ~CMesh2D() = default;
			// We cannot move or copy CMesh2D because it has unique_ptrs which do
			// not support copy and GEO::Mesh which does not support move.
			CMesh2D(CMesh2D &&) = delete;
			CMesh2D &operator=(CMesh2D &&) = delete;
			CMesh2D(const CMesh2D &) = delete;
			CMesh2D &operator=(const CMesh2D &) = delete;

			void refine(const int n_refinement, const double t) override;

			bool is_conforming() const override { return true; }

			int n_faces() const override { return mesh_.facets.nb(); }
			int n_edges() const override { return mesh_.edges.nb(); }
			int n_vertices() const override { return mesh_.vertices.nb(); }

			inline int n_face_vertices(const int f_id) const override { return mesh_.facets.nb_vertices(f_id); }

			inline int face_vertex(const int f_id, const int lv_id) const override { return mesh_.facets.vertex(f_id, lv_id); }
			inline int edge_vertex(const int e_id, const int lv_id) const override { return mesh_.edges.vertex(e_id, lv_id); }
			inline int cell_vertex(const int f_id, const int lv_id) const override { return mesh_.facets.vertex(f_id, lv_id); }

			bool is_boundary_vertex(const int vertex_global_id) const override
			{
				// GEO::Attribute<bool> boundary_vertices(mesh_.vertices.attributes(), "boundary_vertex");
				return (*boundary_vertices_)[vertex_global_id];
			}
			bool is_boundary_edge(const int edge_global_id) const override
			{
				// GEO::Attribute<bool> boundary_edges(mesh_.edges.attributes(), "boundary_edge");
				return (*boundary_edges_)[edge_global_id];
			}

			bool is_boundary_element(const int element_global_id) const override;

			bool save(const std::string &path) const override;

			bool build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) override;

			void attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes) override;
			RowVectorNd edge_node(const Navigation::Index &index, const int n_new_nodes, const int i) const override;
			RowVectorNd face_node(const Navigation::Index &index, const int n_new_nodes, const int i, const int j) const override;

			void normalize() override;

			double edge_length(const int gid) const override;

			void compute_elements_tag() override;
			virtual void update_elements_tag() override;

			void set_point(const int global_index, const RowVectorNd &p) override;

			virtual RowVectorNd point(const int global_index) const override;
			virtual RowVectorNd edge_barycenter(const int index) const override;

			virtual void bounding_box(RowVectorNd &min, RowVectorNd &max) const override;

			void compute_boundary_ids(const double eps) override;
			void compute_boundary_ids(const std::function<int(const RowVectorNd &)> &marker) override;
			void compute_boundary_ids(const std::function<int(const RowVectorNd &, bool)> &marker) override;
			void compute_boundary_ids(const std::function<int(const size_t, const RowVectorNd &, bool)> &marker) override;
			void compute_boundary_ids(const std::function<int(const std::vector<int> &, bool)> &marker) override;
			void compute_boundary_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &marker) override;

			void compute_body_ids(const std::function<int(const size_t, const RowVectorNd &)> &marker) override;

			// Navigation wrapper
			inline Navigation::Index get_index_from_face(int f, int lv = 0) const override { return Navigation::get_index_from_face(mesh_, *c2e_, f, lv); }

			// Navigation in a surface mesh
			inline Navigation::Index switch_vertex(Navigation::Index idx) const override { return Navigation::switch_vertex(mesh_, idx); }
			inline Navigation::Index switch_edge(Navigation::Index idx) const override { return Navigation::switch_edge(mesh_, *c2e_, idx); }
			inline Navigation::Index switch_face(Navigation::Index idx) const override { return Navigation::switch_face(mesh_, *c2e_, idx); }

			void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;

			void append(const Mesh &mesh) override;

		protected:
			bool load(const std::string &path) override;
			bool load(const GEO::Mesh &mesh) override;

		private:
			GEO::Mesh mesh_;
			std::unique_ptr<GEO::Attribute<GEO::index_t>> c2e_;
			std::unique_ptr<GEO::Attribute<bool>> boundary_vertices_;
			std::unique_ptr<GEO::Attribute<bool>> boundary_edges_;
		};
	} // namespace mesh
} // namespace polyfem
