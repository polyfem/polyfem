#ifndef MESH_2D_HPP
#define MESH_2D_HPP

#include <polyfem/Common.hpp>
#include <polyfem/Mesh.hpp>
#include <polyfem/Navigation.hpp>
#include <polyfem/Types.hpp>

#include <Eigen/Dense>
#include <geogram/mesh/mesh.h>

namespace polyfem
{
	class Mesh2D : public Mesh
	{
	public:
		Mesh2D() = default;
		virtual ~Mesh2D() = default;
		POLYFEM_DEFAULT_MOVE_COPY(Mesh2D)

		void refine(const int n_refiniment, const double t, std::vector<int> &parent_nodes) override;

		bool is_volume() const override { return false; }

		int n_cells() const override { return 0; }
		int n_faces() const override { return mesh_.facets.nb(); }
		int n_edges() const override { return mesh_.edges.nb(); }
		int n_vertices() const override { return mesh_.vertices.nb(); }

		inline int n_face_vertices(const int f_id) const {return mesh_.facets.nb_vertices(f_id); }

		inline int face_vertex(const int f_id, const int lv_id) const { return mesh_.facets.vertex(f_id, lv_id); }

		bool is_boundary_vertex(const int vertex_global_id) const override {
			// GEO::Attribute<bool> boundary_vertices(mesh_.vertices.attributes(), "boundary_vertex");
			return (*boundary_vertices_)[vertex_global_id];
		}
		bool is_boundary_edge(const int edge_global_id) const override {
			// GEO::Attribute<bool> boundary_edges(mesh_.edges.attributes(), "boundary_edge");
			return (*boundary_edges_)[edge_global_id];
		}
		bool is_boundary_face(const int face_global_id) const override {
			assert(false);
			return false;
		}

		bool is_boundary_element(const int element_global_id) const override;


		bool load(const std::string &path) override;
		bool load(const GEO::Mesh &mesh) override;
		bool save(const std::string &path) const override;
		bool build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) override;

		void attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes) override;
		RowVectorNd edge_node(const Navigation::Index &index, const int n_new_nodes, const int i) const;
		RowVectorNd face_node(const Navigation::Index &index, const int n_new_nodes, const int i, const int j) const;

		void normalize() override;

		double edge_length(const int gid) const override;

		void compute_elements_tag() override;
		virtual void update_elements_tag() override;

		void set_point(const int global_index, const RowVectorNd &p);

		virtual RowVectorNd point(const int global_index) const override;
		virtual RowVectorNd edge_barycenter(const int index) const override;
		virtual RowVectorNd face_barycenter(const int index) const override;
		virtual RowVectorNd cell_barycenter(const int index) const override { assert(false); return RowVectorNd(2); }


		virtual void bounding_box(RowVectorNd &min, RowVectorNd &max) const override;


		// Navigation wrapper
		inline Navigation::Index get_index_from_face(int f, int lv = 0) const { return Navigation::get_index_from_face(mesh_, *c2e_, f, lv); }

		// Navigation in a surface mesh
		inline Navigation::Index switch_vertex(Navigation::Index idx) const { return Navigation::switch_vertex(mesh_, idx); }
		inline Navigation::Index switch_edge(Navigation::Index idx) const { return Navigation::switch_edge(mesh_, *c2e_, idx); }
		inline Navigation::Index switch_face(Navigation::Index idx) const { return Navigation::switch_face(mesh_, *c2e_, idx); }

		// Iterate in a mesh
		inline Navigation::Index next_around_face(Navigation::Index idx) const { return switch_edge(switch_vertex(idx)); }
		inline Navigation::Index next_around_vertex(Navigation::Index idx) const { return switch_face(switch_edge(idx)); }


		void compute_boundary_ids() override;
		void compute_boundary_ids(const std::function<int(const RowVectorNd&)> &marker) override;

		void compute_element_barycenters(Eigen::MatrixXd &barycenters) const override { face_barycenters(barycenters); }
		void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;
		void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const override;
		void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1, const std::vector<bool> &valid_elements) const override;
	private:
		GEO::Mesh mesh_;
		std::unique_ptr<GEO::Attribute<GEO::index_t>> c2e_;
		std::unique_ptr<GEO::Attribute<bool>> boundary_vertices_;
		std::unique_ptr<GEO::Attribute<bool>> boundary_edges_;
	};
}

#endif //MESH_2D_HPP
