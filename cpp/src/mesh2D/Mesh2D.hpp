#ifndef MESH_2D_HPP
#define MESH_2D_HPP

#include "Mesh.hpp"
#include "Navigation.hpp"

#include <Eigen/Dense>
#include <geogram/mesh/mesh.h>

namespace poly_fem
{
	class Mesh2D : public Mesh
	{
	public:
		void refine(const int n_refiniment, const double t) override;

		bool is_volume() const override { return false; }

		int n_cells() const override { return 0; }
		int n_faces() const override { return mesh_.facets.nb(); }
		int n_edges() const override { return mesh_.edges.nb(); }
		int n_vertices() const override { return mesh_.vertices.nb(); }

		bool is_boundary_vertex(const int vertex_global_id) const override {
			GEO::Attribute<bool> boundary_vertices(mesh_.vertices.attributes(), "boundary_vertex");
			return boundary_vertices[vertex_global_id];
		}
		bool is_boundary_edge(const int edge_global_id) const override {
			GEO::Attribute<bool> boundary_edges(mesh_.edges.attributes(), "boundary_edge");
			return boundary_edges[edge_global_id];
		}
		bool is_boundary_face(const int face_global_id) const override {
			assert(false);
			return false;
		}


		bool load(const std::string &path) override;
		bool save(const std::string &path) const override;

		void compute_elements_tag() const override;


		void point(const int global_index, Eigen::MatrixXd &pt) const override;
		Eigen::RowVector2d point(const int global_index) const;
		void edge_barycenters(Eigen::MatrixXd &barycenters) const override;
		void face_barycenters(Eigen::MatrixXd &barycenters) const override;
		void cell_barycenters(Eigen::MatrixXd &barycenters) const override { }



		//navigation wrapper
		inline Navigation::Index get_index_from_face(int f, int lv = 0) const { return Navigation::get_index_from_face(mesh_, f, lv); }

		// Navigation in a surface mesh
		inline Navigation::Index switch_vertex(Navigation::Index idx) const { return Navigation::switch_vertex(mesh_, idx); }
		inline Navigation::Index switch_edge(Navigation::Index idx) const { return Navigation::switch_edge(mesh_, idx); }
		inline Navigation::Index switch_face(Navigation::Index idx) const { return Navigation::switch_face(mesh_, idx); }

		// Iterate in a mesh
		inline Navigation::Index next_around_face(Navigation::Index idx) const { return switch_edge(switch_vertex(idx)); }
		inline Navigation::Index next_around_vertex(Navigation::Index idx) const { return switch_face(switch_edge(idx)); }
		inline Navigation::Index next_around_vertex_inv(Navigation::Index idx) const {
			auto tmp = switch_face(idx);
			if(tmp.face < 0) return tmp;
			return switch_edge(tmp);
		}


		void fill_boundary_tags(std::vector<int> &tags) const override;

		void compute_element_barycenters(Eigen::MatrixXd &barycenters) const override;
		void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;
		void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const override;
	private:
		GEO::Mesh mesh_;
	};
}

#endif //MESH_2D_HPP
