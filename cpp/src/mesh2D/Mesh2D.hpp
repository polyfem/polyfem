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
		void scale(const double scaling) override { assert(false); }

		void refine(const int n_refiniment, const double t) override;

		inline bool is_volume() const override { return false; }

		inline int n_edges() const { return mesh_.edges.nb(); }
		inline int n_elements() const override { return mesh_.facets.nb(); }
		inline int n_pts() const { return mesh_.vertices.nb(); }

		inline int n_element_vertices(const int element_index) const override { return mesh_.facets.nb_vertices(element_index);}
		inline int vertex_global_index(const int element_index, const int local_index) const { return mesh_.facets.vertex(element_index, local_index); }

		void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;
		// void element_bounday_polygon(const int index, Eigen::MatrixXd &poly) const;

		void set_boundary_tags(std::vector<int> &tags) const override;

		void point(const int global_index, Eigen::MatrixXd &pt) const override;

		inline bool is_vertex_boundary(const int v_id) const
		{
			GEO::Attribute<bool> vertices_real_boundary(mesh_.vertices.attributes(), "vertices_real_boundary");
			return vertices_real_boundary[v_id];
		}

		bool load(const std::string &path) override;
		bool save(const std::string &path) const override;

		void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const override;

		Eigen::MatrixXd edge_mid_point(const int edge_id) const;

		//get nodes ids
		int edge_node_id(const int edge_id) const;
		int vertex_node_id(const int vertex_id) const;
		bool node_id_from_edge_index(const Navigation::Index &index, int &id) const;


		//get nodes positions
		Eigen::MatrixXd node_from_edge_index(const Navigation::Index &index) const;
		Eigen::MatrixXd node_from_face(const int face_id) const;
		Eigen::MatrixXd node_from_vertex(const int vertex_id) const;

		//navigation wrapper
		Navigation::Index get_index_from_face(int f, int lv = 0) const;

		// Navigation in a surface mesh
		Navigation::Index switch_vertex(Navigation::Index idx) const;
		Navigation::Index switch_edge(Navigation::Index idx) const;
		Navigation::Index switch_face(Navigation::Index idx) const;

		// Iterate in a mesh
		inline Navigation::Index next_around_face(Navigation::Index idx) const { return switch_edge(switch_vertex(idx)); }
		// inline Navigation::Index next_around_edge(Navigation::Index idx) const { return switch_vertex(switch_face(idx)); }
		inline Navigation::Index next_around_vertex(Navigation::Index idx) const { return switch_face(switch_edge(idx)); }

		void create_boundary_nodes();

		void compute_element_tag(std::vector<ElementType> &ele_tag) const override;
		void compute_barycenter(Eigen::MatrixXd &barycenters) const override;

		const GEO::Mesh &geo_mesh() const { std::cerr<<"never user this function"<<std::endl; return mesh_; }
	private:
		GEO::Mesh mesh_;
	};
}

#endif //MESH_2D_HPP
