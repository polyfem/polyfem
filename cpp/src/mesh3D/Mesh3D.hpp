#pragma once

#include "Mesh.hpp"

#include "Navigation3D.hpp"
#include "Mesh3DStorage.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <array>

namespace poly_fem
{
	class Mesh3D : public Mesh
	{
	public:
		void scale(const double scaling) override { mesh_.points *= scaling; }

		void refine(const int n_refiniment, const double t) override;

		inline bool is_volume() const override { return true; }

		inline bool is_boundary_vertex(const int vertex_global_id) const { return mesh_.vertices[vertex_global_id].boundary; }
		inline bool is_boundary_edge(const int edge_global_id) const { return mesh_.edges[edge_global_id].boundary; }
		inline bool is_boundary_face(const int face_global_id) const { return mesh_.faces[face_global_id].boundary; }

		inline int n_elements() const override { return int(mesh_.elements.size()); }
		inline int n_faces() const { return int(mesh_.faces.size()); }
		inline int n_edges() const { return int(mesh_.edges.size()); }
		inline int n_pts() const { return int(mesh_.points.cols()); }

		inline int n_face_vertices(const int face_index) const { return int(mesh_.faces[face_index].vs.size()); }

		inline int n_element_vertices(const int element_index) const override { return int(mesh_.elements[element_index].vs.size());}
		inline int n_element_faces(const int element_index) const { return int(mesh_.elements[element_index].fs.size());}

		inline int vertex_global_index(const int element_index, const int local_index) const { return mesh_.elements[element_index].vs[local_index]; }
		inline int vertex_local_index(const int element_index, const int vertex_global_index) const {
			return (int) std::distance(mesh_.elements[element_index].vs.begin(), std::find(mesh_.elements[element_index].vs.begin(), mesh_.elements[element_index].vs.end(), vertex_global_index));
		}

		void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;
		// void element_bounday_polygon(const int index, Eigen::MatrixXd &poly) const;

		void set_boundary_tags(std::vector<int> &tags) const override;

		void point(const int global_index, Eigen::MatrixXd &pt) const override;

		bool load(const std::string &path) override;
		bool save(const std::string &path) const override;

		void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const override;

		//get nodes ids
		int face_node_id(const int face_id) const;
		int edge_node_id(const int edge_id) const;
		int vertex_node_id(const int vertex_id) const;
		bool node_id_from_face_index(const Navigation3D::Index &index, int &id) const;
		bool node_id_from_edge_index(const Navigation3D::Index &index, int &id) const;
		bool node_id_from_vertex_index(const Navigation3D::Index &index, int &id) const;


		//get nodes positions
		Eigen::MatrixXd node_from_element(const int el_id) const;
		Eigen::MatrixXd node_from_face(const int face_id) const;
		Eigen::MatrixXd node_from_face_index(const Navigation3D::Index &index) const;
		Eigen::MatrixXd node_from_edge_index(const Navigation3D::Index &index) const;
		Eigen::MatrixXd node_from_vertex_index(const Navigation3D::Index &index) const;
		Eigen::MatrixXd node_from_edge(const int edge_id) const;
		Eigen::MatrixXd node_from_vertex(const int vertex_id) const;

		//navigation wrapper
		Navigation3D::Index get_index_from_element(int hi, int lf, int lv) const;
		Navigation3D::Index get_index_from_element(int hi) const;


		// Navigation in a surface mesh
		Navigation3D::Index switch_vertex(Navigation3D::Index idx) const;
		Navigation3D::Index switch_edge(Navigation3D::Index idx) const;
		Navigation3D::Index switch_face(Navigation3D::Index idx) const;
		Navigation3D::Index switch_element(Navigation3D::Index idx) const;

		// Iterate in a mesh
		// inline Navigation3D::Index next_around_face(Navigation3D::Index idx) const { return Navigation3D::next_around_face(mesh_, idx); }
		inline Navigation3D::Index next_around_edge(Navigation3D::Index idx) const { return Navigation3D::next_around_3Dedge(mesh_, idx); }
		// inline Navigation3D::Index next_around_vertex(Navigation3D::Index idx) const { return Navigation3D::next_around_2Dvertex(mesh_, idx); }

		// inline Navigation3D::Index next_around_element(Navigation3D::Index idx) const { return Navigation3D::next_around_3Delement(mesh_, idx); }

		 inline Navigation3D::Index next_around_face_of_element(Navigation3D::Index idx) const { return Navigation3D::next_around_2Dface(mesh_, idx); }

		void create_boundary_nodes();

		// bool is_boundary_edge(int eid);
		// bool is_boundary_vertex(int vid);
		//for visualizing different types of elements
		void compute_element_tag(std::vector<ElementType> &ele_tag) const override;

		void compute_barycenter(Eigen::MatrixXd &barycenters) const override;

		void to_face_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> &to_face) const;
		void to_vertex_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> &to_vertex) const;
		void to_edge_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 12> &to_edge) const;

		// Retrieves hex vertices numbered as follows:
		//   v7────v6
		//   ╱┆    ╱│
		// v4─┼──v5 │
		//  │v3┄┄┄┼v2
		//  │╱    │╱
		// v0────v1
		std::array<int, 8> get_ordered_vertices_from_hex(const int element_index) const;

		static void geomesh_2_mesh_storage(const GEO::Mesh &gm, Mesh3DStorage &m);

		Mesh3DStorage &mesh_storge() { std::cerr<<"never user this function"<<std::endl; return mesh_; }
	private:
		Mesh3DStorage mesh_;

		std::vector<int> faces_node_id_;
		std::vector< Eigen::Matrix<double, 1, 3> > faces_node_;

		std::vector<int> edges_node_id_;
		std::vector< Eigen::Matrix<double, 1, 3> > edges_node_;

		std::vector<int> vertices_node_id_;
		std::vector< Eigen::Matrix<double, 1, 3> > vertices_node_;

		int node_id_from_vertex_index_explore(const Navigation3D::Index &index, int &id, Eigen::MatrixXd &node) const;
	};
}
