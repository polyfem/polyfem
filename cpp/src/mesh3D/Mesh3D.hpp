#pragma once

#include "Navigation3D.hpp"
#include "Mesh3DStorage.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class Mesh3D
	{
	public:
		void refine(const int n_refiniment);

		inline bool is_volume() const { return true; }

		inline int n_elements() const { return int(mesh_.elements.size()); }
		inline int n_pts() const { return int(mesh_.points.size()); }

		inline int n_element_vertices(const int element_index) const { return int(mesh_.elements[element_index].vs.size());}
		inline int vertex_global_index(const int element_index, const int local_index) const { return mesh_.elements[element_index].vs[local_index]; }

		double compute_mesh_size() const;

		void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts) const;
		// void element_bounday_polygon(const int index, Eigen::MatrixXd &poly) const;

		void set_boundary_tags(std::vector<int> &tags) const;

		void point(const int global_index, Eigen::MatrixXd &pt) const;

		bool load(const std::string &path);
		bool save(const std::string &path) const;

		void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1);

		//get nodes ids
		int face_node_id(const int edge_id) const;
		int edge_node_id(const int edge_id) const;
		int vertex_node_id(const int vertex_id) const;
		bool node_id_from_face_index(const Navigation3D::Index &index, int &id) const;


		//get nodes positions
		Eigen::MatrixXd node_from_element(const int el_id) const;
		Eigen::MatrixXd node_from_face(const int face_id) const;
		Eigen::MatrixXd node_from_edge_index(const Navigation3D::Index &index) const;
		Eigen::MatrixXd node_from_vertex(const int vertex_id) const;

		//navigation wrapper
		Navigation3D::Index get_index_from_element_face(int hi, int lf, int lv = 0) const;


		// Navigation in a surface mesh
		Navigation3D::Index switch_vertex(Navigation3D::Index idx) const;
		Navigation3D::Index switch_edge(Navigation3D::Index idx) const;
		Navigation3D::Index switch_face(Navigation3D::Index idx) const;
		Navigation3D::Index switch_element(Navigation3D::Index idx) const;

		// Iterate in a mesh
		inline Navigation3D::Index next_around_element(Navigation3D::Index idx) const { return Navigation3D::next_around_element(mesh_, idx); }
		inline Navigation3D::Index next_around_face(Navigation3D::Index idx) const { return Navigation3D::next_around_face(mesh_, idx); }
		inline Navigation3D::Index next_around_edge(Navigation3D::Index idx) const { return Navigation3D::next_around_edge(mesh_, idx); }
		inline Navigation3D::Index next_around_vertex(Navigation3D::Index idx) const { return Navigation3D::next_around_vertex(mesh_, idx); }

		void create_boundary_nodes();
	private:

		Mesh3DStorage mesh_;
	};
}
