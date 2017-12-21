#pragma once

#include "Mesh.hpp"

#include "Navigation3D.hpp"
#include "Mesh3DStorage.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class Mesh3D : public Mesh
	{
	public:
		void refine(const int n_refiniment) override;

		inline bool is_volume() const override { return true; }

		inline int n_elements() const override { return int(mesh_.elements.size()); }
		inline int n_pts() const override { return int(mesh_.points.size()); }

		inline int n_element_vertices(const int element_index) const override { return int(mesh_.elements[element_index].vs.size());}
		inline int vertex_global_index(const int element_index, const int local_index) const override { return mesh_.elements[element_index].vs[local_index]; }

		double compute_mesh_size() const override;

		void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;
		// void element_bounday_polygon(const int index, Eigen::MatrixXd &poly) const;

		void set_boundary_tags(std::vector<int> &tags) const override;

		void point(const int global_index, Eigen::MatrixXd &pt) const override;

		bool load(const std::string &path) override;
		bool save(const std::string &path) const override;

		void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const override;

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
		Navigation3D::Index get_index_from_element(int hi, int lf = 0, int lv = 0) const;


		// Navigation in a surface mesh
		Navigation3D::Index switch_vertex(Navigation3D::Index idx) const;
		Navigation3D::Index switch_edge(Navigation3D::Index idx) const;
		Navigation3D::Index switch_face(Navigation3D::Index idx) const;
		Navigation3D::Index switch_element(Navigation3D::Index idx) const;

		// Iterate in a mesh
		// inline Navigation3D::Index next_around_face(Navigation3D::Index idx) const { return Navigation3D::next_around_face(mesh_, idx); }
		inline Navigation3D::Index next_around_edge(Navigation3D::Index idx) const { return Navigation3D::next_around_3Dedge(mesh_, idx); }
		// inline Navigation3D::Index next_around_vertex(Navigation3D::Index idx) const { return Navigation3D::next_around_2Dvertex(mesh_, idx); }

		inline Navigation3D::Index next_around_element(Navigation3D::Index idx) const { return Navigation3D::next_around_3Delement(mesh_, idx); }

		 inline Navigation3D::Index next_around_face_of_element(Navigation3D::Index idx) const { return Navigation3D::next_around_2Dface(mesh_, idx); }

		void create_boundary_nodes();
		//for visualizing different types of elements
		void compute_element_tag(std::vector<ElementType> &ele_tag) const override;

		void compute_barycenter(Eigen::MatrixXd &barycenters) const override;
	private:

		Mesh3DStorage mesh_;
	};
}
