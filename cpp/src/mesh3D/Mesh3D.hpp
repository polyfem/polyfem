#pragma once

#include "Navigation3D.hpp"
#include <Eigen/Dense>
#include <vector>

namespace poly_fem
{
	class Mesh3D
	{
	public:
		struct Vertex
		{
			int id;
			std::vector<double> v;
			std::vector<int> neighbor_vs;
			std::vector<int> neighbor_es;
			std::vector<int> neighbor_fs;
			std::vector<int> neighbor_hs;

			bool boundary;
		};
		struct Edge
		{
			int id;
			std::vector<int> vs;
			std::vector<int> neighbor_fs;
			std::vector<int> neighbor_hs;

			bool boundary;
		};
		struct Face
		{
			int id;
			std::vector<int> vs;
			std::vector<int> es;
			std::vector<int> neighbor_hs;
			bool boundary;
		};

		struct Element
		{
			int id;
			std::vector<int> vs;
			std::vector<int> es;
			std::vector<int> fs;
			std::vector<bool> fs_flag;
			bool hex = false;
		};

		enum MeshType {
			Tri = 0,
			Qua,
			Tet,
			Hyb,
			Hex,
			PHr
		};

		struct Mesh3DStorage
		{
			MeshType type;
			Eigen::MatrixXd points;
			std::vector<Vertex> vertices;
			std::vector<Edge> edges;
			std::vector<Face> faces;
			std::vector<Element> elements;
		};







		void refine(const int n_refiniment);

		inline bool is_volume() const { return true; }

		inline int n_elements() const { return int(elements.size()); }
		inline int n_pts() const { return int(points.size()); }

		inline int n_element_vertices(const int element_index) const { return int(elements[element_index].vs.size());}
		inline int vertex_global_index(const int element_index, const int local_index) const { return elements[element_index].vs[local_index]; }

		double compute_mesh_size() const;

		void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts) const;
		// void element_bounday_polygon(const int index, Eigen::MatrixXd &poly) const;

		void set_boundary_tags(std::vector<int> &tags) const;

		void point(const int global_index, Eigen::MatrixXd &pt) const;

		bool load(const std::string &path);
		bool save(const std::string &path) const;

		void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1);

		//get nodes ids
		int edge_node_id(const int edge_id) const;
		int vertex_node_id(const int vertex_id) const;
		bool node_id_from_edge_index(const Navigation3D::Index &index, int &id) const;


		//get nodes positions
		Eigen::MatrixXd node_from_edge_index(const Navigation3D::Index &index) const;
		Eigen::MatrixXd node_from_face(const int face_id) const;
		Eigen::MatrixXd node_from_vertex(const int vertex_id) const;

		//navigation wrapper
		Navigation3D::Index get_index_from_face(int f, int lv = 0) const;

		// Navigation in a surface mesh
		Navigation3D::Index switch_vertex(Navigation3D::Index idx) const;
		Navigation3D::Index switch_edge(Navigation3D::Index idx) const;
		Navigation3D::Index switch_face(Navigation3D::Index idx) const;

		// Iterate in a mesh
		inline Navigation3D::Index next_around_face(Navigation3D::Index idx) const { return switch_edge(switch_vertex(idx)); }
		inline Navigation3D::Index next_around_edge(Navigation3D::Index idx) const { return switch_vertex(switch_face(idx)); }
		inline Navigation3D::Index next_around_vertex(Navigation3D::Index idx) const { return switch_face(switch_edge(idx)); }

		void create_boundary_nodes();
	private:
		

		Mesh3DStorage mesh_;
	};
}
