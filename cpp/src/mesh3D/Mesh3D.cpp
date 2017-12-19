#include "Mesh3D.hpp"

namespace poly_fem
{
	void Mesh3D::refine(const int n_refiniment)
	{
		//TODO implement me
		assert(false);
	}


	double Mesh3D::compute_mesh_size() const
	{
		//TODO implement me
		assert(false);
	}

	void Mesh3D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts) const
	{
		//TODO implement me
		assert(false);
	}

	void Mesh3D::set_boundary_tags(std::vector<int> &tags) const
	{
		//TODO implement me
		assert(false);
	}

	void Mesh3D::point(const int global_index, Eigen::MatrixXd &pt) const
	{
		//TODO implement me
		assert(false);
	}

	bool Mesh3D::load(const std::string &path)
	{
		//TODO implement me
		assert(false);

		Navigation3D::prepare_mesh(mesh_);
		Navigation3D::build_connectivity(mesh_);
	}
	bool Mesh3D::save(const std::string &path) const
	{
		//TODO implement me
		assert(false);
	}

	void Mesh3D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1)
	{
		//TODO implement me
		assert(false);
	}

		//get nodes ids
	int Mesh3D::edge_node_id(const int edge_id) const
	{
		//TODO implement me
		assert(false);
	}
	int Mesh3D::vertex_node_id(const int vertex_id) const
	{
		//TODO implement me
		assert(false);
	}
	bool Mesh3D::node_id_from_edge_index(const Navigation3D::Index &index, int &id) const
	{
		//TODO implement me
		assert(false);
	}


		//get nodes positions
	Eigen::MatrixXd Mesh3D::node_from_edge_index(const Navigation3D::Index &index) const
	{
		//TODO implement me
		assert(false);
	}
	Eigen::MatrixXd Mesh3D::node_from_face(const int face_id) const
	{
		//TODO implement me
		assert(false);
	}
	Eigen::MatrixXd Mesh3D::node_from_vertex(const int vertex_id) const
	{
		//TODO implement me
		assert(false);
	}

		//navigation wrapper
	Navigation3D::Index Mesh3D::get_index_from_face(int f, int lv = 0) const
	{
		//TODO implement me
		assert(false);
	}

		// Navigation in a surface mesh
	Navigation3D::Index Mesh3D::switch_vertex(Navigation3D::Index idx) const
	{
		//TODO implement me
		assert(false);
	}
	Navigation3D::Index Mesh3D::switch_edge(Navigation3D::Index idx) const
	{
		//TODO implement me
		assert(false);
	}
	Navigation3D::Index Mesh3D::switch_face(Navigation3D::Index idx) const
	{
		//TODO implement me
		assert(false);
	}
}
