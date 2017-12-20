#ifndef MESH_STORAGE_HPP__
#define MESH_STORAGE_HPP__

#include <vector>
#include <Eigen/Dense>

namespace poly_fem
{
	enum ElementType {
		Regular_Hex = 0,//an interior hex, all its 12 edges are non-singular
		Onesingular_Hex,//an interior hex, one out of its 12 edges is singular
		Boundary_Hex,//either on boundary or attaching to a Non_hex
		Non_Hex,
	};

	struct Vertex
	{
		int id;
		std::vector<double> v;
		std::vector<uint32_t> neighbor_vs;
		std::vector<uint32_t> neighbor_es;
		std::vector<uint32_t> neighbor_fs;
		std::vector<uint32_t> neighbor_hs;

		bool boundary;
	};
	struct Edge
	{
		int id;
		std::vector<uint32_t> vs;
		std::vector<uint32_t> neighbor_fs;
		std::vector<uint32_t> neighbor_hs;

		bool boundary;
	};
	struct Face
	{
		int id;
		std::vector<uint32_t> vs;
		std::vector<uint32_t> es;
		std::vector<uint32_t> neighbor_hs;
		bool boundary;
	};

	struct Element
	{
		int id;
		std::vector<uint32_t> vs;
		std::vector<uint32_t> es;
		std::vector<uint32_t> fs;
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
}

#endif
