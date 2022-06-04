#pragma once

#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

namespace polyfem
{
	namespace mesh
	{
		struct Vertex
		{
			int id;
			std::vector<double> v;
			std::vector<uint32_t> neighbor_vs;
			std::vector<uint32_t> neighbor_es;
			std::vector<uint32_t> neighbor_fs;
			std::vector<uint32_t> neighbor_hs;

			bool boundary;
			bool boundary_hex;
		};
		struct Edge
		{
			int id;
			std::vector<uint32_t> vs;
			std::vector<uint32_t> neighbor_fs;
			std::vector<uint32_t> neighbor_hs;

			bool boundary;
			bool boundary_hex;
		};
		struct Face
		{
			int id;
			std::vector<uint32_t> vs;
			std::vector<uint32_t> es;
			std::vector<uint32_t> neighbor_hs;
			bool boundary;
			bool boundary_hex;
		};

		struct Element
		{
			int id;
			std::vector<uint32_t> vs;
			std::vector<uint32_t> es;
			std::vector<uint32_t> fs;
			std::vector<bool> fs_flag;
			bool hex = false;
			std::vector<double> v_in_Kernel;
		};

		enum MeshType
		{
			Tri = 0,
			Qua,
			HSur,
			Tet,
			Hyb,
			Hex
		};

		struct Mesh3DStorage
		{
			MeshType type;
			Eigen::MatrixXd points;
			std::vector<Vertex> vertices;
			std::vector<Edge> edges;
			std::vector<Face> faces;
			std::vector<Element> elements;

			Eigen::MatrixXi EV;              //EV(2, ne)
			Eigen::MatrixXi FV, FE, FH, FHi; //FV (3, nf), FE(3, nf), FH (2, nf), FHi(2, nf)
			Eigen::MatrixXi HV, HF;          //HV(4, nh), HE(6, nh), HF(4, nh)
		};

		struct Mesh_Quality
		{
			std::string Name;
			double min_Jacobian;
			double ave_Jacobian;
			double deviation_Jacobian;
			VectorXd V_Js;
			VectorXd H_Js;
			VectorXd Num_Js;
			int32_t V_num, H_num;
		};
	} // namespace mesh
} // namespace polyfem
