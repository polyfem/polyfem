#ifndef MESH_HPP
#define MESH_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class Mesh
	{
	public:
		Eigen::MatrixXd pts;
		Eigen::MatrixXi els;

		int n_x;
		int n_y;
		int n_z;

		bool is_volume;
	};
}

#endif //MESH_HPP
