#ifndef ASSEMBLY_VALUES_HPP
#define ASSEMBLY_VALUES_HPP


#include "Quadrature.hpp"
#include "Basis.hpp"

#include <Eigen/Dense>
#include <iostream>

namespace poly_fem
{
	class AssemblyValues
	{
	public:
		std::vector< Local2Global > global;
		Eigen::MatrixXd val;
		Eigen::MatrixXd grad;

		Eigen::MatrixXd grad_t_m;


		void finalize()
		{
			grad_t_m.resize(grad.rows(), grad.cols());
		}
	};
}

#endif //ASSEMBLY_VALUES_HPP
