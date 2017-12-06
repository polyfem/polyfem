#ifndef ASSEMBLY_VALUES_HPP
#define ASSEMBLY_VALUES_HPP

#include <Eigen/Dense>
#include "Quadrature.hpp"

namespace poly_fem
{
	class AssemblyValues
	{
	public:
		int global_index;
		Eigen::MatrixXd val;
		Eigen::MatrixXd grad;

		Eigen::MatrixXd grad_t_m;


		void finalize(const Eigen::MatrixXd jac_it)
		{
			grad_t_m.resize(grad.rows(), grad.cols());

			for(long j=0; j<grad.rows(); ++j)
			{
				grad_t_m.row(j)=grad.row(j)*jac_it;
			}
		}
	};
}

#endif //ASSEMBLY_VALUES_HPP