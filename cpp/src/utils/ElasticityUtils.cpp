#include "ElasticityUtils.hpp"


namespace poly_fem
{
	double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress)
	{
		double von_mises_stress =  0.5 * ( stress(0, 0) - stress(1, 1) ) * ( stress(0, 0) - stress(1, 1) ) + 3.0  *  stress(0, 1) * stress(0, 1);

		if(stress.rows() == 3)
		{
			von_mises_stress += 0.5 * (stress(2, 2) - stress(1, 1)) * (stress(2, 2) - stress(1, 1)) + 3.0  * stress(2, 1) * stress(2, 1);
			von_mises_stress += 0.5 * (stress(2, 2) - stress(0, 0)) * (stress(2, 2) - stress(0, 0)) + 3.0  * stress(2, 0) * stress(2, 0);
		}

		von_mises_stress = sqrt( fabs(von_mises_stress) );

		return von_mises_stress;
	}

}