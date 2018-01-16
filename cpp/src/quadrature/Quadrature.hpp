#ifndef QUADRATURE_HPP
#define QUADRATURE_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class Quadrature
	{
	public:
		Eigen::MatrixXd points;
		Eigen::VectorXd weights;
		double volume = 1;
	};
}

#endif //QUADRATURE_HPP
