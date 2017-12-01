#ifndef QUADRATURE_HPP
#define QUADRATURE_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class Quadrature
	{
	public:
		Eigen::MatrixXd points;
		Eigen::MatrixXd weights;
	};
}

#endif //QUADRATURE_HPP