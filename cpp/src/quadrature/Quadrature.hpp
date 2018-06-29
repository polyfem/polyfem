#ifndef QUADRATURE_HPP
#define QUADRATURE_HPP

#include <Eigen/Dense>

namespace polyfem
{
	class Quadrature
	{
	public:
		Eigen::MatrixXd points;
		Eigen::VectorXd weights;
	};
}

#endif //QUADRATURE_HPP
