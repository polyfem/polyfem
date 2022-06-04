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
} // namespace polyfem

#endif //QUADRATURE_HPP
