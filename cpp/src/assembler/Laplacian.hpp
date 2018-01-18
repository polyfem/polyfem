#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class Laplacian
	{
	public:
		Eigen::Matrix<double, 1, 1> assemble(const Eigen::MatrixXd &gradi, const Eigen::MatrixXd &gradj, const Eigen::VectorXd &da) const;

		inline int size() const { return 1; }
	};
}

#endif //LAPLACIAN_HPP
