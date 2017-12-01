#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class Laplacian
	{
	public:
		void assemble(const Eigen::MatrixXd &gradi, const Eigen::MatrixXd &gradj, Eigen::MatrixXd &res) const;

		inline int size() const { return 1; }
	};
}

#endif //LAPLACIAN_HPP
