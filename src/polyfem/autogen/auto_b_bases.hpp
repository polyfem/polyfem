#include <Eigen/Dense>
#include <cassert>

namespace polyfem
{
	namespace autogen
	{

		void b_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

		void b_grad_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);

	} // namespace autogen
} // namespace polyfem