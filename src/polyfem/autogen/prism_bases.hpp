
#include <Eigen/Dense>

namespace polyfem
{
	namespace autogen
	{
		void prism_nodes_3d(const int p, const int q, Eigen::MatrixXd &val);
		void prism_basis_value_3d(const int p, const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
		void prism_grad_basis_value_3d(const int p, const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val);
	} // namespace autogen
} // namespace polyfem