#include "prism_bases.hpp"

#include "auto_p_bases.hpp"
#include "auto_q_bases.hpp"

namespace polyfem
{
	namespace autogen
	{
		void prism_nodes_3d(const int p, const int q, Eigen::MatrixXd &val)
		{
			Eigen::MatrixXd tmpp, tmpq;

			p_nodes_2d(p, tmpp);
			q_nodes_1d(q, tmpq);

			val.resize(tmpp.rows() * tmpq.rows(), 3);
			int index = 0;

			for (int i = 0; i < tmpq.rows(); ++i)
			{
				for (int j = 0; j < tmpp.rows(); ++j)
				{
					val.row(index) << tmpp(j, 0), tmpp(j, 1), tmpq(i, 0);
					++index;
				}
			}
		}

		void prism_basis_value_3d(const int p, const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			assert(uv.cols() == 3);

			Eigen::MatrixXd tmpp, tmpq;

			Eigen::MatrixXd nn;
			p_nodes_2d(p, nn);
			const int n_p = nn.rows();

			p_basis_value_2d(p, local_index % n_p, uv.leftCols(2), tmpp);
			q_basis_value_1d(q, local_index / n_p, uv.col(2), tmpq);

			val = tmpp.array() * tmpq.array();
		}

		void prism_grad_basis_value_3d(const int p, const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			assert(uv.cols() == 3);

			Eigen::MatrixXd nn;
			p_nodes_2d(p, nn);
			const int n_p = nn.rows();

			Eigen::MatrixXd tmpp, tmpq, tmpgp, tmpgq;

			p_basis_value_2d(p, local_index % n_p, uv.leftCols(2), tmpp);
			q_basis_value_1d(q, local_index / n_p, uv.col(2), tmpq);

			p_grad_basis_value_2d(p, local_index % n_p, uv.leftCols(2), tmpgp);
			q_grad_basis_value_1d(q, local_index / n_p, uv.col(2), tmpgq);

			val.resize(uv.rows(), 3);
			val.col(0) = tmpgp.col(0).array() * tmpq.array();
			val.col(1) = tmpgp.col(1).array() * tmpq.array();
			val.col(2) = tmpp.array() * tmpgq.array();
		}
	} // namespace autogen
} // namespace polyfem