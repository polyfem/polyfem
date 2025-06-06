#include "prism_bases.hpp"

#include "auto_p_bases.hpp"
#include "auto_q_bases.hpp"

namespace polyfem
{
	namespace autogen
	{
		namespace
		{
			int index_mapping(const int p, const int q, const int i)
			{
				// base vertices
				if (i < 3)
					return i;
				const int npe = (p - 1) * 3;
				const int npf = std::max(0, (p - 1) * (p - 2) / 2);
				const int nl = 3 + npe + npf;
				const int nqe = q - 1;

				// top vertices
				if (i >= nl * q && i < nl * q + 3)
					return i - nl * q + 3;

				// base edges
				if (i >= 3 && i < 3 + npe)
					return i - 3 + 6;

				// top edges
				if (i >= nl * q + 3 && i < nl * q + 3 + npe)
					return i - nl * q - 3 + 6 + npe;

				// first vertical edge
				if (i % nl == 0)
					return 6 + 2 * npe + i / nl - 1;

				// second vertical edge
				if (i % nl == 1)
					return 6 + 2 * npe + i / nl - 1 + nqe;

				// third vertical edge
				if (i % nl == 2)
					return 6 + 2 * npe + i / nl - 1 + 2 * nqe;

				// base face
				if (i >= 3 + npe && i < 3 + npe + npf)
					return 6 + 2 * npe + 3 * nqe + i - (3 + npe);

				// top face
				if (i >= nl * q + 3 + npe && i < nl * q + 3 + npe + npf)
					return 6 + 2 * npe + 3 * nqe + i - (nl * q + 3 + npe) + npf;

				int mod = i % nl - 3;
				const int div = i / nl - 1;

				// quad faces
				if (mod < npe)
					return 2 * nl + nqe * 3 + (mod % (npe / 3)) + (mod / (npe / 3)) * npe / 3 * nqe + div * npe / 3;
				mod -= npe;
				// volume
				if (mod < npf)
					return 2 * nl + nqe * 3 + npe * nqe + mod + npf * div;

				// assert false
				return -1;
			}

			int inverse_index_mapping(const int p, const int q, const int li)
			{
				Eigen::MatrixXd tmpp, tmpq;

				p_nodes_2d(p, tmpp);
				q_nodes_1d(q, tmpq);

				int index = 0;

				for (int i = 0; i < tmpq.rows(); ++i)
				{
					for (int j = 0; j < tmpp.rows(); ++j)
					{
						if (index_mapping(p, q, index) == li)
							return index;
						++index;
					}
				}
				assert(false);
				return -1;
			}
		} // namespace

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
					val.row(index_mapping(p, q, index)) << tmpp(j, 0), tmpp(j, 1), tmpq(i, 0);
					++index;
				}
			}
		}

		void prism_basis_value_3d(const int p, const int q, const int li, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			assert(uv.cols() == 3);

			Eigen::MatrixXd tmpp, tmpq;

			Eigen::MatrixXd nn;
			p_nodes_2d(p, nn);
			const int n_p = nn.rows();

			const int local_index = inverse_index_mapping(p, q, li);

			p_basis_value_2d(false, p, local_index % n_p, uv.leftCols(2), tmpp);
			q_basis_value_1d(q, local_index / n_p, uv.col(2), tmpq);

			val = tmpp.array() * tmpq.array();
		}

		void prism_grad_basis_value_3d(const int p, const int q, const int li, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			assert(uv.cols() == 3);

			Eigen::MatrixXd nn;
			p_nodes_2d(p, nn);
			const int n_p = nn.rows();

			Eigen::MatrixXd tmpp, tmpq, tmpgp, tmpgq;

			const int local_index = inverse_index_mapping(p, q, li);

			p_basis_value_2d(false, p, local_index % n_p, uv.leftCols(2), tmpp);
			q_basis_value_1d(q, local_index / n_p, uv.col(2), tmpq);

			p_grad_basis_value_2d(false, p, local_index % n_p, uv.leftCols(2), tmpgp);
			q_grad_basis_value_1d(q, local_index / n_p, uv.col(2), tmpgq);

			val.resize(uv.rows(), 3);
			val.col(0) = tmpgp.col(0).array() * tmpq.array();
			val.col(1) = tmpgp.col(1).array() * tmpq.array();
			val.col(2) = tmpp.array() * tmpgq.array();
		}
	} // namespace autogen
} // namespace polyfem