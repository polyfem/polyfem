
#include "auto_b_bases.hpp"

namespace polyfem
{
	namespace autogen
	{

		void b_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &res)
		{
			assert(p == 2);

			auto u = uv.col(0).array();
			auto v = uv.col(1).array();

			res.resize(u.size(), 1);
			switch (local_index)
			{
			case 0:
			{
				res = (-u - v + 1) * (-u - v + 1);
				break;
			}
			case 1:
			{
				res = u * u;
				break;
			}
			case 2:
			{
				res = v * v;
				break;
			}
			case 3:
			{
				res = 2 * u * (-u - v + 1);
				break;
			}
			case 4:
			{
				res = 2 * u * v;
				break;
			}
			case 5:
			{
				res = 2 * v * (-u - v + 1);
				break;
			}
			default:
				assert(false);
			}
		}

		void b_grad_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &res)
		{
			assert(p == 2);

			auto u = uv.col(0).array();
			auto v = uv.col(1).array();

			res.resize(u.size(), 2);
			switch (local_index)
			{
			case 0:
			{
				res.col(0) = 2 * u + 2 * v - 2;
				res.col(1) = 2 * u + 2 * v - 2;
				break;
			}
			case 1:
			{
				res.col(0) = 2 * u;
				res.col(1).setZero();
				break;
			}
			case 2:
			{
				res.col(0).setZero();
				res.col(1) = 2 * v;
				break;
			}
			case 3:
			{
				res.col(0) = -4 * u - 2 * v + 2;
				res.col(1) = -2 * u;
				break;
			}
			case 4:
			{
				res.col(0) = 2 * v;
				res.col(1) = 2 * u;
				break;
			}
			case 5:
			{
				res.col(0) = -2 * v;
				res.col(1) = -2 * u - 4 * v + 2;
				break;
			}
			default:
				assert(false);
			}
		}

	} // namespace autogen
} // namespace polyfem