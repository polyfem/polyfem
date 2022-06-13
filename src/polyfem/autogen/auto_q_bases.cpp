#include "auto_q_bases.hpp"

namespace polyfem
{
	namespace autogen
	{
		namespace
		{
			void q_0_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				result_0.resize(x.size(), 1);
				switch (local_index)
				{
				case 0:
				{
					result_0.setOnes();
				}
				break;
				default:
					assert(false);
				}
			}
			void q_0_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				val.resize(uv.rows(), uv.cols());
				Eigen::ArrayXd result_0(uv.rows());
				switch (local_index)
				{
				case 0:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void q_0_nodes_2d(Eigen::MatrixXd &res)
			{
				res.resize(1, 2);
				res << 0.5, 0.5;
			}

			void q_1_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = 1.0 * (x - 1) * (y - 1);
				}
				break;
				case 1:
				{
					result_0 = -1.0 * x * (y - 1);
				}
				break;
				case 2:
				{
					result_0 = 1.0 * x * y;
				}
				break;
				case 3:
				{
					result_0 = -1.0 * y * (x - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void q_1_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				val.resize(uv.rows(), uv.cols());
				Eigen::ArrayXd result_0(uv.rows());
				switch (local_index)
				{
				case 0:
				{
					{
						result_0 = 1.0 * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 1.0 * (x - 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = 1.0 * (1 - y);
						val.col(0) = result_0;
					}
					{
						result_0 = -1.0 * x;
						val.col(1) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						result_0 = 1.0 * y;
						val.col(0) = result_0;
					}
					{
						result_0 = 1.0 * x;
						val.col(1) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						result_0 = -1.0 * y;
						val.col(0) = result_0;
					}
					{
						result_0 = 1.0 * (1 - x);
						val.col(1) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void q_1_nodes_2d(Eigen::MatrixXd &res)
			{
				res.resize(4, 2);
				res << 0, 0,
					1, 0,
					1, 1,
					0, 1;
			}

			void q_2_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = 1.0 * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0);
				}
				break;
				case 1:
				{
					result_0 = 1.0 * x * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0);
				}
				break;
				case 2:
				{
					result_0 = 1.0 * x * y * (2.0 * x - 1.0) * (2.0 * y - 1.0);
				}
				break;
				case 3:
				{
					result_0 = 1.0 * y * (x - 1) * (2.0 * x - 1.0) * (2.0 * y - 1.0);
				}
				break;
				case 4:
				{
					result_0 = -4.0 * x * (x - 1) * (y - 1) * (2.0 * y - 1.0);
				}
				break;
				case 5:
				{
					result_0 = -4.0 * x * y * (2.0 * x - 1.0) * (y - 1);
				}
				break;
				case 6:
				{
					result_0 = -4.0 * x * y * (x - 1) * (2.0 * y - 1.0);
				}
				break;
				case 7:
				{
					result_0 = -4.0 * y * (x - 1) * (2.0 * x - 1.0) * (y - 1);
				}
				break;
				case 8:
				{
					result_0 = 16.0 * x * y * (x - 1) * (y - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void q_2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				val.resize(uv.rows(), uv.cols());
				Eigen::ArrayXd result_0(uv.rows());
				switch (local_index)
				{
				case 0:
				{
					{
						result_0 = (4.0 * x - 3.0) * (y - 1) * (2.0 * y - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = (x - 1) * (2.0 * x - 1.0) * (4.0 * y - 3.0);
						val.col(1) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = (4.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = x * (2.0 * x - 1.0) * (4.0 * y - 3.0);
						val.col(1) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						result_0 = y * (4.0 * x - 1.0) * (2.0 * y - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = x * (2.0 * x - 1.0) * (4.0 * y - 1.0);
						val.col(1) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						result_0 = y * (4.0 * x - 3.0) * (2.0 * y - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = (x - 1) * (2.0 * x - 1.0) * (4.0 * y - 1.0);
						val.col(1) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						result_0 = -4.0 * (2 * x - 1) * (y - 1) * (2.0 * y - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * (x - 1) * (16.0 * y - 12.0);
						val.col(1) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						result_0 = -y * (16.0 * x - 4.0) * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4.0 * x * (2.0 * x - 1.0) * (2 * y - 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						result_0 = -4.0 * y * (2 * x - 1) * (2.0 * y - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * (x - 1) * (16.0 * y - 4.0);
						val.col(1) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						result_0 = -y * (16.0 * x - 12.0) * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4.0 * (x - 1) * (2.0 * x - 1.0) * (2 * y - 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 8:
				{
					{
						result_0 = 16.0 * y * (2 * x - 1) * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 16.0 * x * (x - 1) * (2 * y - 1);
						val.col(1) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void q_2_nodes_2d(Eigen::MatrixXd &res)
			{
				res.resize(9, 2);
				res << 0, 0,
					1, 0,
					1, 1,
					0, 1,
					1.0 / 2.0, 0,
					1, 1.0 / 2.0,
					1.0 / 2.0, 1,
					0, 1.0 / 2.0,
					1.0 / 2.0, 1.0 / 2.0;
			}

			void q_3_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = 1.0 * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0);
				}
				break;
				case 1:
				{
					result_0 = -1.0 * x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0);
				}
				break;
				case 2:
				{
					result_0 = 1.0 * x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996);
				}
				break;
				case 3:
				{
					result_0 = -1.0 * y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996);
				}
				break;
				case 4:
				{
					result_0 = -4.4999999999999991 * x * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0);
				}
				break;
				case 5:
				{
					result_0 = 4.4999999999999991 * x * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0);
				}
				break;
				case 6:
				{
					result_0 = 4.4999999999999991 * x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 2.0);
				}
				break;
				case 7:
				{
					result_0 = -4.4999999999999991 * x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 1.0);
				}
				break;
				case 8:
				{
					result_0 = -4.4999999999999991 * x * y * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996);
				}
				break;
				case 9:
				{
					result_0 = 4.4999999999999991 * x * y * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996);
				}
				break;
				case 10:
				{
					result_0 = 4.4999999999999991 * y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0);
				}
				break;
				case 11:
				{
					result_0 = -4.4999999999999991 * y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0);
				}
				break;
				case 12:
				{
					result_0 = 20.249999999999993 * x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 2.0);
				}
				break;
				case 13:
				{
					result_0 = -20.249999999999993 * x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 1.0);
				}
				break;
				case 14:
				{
					result_0 = -20.249999999999993 * x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0);
				}
				break;
				case 15:
				{
					result_0 = 20.249999999999993 * x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0);
				}
				break;
				default:
					assert(false);
				}
			}
			void q_3_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				val.resize(uv.rows(), uv.cols());
				Eigen::ArrayXd result_0(uv.rows());
				switch (local_index)
				{
				case 0:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * x;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = -(y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = -x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * x;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = y * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * y;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(1) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = -y * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * y;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = -(x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(1) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 8.9999999999999982;
						result_0 = -(y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = -x * (x - 1) * (3.0 * x - 2.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 4.4999999999999991;
						result_0 = (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = x * (x - 1) * (3.0 * x - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = y * (y - 1) * (3.0 * y - 2.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * x + 6.7499999999999973 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 8.9999999999999982;
						result_0 = x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = -y * (y - 1) * (3.0 * y - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * x + 6.7499999999999973 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 4.4999999999999991;
						result_0 = -x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
				}
				break;
				case 8:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 4.4999999999999991;
						result_0 = -y * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = -x * (x - 1) * (3.0 * x - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * y + 6.7499999999999973 * helper_1 * y);
						val.col(1) = result_0;
					}
				}
				break;
				case 9:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 8.9999999999999982;
						result_0 = y * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = x * (x - 1) * (3.0 * x - 2.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * y + 6.7499999999999973 * helper_1 * y);
						val.col(1) = result_0;
					}
				}
				break;
				case 10:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = y * (y - 1) * (3.0 * y - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 4.4999999999999991;
						result_0 = (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
				}
				break;
				case 11:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = -y * (y - 1) * (3.0 * y - 2.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 8.9999999999999982;
						result_0 = -(x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
				}
				break;
				case 12:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 40.499999999999986;
						result_0 = y * (y - 1) * (3.0 * y - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 40.499999999999986;
						result_0 = x * (x - 1) * (3.0 * x - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
				}
				break;
				case 13:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 40.499999999999986;
						result_0 = -y * (y - 1) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 20.249999999999993;
						result_0 = -x * (x - 1) * (3.0 * x - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
				}
				break;
				case 14:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 20.249999999999993;
						result_0 = -y * (y - 1) * (3.0 * y - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 40.499999999999986;
						result_0 = -x * (x - 1) * (3.0 * x - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
				}
				break;
				case 15:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 20.249999999999993;
						result_0 = y * (y - 1) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 20.249999999999993;
						result_0 = x * (x - 1) * (3.0 * x - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void q_3_nodes_2d(Eigen::MatrixXd &res)
			{
				res.resize(16, 2);
				res << 0, 0,
					1, 0,
					1, 1,
					0, 1,
					1.0 / 3.0, 0,
					2.0 / 3.0, 0,
					1, 1.0 / 3.0,
					1, 2.0 / 3.0,
					2.0 / 3.0, 1,
					1.0 / 3.0, 1,
					0, 2.0 / 3.0,
					0, 1.0 / 3.0,
					1.0 / 3.0, 1.0 / 3.0,
					1.0 / 3.0, 2.0 / 3.0,
					2.0 / 3.0, 1.0 / 3.0,
					2.0 / 3.0, 2.0 / 3.0;
			}

			void q_m2_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = -1.0 * (x - 1) * (y - 1) * (2 * x + 2 * y - 1);
				}
				break;
				case 1:
				{
					result_0 = 1.0 * x * (y - 1) * (-2 * x + 2 * y + 1);
				}
				break;
				case 2:
				{
					result_0 = x * y * (2.0 * x + 2.0 * y - 3.0);
				}
				break;
				case 3:
				{
					result_0 = 1.0 * y * (x - 1) * (2 * x - 2 * y + 1);
				}
				break;
				case 4:
				{
					result_0 = 4 * x * (x - 1) * (y - 1);
				}
				break;
				case 5:
				{
					result_0 = -4 * x * y * (y - 1);
				}
				break;
				case 6:
				{
					result_0 = -4 * x * y * (x - 1);
				}
				break;
				case 7:
				{
					result_0 = 4 * y * (x - 1) * (y - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void q_m2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				val.resize(uv.rows(), uv.cols());
				Eigen::ArrayXd result_0(uv.rows());
				switch (local_index)
				{
				case 0:
				{
					{
						result_0 = -(y - 1) * (4.0 * x + 2.0 * y - 3.0);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 4.0 * y;
						result_0 = -helper_0 * x + helper_0 - 2.0 * pow(x, 2) + 5.0 * x - 3.0;
						val.col(1) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = (y - 1) * (-4.0 * x + 2.0 * y + 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * (2.0 * x - 4.0 * y + 1.0);
						val.col(1) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						result_0 = y * (4.0 * x + 2.0 * y - 3.0);
						val.col(0) = result_0;
					}
					{
						result_0 = x * (2.0 * x + 4.0 * y - 3.0);
						val.col(1) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						result_0 = -y * (-4.0 * x + 2.0 * y + 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = (x - 1) * (2.0 * x - 4.0 * y + 1.0);
						val.col(1) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						result_0 = 4 * (2 * x - 1) * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * x * (x - 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						result_0 = -4 * y * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * x * (2 * y - 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						result_0 = -4 * y * (2 * x - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * x * (x - 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						result_0 = 4 * y * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * (x - 1) * (2 * y - 1);
						val.col(1) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void q_m2_nodes_2d(Eigen::MatrixXd &res)
			{
				res.resize(8, 2);
				res << 0.0, 0.0,
					1.0, 0.0,
					1.0, 1.0,
					0.0, 1.0,
					0.5, 0.0,
					1.0, 0.5,
					0.5, 1.0,
					0.0, 0.5;
			}

		} // namespace

		void q_nodes_2d(const int q, Eigen::MatrixXd &val)
		{
			switch (q)
			{
			case 0:
				q_0_nodes_2d(val);
				break;
			case 1:
				q_1_nodes_2d(val);
				break;
			case 2:
				q_2_nodes_2d(val);
				break;
			case 3:
				q_3_nodes_2d(val);
				break;
			case -2:
				q_m2_nodes_2d(val);
				break;
			default:
				assert(false);
			}
		}
		void q_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			switch (q)
			{
			case 0:
				q_0_basis_value_2d(local_index, uv, val);
				break;
			case 1:
				q_1_basis_value_2d(local_index, uv, val);
				break;
			case 2:
				q_2_basis_value_2d(local_index, uv, val);
				break;
			case 3:
				q_3_basis_value_2d(local_index, uv, val);
				break;
			case -2:
				q_m2_basis_value_2d(local_index, uv, val);
				break;
			default:
				assert(false);
			}
		}

		void q_grad_basis_value_2d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			switch (q)
			{
			case 0:
				q_0_basis_grad_value_2d(local_index, uv, val);
				break;
			case 1:
				q_1_basis_grad_value_2d(local_index, uv, val);
				break;
			case 2:
				q_2_basis_grad_value_2d(local_index, uv, val);
				break;
			case 3:
				q_3_basis_grad_value_2d(local_index, uv, val);
				break;
			case -2:
				q_m2_basis_grad_value_2d(local_index, uv, val);
				break;
			default:
				assert(false);
			}
		}

		namespace
		{
			void q_0_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				result_0.resize(x.size(), 1);
				switch (local_index)
				{
				case 0:
				{
					result_0.setOnes();
				}
				break;
				default:
					assert(false);
				}
			}
			void q_0_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				val.resize(uv.rows(), uv.cols());
				Eigen::ArrayXd result_0(uv.rows());
				switch (local_index)
				{
				case 0:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void q_0_nodes_3d(Eigen::MatrixXd &res)
			{
				res.resize(1, 3);
				res << 0.5, 0.5, 0.5;
			}

			void q_1_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = -1.0 * (x - 1) * (y - 1) * (z - 1);
				}
				break;
				case 1:
				{
					result_0 = 1.0 * x * (y - 1) * (z - 1);
				}
				break;
				case 2:
				{
					result_0 = -1.0 * x * y * (z - 1);
				}
				break;
				case 3:
				{
					result_0 = 1.0 * y * (x - 1) * (z - 1);
				}
				break;
				case 4:
				{
					result_0 = 1.0 * z * (x - 1) * (y - 1);
				}
				break;
				case 5:
				{
					result_0 = -1.0 * x * z * (y - 1);
				}
				break;
				case 6:
				{
					result_0 = 1.0 * x * y * z;
				}
				break;
				case 7:
				{
					result_0 = -1.0 * y * z * (x - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void q_1_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				val.resize(uv.rows(), uv.cols());
				Eigen::ArrayXd result_0(uv.rows());
				switch (local_index)
				{
				case 0:
				{
					{
						result_0 = -1.0 * (y - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -1.0 * (x - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -1.0 * (x - 1) * (y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = 1.0 * (y - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 1.0 * x * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 1.0 * x * (y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						result_0 = -1.0 * y * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -1.0 * x * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -1.0 * x * y;
						val.col(2) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						result_0 = 1.0 * y * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 1.0 * (x - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 1.0 * y * (x - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						result_0 = 1.0 * z * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 1.0 * z * (x - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 1.0 * (x - 1) * (y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						result_0 = -1.0 * z * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -1.0 * x * z;
						val.col(1) = result_0;
					}
					{
						result_0 = -1.0 * x * (y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						result_0 = 1.0 * y * z;
						val.col(0) = result_0;
					}
					{
						result_0 = 1.0 * x * z;
						val.col(1) = result_0;
					}
					{
						result_0 = 1.0 * x * y;
						val.col(2) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						result_0 = -1.0 * y * z;
						val.col(0) = result_0;
					}
					{
						result_0 = -1.0 * z * (x - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -1.0 * y * (x - 1);
						val.col(2) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void q_1_nodes_3d(Eigen::MatrixXd &res)
			{
				res.resize(8, 3);
				res << 0, 0, 0,
					1, 0, 0,
					1, 1, 0,
					0, 1, 0,
					0, 0, 1,
					1, 0, 1,
					1, 1, 1,
					0, 1, 1;
			}

			void q_2_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = 1.0 * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
				}
				break;
				case 1:
				{
					result_0 = 1.0 * x * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
				}
				break;
				case 2:
				{
					result_0 = 1.0 * x * y * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
				}
				break;
				case 3:
				{
					result_0 = 1.0 * y * (x - 1) * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
				}
				break;
				case 4:
				{
					result_0 = 1.0 * z * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
				}
				break;
				case 5:
				{
					result_0 = 1.0 * x * z * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
				}
				break;
				case 6:
				{
					result_0 = 1.0 * x * y * z * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
				}
				break;
				case 7:
				{
					result_0 = 1.0 * y * z * (x - 1) * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
				}
				break;
				case 8:
				{
					result_0 = -4.0 * x * (x - 1) * (y - 1) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
				}
				break;
				case 9:
				{
					result_0 = -4.0 * x * y * (2.0 * x - 1.0) * (y - 1) * (z - 1) * (2.0 * z - 1.0);
				}
				break;
				case 10:
				{
					result_0 = -4.0 * x * y * (x - 1) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
				}
				break;
				case 11:
				{
					result_0 = -4.0 * y * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (z - 1) * (2.0 * z - 1.0);
				}
				break;
				case 12:
				{
					result_0 = -4.0 * z * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (z - 1);
				}
				break;
				case 13:
				{
					result_0 = -4.0 * x * z * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (z - 1);
				}
				break;
				case 14:
				{
					result_0 = -4.0 * x * y * z * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (z - 1);
				}
				break;
				case 15:
				{
					result_0 = -4.0 * y * z * (x - 1) * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (z - 1);
				}
				break;
				case 16:
				{
					result_0 = -4.0 * x * z * (x - 1) * (y - 1) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
				}
				break;
				case 17:
				{
					result_0 = -4.0 * x * y * z * (2.0 * x - 1.0) * (y - 1) * (2.0 * z - 1.0);
				}
				break;
				case 18:
				{
					result_0 = -4.0 * x * y * z * (x - 1) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
				}
				break;
				case 19:
				{
					result_0 = -4.0 * y * z * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (2.0 * z - 1.0);
				}
				break;
				case 20:
				{
					result_0 = 16.0 * y * z * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (z - 1);
				}
				break;
				case 21:
				{
					result_0 = 16.0 * x * y * z * (2.0 * x - 1.0) * (y - 1) * (z - 1);
				}
				break;
				case 22:
				{
					result_0 = 16.0 * x * z * (x - 1) * (y - 1) * (2.0 * y - 1.0) * (z - 1);
				}
				break;
				case 23:
				{
					result_0 = 16.0 * x * y * z * (x - 1) * (2.0 * y - 1.0) * (z - 1);
				}
				break;
				case 24:
				{
					result_0 = 16.0 * x * y * (x - 1) * (y - 1) * (z - 1) * (2.0 * z - 1.0);
				}
				break;
				case 25:
				{
					result_0 = 16.0 * x * y * z * (x - 1) * (y - 1) * (2.0 * z - 1.0);
				}
				break;
				case 26:
				{
					result_0 = -64.0 * x * y * z * (x - 1) * (y - 1) * (z - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void q_2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				val.resize(uv.rows(), uv.cols());
				Eigen::ArrayXd result_0(uv.rows());
				switch (local_index)
				{
				case 0:
				{
					{
						result_0 = (4.0 * x - 3.0) * (y - 1) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = (x - 1) * (2.0 * x - 1.0) * (4.0 * y - 3.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = (x - 1) * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (4.0 * z - 3.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = (4.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = x * (2.0 * x - 1.0) * (4.0 * y - 3.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = x * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (4.0 * z - 3.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						result_0 = y * (4.0 * x - 1.0) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = x * (2.0 * x - 1.0) * (4.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = x * y * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (4.0 * z - 3.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						result_0 = y * (4.0 * x - 3.0) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = (x - 1) * (2.0 * x - 1.0) * (4.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = y * (x - 1) * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (4.0 * z - 3.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						result_0 = z * (4.0 * x - 3.0) * (y - 1) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = z * (x - 1) * (2.0 * x - 1.0) * (4.0 * y - 3.0) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = (x - 1) * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (4.0 * z - 1.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						result_0 = z * (4.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = x * z * (2.0 * x - 1.0) * (4.0 * y - 3.0) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = x * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (4.0 * z - 1.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						result_0 = y * z * (4.0 * x - 1.0) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = x * z * (2.0 * x - 1.0) * (4.0 * y - 1.0) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = x * y * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (4.0 * z - 1.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						result_0 = y * z * (4.0 * x - 3.0) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = z * (x - 1) * (2.0 * x - 1.0) * (4.0 * y - 1.0) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = y * (x - 1) * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (4.0 * z - 1.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 8:
				{
					{
						result_0 = -4.0 * (2 * x - 1) * (y - 1) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * (x - 1) * (16.0 * y - 12.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -x * (x - 1) * (y - 1) * (2.0 * y - 1.0) * (16.0 * z - 12.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 9:
				{
					{
						result_0 = -y * (16.0 * x - 4.0) * (y - 1) * (z - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -4.0 * x * (2.0 * x - 1.0) * (2 * y - 1) * (z - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -x * y * (2.0 * x - 1.0) * (y - 1) * (16.0 * z - 12.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 10:
				{
					{
						result_0 = -4.0 * y * (2 * x - 1) * (2.0 * y - 1.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * (x - 1) * (16.0 * y - 4.0) * (z - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -x * y * (x - 1) * (2.0 * y - 1.0) * (16.0 * z - 12.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 11:
				{
					{
						result_0 = -y * (16.0 * x - 12.0) * (y - 1) * (z - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -4.0 * (x - 1) * (2.0 * x - 1.0) * (2 * y - 1) * (z - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -y * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (16.0 * z - 12.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 12:
				{
					{
						result_0 = -z * (16.0 * x - 12.0) * (y - 1) * (2.0 * y - 1.0) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -z * (x - 1) * (2.0 * x - 1.0) * (16.0 * y - 12.0) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4.0 * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 13:
				{
					{
						result_0 = -z * (16.0 * x - 4.0) * (y - 1) * (2.0 * y - 1.0) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * z * (2.0 * x - 1.0) * (16.0 * y - 12.0) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4.0 * x * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 14:
				{
					{
						result_0 = -y * z * (16.0 * x - 4.0) * (2.0 * y - 1.0) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * z * (2.0 * x - 1.0) * (16.0 * y - 4.0) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4.0 * x * y * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 15:
				{
					{
						result_0 = -y * z * (16.0 * x - 12.0) * (2.0 * y - 1.0) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -z * (x - 1) * (2.0 * x - 1.0) * (16.0 * y - 4.0) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4.0 * y * (x - 1) * (2.0 * x - 1.0) * (2.0 * y - 1.0) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 16:
				{
					{
						result_0 = -4.0 * z * (2 * x - 1) * (y - 1) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * z * (x - 1) * (16.0 * y - 12.0) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -x * (x - 1) * (y - 1) * (2.0 * y - 1.0) * (16.0 * z - 4.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 17:
				{
					{
						result_0 = -y * z * (16.0 * x - 4.0) * (y - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -4.0 * x * z * (2.0 * x - 1.0) * (2 * y - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -x * y * (2.0 * x - 1.0) * (y - 1) * (16.0 * z - 4.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 18:
				{
					{
						result_0 = -4.0 * y * z * (2 * x - 1) * (2.0 * y - 1.0) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * z * (x - 1) * (16.0 * y - 4.0) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -x * y * (x - 1) * (2.0 * y - 1.0) * (16.0 * z - 4.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 19:
				{
					{
						result_0 = -y * z * (16.0 * x - 12.0) * (y - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -4.0 * z * (x - 1) * (2.0 * x - 1.0) * (2 * y - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -y * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (16.0 * z - 4.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 20:
				{
					{
						result_0 = y * z * (64.0 * x - 48.0) * (y - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 16.0 * z * (x - 1) * (2.0 * x - 1.0) * (2 * y - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 16.0 * y * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 21:
				{
					{
						result_0 = y * z * (64.0 * x - 16.0) * (y - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 16.0 * x * z * (2.0 * x - 1.0) * (2 * y - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 16.0 * x * y * (2.0 * x - 1.0) * (y - 1) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 22:
				{
					{
						result_0 = 16.0 * z * (2 * x - 1) * (y - 1) * (2.0 * y - 1.0) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = x * z * (x - 1) * (64.0 * y - 48.0) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 16.0 * x * (x - 1) * (y - 1) * (2.0 * y - 1.0) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 23:
				{
					{
						result_0 = 16.0 * y * z * (2 * x - 1) * (2.0 * y - 1.0) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = x * z * (x - 1) * (64.0 * y - 16.0) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 16.0 * x * y * (x - 1) * (2.0 * y - 1.0) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 24:
				{
					{
						result_0 = 16.0 * y * (2 * x - 1) * (y - 1) * (z - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = 16.0 * x * (x - 1) * (2 * y - 1) * (z - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = x * y * (x - 1) * (y - 1) * (64.0 * z - 48.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 25:
				{
					{
						result_0 = 16.0 * y * z * (2 * x - 1) * (y - 1) * (2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = 16.0 * x * z * (x - 1) * (2 * y - 1) * (2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = x * y * (x - 1) * (y - 1) * (64.0 * z - 16.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 26:
				{
					{
						result_0 = -64.0 * y * z * (2 * x - 1) * (y - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -64.0 * x * z * (x - 1) * (2 * y - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -64.0 * x * y * (x - 1) * (y - 1) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void q_2_nodes_3d(Eigen::MatrixXd &res)
			{
				res.resize(27, 3);
				res << 0, 0, 0,
					1, 0, 0,
					1, 1, 0,
					0, 1, 0,
					0, 0, 1,
					1, 0, 1,
					1, 1, 1,
					0, 1, 1,
					1.0 / 2.0, 0, 0,
					1, 1.0 / 2.0, 0,
					1.0 / 2.0, 1, 0,
					0, 1.0 / 2.0, 0,
					0, 0, 1.0 / 2.0,
					1, 0, 1.0 / 2.0,
					1, 1, 1.0 / 2.0,
					0, 1, 1.0 / 2.0,
					1.0 / 2.0, 0, 1,
					1, 1.0 / 2.0, 1,
					1.0 / 2.0, 1, 1,
					0, 1.0 / 2.0, 1,
					0, 1.0 / 2.0, 1.0 / 2.0,
					1, 1.0 / 2.0, 1.0 / 2.0,
					1.0 / 2.0, 0, 1.0 / 2.0,
					1.0 / 2.0, 1, 1.0 / 2.0,
					1.0 / 2.0, 1.0 / 2.0, 0,
					1.0 / 2.0, 1.0 / 2.0, 1,
					1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0;
			}

			void q_3_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = -1.0 * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 1:
				{
					result_0 = 1.0 * x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 2:
				{
					result_0 = -1.0 * x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 3:
				{
					result_0 = 1.0 * y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 4:
				{
					result_0 = 1.0 * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 5:
				{
					result_0 = -1.0 * x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 6:
				{
					result_0 = 1.0 * x * y * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 7:
				{
					result_0 = -1.0 * y * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 8:
				{
					result_0 = 4.4999999999999991 * x * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 9:
				{
					result_0 = -4.4999999999999991 * x * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 10:
				{
					result_0 = -4.4999999999999991 * x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 11:
				{
					result_0 = 4.4999999999999991 * x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 12:
				{
					result_0 = 4.4999999999999991 * x * y * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 13:
				{
					result_0 = -4.4999999999999991 * x * y * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 14:
				{
					result_0 = -4.4999999999999991 * y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 15:
				{
					result_0 = 4.4999999999999991 * y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 16:
				{
					result_0 = 4.4999999999999991 * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 17:
				{
					result_0 = -4.4999999999999991 * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 18:
				{
					result_0 = 4.4999999999999991 * x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 19:
				{
					result_0 = -4.4999999999999991 * x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 20:
				{
					result_0 = -4.4999999999999991 * x * y * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 21:
				{
					result_0 = 4.4999999999999991 * x * y * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 22:
				{
					result_0 = 4.4999999999999991 * y * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 23:
				{
					result_0 = -4.4999999999999991 * y * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 24:
				{
					result_0 = -4.4999999999999991 * x * z * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 25:
				{
					result_0 = 4.4999999999999991 * x * z * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 26:
				{
					result_0 = 4.4999999999999991 * x * y * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 27:
				{
					result_0 = -4.4999999999999991 * x * y * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 28:
				{
					result_0 = -4.4999999999999991 * x * y * z * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 29:
				{
					result_0 = 4.4999999999999991 * x * y * z * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 30:
				{
					result_0 = 4.4999999999999991 * y * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 31:
				{
					result_0 = -4.4999999999999991 * y * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 32:
				{
					result_0 = -20.249999999999993 * y * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 33:
				{
					result_0 = 20.249999999999993 * y * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 34:
				{
					result_0 = 20.249999999999993 * y * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 35:
				{
					result_0 = -20.249999999999993 * y * z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 36:
				{
					result_0 = 20.249999999999993 * x * y * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 37:
				{
					result_0 = -20.249999999999993 * x * y * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 38:
				{
					result_0 = -20.249999999999993 * x * y * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 39:
				{
					result_0 = 20.249999999999993 * x * y * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 40:
				{
					result_0 = -20.249999999999993 * x * z * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 41:
				{
					result_0 = 20.249999999999993 * x * z * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 42:
				{
					result_0 = 20.249999999999993 * x * z * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 43:
				{
					result_0 = -20.249999999999993 * x * z * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 44:
				{
					result_0 = 20.249999999999993 * x * y * z * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 45:
				{
					result_0 = -20.249999999999993 * x * y * z * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 46:
				{
					result_0 = -20.249999999999993 * x * y * z * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 47:
				{
					result_0 = 20.249999999999993 * x * y * z * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 48:
				{
					result_0 = -20.249999999999993 * x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 49:
				{
					result_0 = 20.249999999999993 * x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 50:
				{
					result_0 = 20.249999999999993 * x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 51:
				{
					result_0 = -20.249999999999993 * x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0);
				}
				break;
				case 52:
				{
					result_0 = 20.249999999999993 * x * y * z * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 53:
				{
					result_0 = -20.249999999999993 * x * y * z * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 54:
				{
					result_0 = -20.249999999999993 * x * y * z * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 55:
				{
					result_0 = 20.249999999999993 * x * y * z * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996);
				}
				break;
				case 56:
				{
					result_0 = 91.124999999999957 * x * y * z * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 57:
				{
					result_0 = -91.124999999999957 * x * y * z * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 58:
				{
					result_0 = -91.124999999999957 * x * y * z * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 59:
				{
					result_0 = 91.124999999999957 * x * y * z * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 60:
				{
					result_0 = -91.124999999999957 * x * y * z * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 61:
				{
					result_0 = 91.124999999999957 * x * y * z * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				case 62:
				{
					result_0 = 91.124999999999957 * x * y * z * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0);
				}
				break;
				case 63:
				{
					result_0 = -91.124999999999957 * x * y * z * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0);
				}
				break;
				default:
					assert(false);
				}
			}
			void q_3_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				val.resize(uv.rows(), uv.cols());
				Eigen::ArrayXd result_0(uv.rows());
				switch (local_index)
				{
				case 0:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = -(y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = -(x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = -(x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * x;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * x;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = -y * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * y;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = -x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = -x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = y * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * y;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * z;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(2) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * x;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = -z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = -x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * z;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = -x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(2) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * x;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * y;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * z;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(2) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = -y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (3.0 * helper_0 * helper_1 + 1.5 * helper_0 * helper_2 + 1.0 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * y;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = -z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z;
						const auto helper_1 = helper_0 - 0.49999999999999989;
						const auto helper_2 = 2.9999999999999996 * z;
						const auto helper_3 = helper_2 - 1.9999999999999996;
						result_0 = -y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_3 + helper_1 * helper_2 + 1.0 * helper_1 * helper_3);
						val.col(2) = result_0;
					}
				}
				break;
				case 8:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 8.9999999999999982;
						result_0 = (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = x * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = x * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 9:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 4.4999999999999991;
						result_0 = -(y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = -x * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = -x * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 10:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = -y * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * x + 6.7499999999999973 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 8.9999999999999982;
						result_0 = -x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = -x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 2.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 11:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = y * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * x + 6.7499999999999973 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 4.4999999999999991;
						result_0 = x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 12:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 4.4999999999999991;
						result_0 = y * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = x * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * y + 6.7499999999999973 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = x * y * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 13:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 8.9999999999999982;
						result_0 = -y * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = -x * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * y + 6.7499999999999973 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = -x * y * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 14:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = -y * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 4.4999999999999991;
						result_0 = -(x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = -y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 15:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = y * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 8.9999999999999982;
						result_0 = (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 16:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 2.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 13.499999999999996 * z - 8.9999999999999982;
						result_0 = (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 17:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = -z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = -z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 13.499999999999996 * z - 4.4999999999999991;
						result_0 = -(x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 18:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * x + 6.7499999999999973 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 13.499999999999996 * z - 4.4999999999999991;
						result_0 = x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 19:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = -z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * x + 6.7499999999999973 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = -x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 13.499999999999996 * z - 8.9999999999999982;
						result_0 = -x * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 20:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = -y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * x + 6.7499999999999973 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = -x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * y + 6.7499999999999973 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 13.499999999999996 * z - 4.4999999999999991;
						result_0 = -x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 21:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * x + 6.7499999999999973 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * y + 6.7499999999999973 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 13.499999999999996 * z - 8.9999999999999982;
						result_0 = x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 22:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * y + 6.7499999999999973 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 13.499999999999996 * z - 4.4999999999999991;
						result_0 = y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 23:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = -y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = -z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 2.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * y + 6.7499999999999973 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 13.499999999999996 * z - 8.9999999999999982;
						result_0 = -y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 24:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 8.9999999999999982;
						result_0 = -z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = -x * z * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = -x * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * z + 6.7499999999999973 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 25:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 4.4999999999999991;
						result_0 = z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = x * z * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = x * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * z + 6.7499999999999973 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 26:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = y * z * (y - 1) * (3.0 * y - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * x + 6.7499999999999973 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 8.9999999999999982;
						result_0 = x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 2.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * z + 6.7499999999999973 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 27:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = -y * z * (y - 1) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * x + 6.7499999999999973 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 4.4999999999999991;
						result_0 = -x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = -x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * z + 6.7499999999999973 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 28:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 4.4999999999999991;
						result_0 = -y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = -x * z * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * y + 6.7499999999999973 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = -x * y * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * z + 6.7499999999999973 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 29:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 13.499999999999996 * x - 8.9999999999999982;
						result_0 = y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = x * z * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * y + 6.7499999999999973 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = x * y * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * z + 6.7499999999999973 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 30:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = y * z * (y - 1) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 4.4999999999999991;
						result_0 = z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * z + 6.7499999999999973 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 31:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = -y * z * (y - 1) * (3.0 * y - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (13.499999999999998 * helper_0 * helper_1 + 6.7499999999999991 * helper_0 * helper_2 + 4.4999999999999991 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 13.499999999999996 * y - 8.9999999999999982;
						result_0 = -z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 13.499999999999998 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = -y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (4.4999999999999991 * helper_0 * helper_1 + 13.499999999999995 * helper_0 * z + 6.7499999999999973 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 32:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = -y * z * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 20.249999999999993;
						result_0 = -z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 20.249999999999993;
						result_0 = -y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 33:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = y * z * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 20.249999999999993;
						result_0 = z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 40.499999999999986;
						result_0 = y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 34:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = y * z * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 1.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 40.499999999999986;
						result_0 = z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 20.249999999999993;
						result_0 = y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 35:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 1.5 * x - 1.0;
						const auto helper_2 = 3.0 * x - 1.0;
						result_0 = -y * z * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 2.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 40.499999999999986;
						result_0 = -z * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 40.499999999999986;
						result_0 = -y * (x - 1) * (1.5 * x - 1.0) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 36:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = y * z * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 2.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * x + 30.374999999999986 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 40.499999999999986;
						result_0 = x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 40.499999999999986;
						result_0 = x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 37:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = -y * z * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 1.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * x + 30.374999999999986 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 40.499999999999986;
						result_0 = -x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 20.249999999999993;
						result_0 = -x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 38:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = -y * z * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * x + 30.374999999999986 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 20.249999999999993;
						result_0 = -x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 40.499999999999986;
						result_0 = -x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 39:
				{
					{
						const auto helper_0 = 1.4999999999999998 * x - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * x - 1.9999999999999996;
						result_0 = y * z * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * x + 30.374999999999986 * helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 20.249999999999993;
						result_0 = x * z * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 20.249999999999993;
						result_0 = x * y * (1.4999999999999998 * x - 0.49999999999999989) * (2.9999999999999996 * x - 1.9999999999999996) * (y - 1) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 40:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 40.499999999999986;
						result_0 = -z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = -x * z * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (3.0 * z - 2.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 40.499999999999986;
						result_0 = -x * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 41:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 40.499999999999986;
						result_0 = z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = x * z * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (3.0 * z - 1.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 20.249999999999993;
						result_0 = x * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 42:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 20.249999999999993;
						result_0 = z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = x * z * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 2.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 40.499999999999986;
						result_0 = x * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 43:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 20.249999999999993;
						result_0 = -z * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 1.5 * y - 1.0;
						const auto helper_2 = 3.0 * y - 1.0;
						result_0 = -x * z * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 1.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 20.249999999999993;
						result_0 = -x * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (1.5 * y - 1.0) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 44:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 40.499999999999986;
						result_0 = y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = x * z * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (3.0 * z - 2.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * y + 30.374999999999986 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 40.499999999999986;
						result_0 = x * y * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 45:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 40.499999999999986;
						result_0 = -y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = -x * z * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (3.0 * z - 1.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * y + 30.374999999999986 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 20.249999999999993;
						result_0 = -x * y * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 46:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 20.249999999999993;
						result_0 = -y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = -x * z * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 2.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * y + 30.374999999999986 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 40.499999999999986;
						result_0 = -x * y * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 47:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 20.249999999999993;
						result_0 = y * z * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * y - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * y - 1.9999999999999996;
						result_0 = x * z * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 1.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * y + 30.374999999999986 * helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 60.749999999999979 * z - 20.249999999999993;
						result_0 = x * y * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * y - 0.49999999999999989) * (2.9999999999999996 * y - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 48:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 40.499999999999986;
						result_0 = -y * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 40.499999999999986;
						result_0 = -x * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = -x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 2.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 49:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 40.499999999999986;
						result_0 = y * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 20.249999999999993;
						result_0 = x * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 1.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 50:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 20.249999999999993;
						result_0 = y * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 40.499999999999986;
						result_0 = x * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 51:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 20.249999999999993;
						result_0 = -y * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 20.249999999999993;
						result_0 = -x * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (1.5 * z - 1.0) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 1.5 * z - 1.0;
						const auto helper_2 = 3.0 * z - 1.0;
						result_0 = -x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (60.749999999999979 * helper_0 * helper_1 + 30.374999999999989 * helper_0 * helper_2 + 20.249999999999993 * helper_1 * helper_2);
						val.col(2) = result_0;
					}
				}
				break;
				case 52:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 40.499999999999986;
						result_0 = y * z * (y - 1) * (3.0 * y - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 40.499999999999986;
						result_0 = x * z * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 2.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * z + 30.374999999999986 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 53:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 40.499999999999986;
						result_0 = -y * z * (y - 1) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 20.249999999999993;
						result_0 = -x * z * (x - 1) * (3.0 * x - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = -x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 1.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * z + 30.374999999999986 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 54:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 20.249999999999993;
						result_0 = -y * z * (y - 1) * (3.0 * y - 2.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 40.499999999999986;
						result_0 = -x * z * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = -x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * z + 30.374999999999986 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 55:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 60.749999999999979 * x - 20.249999999999993;
						result_0 = y * z * (y - 1) * (3.0 * y - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 60.749999999999979 * y - 20.249999999999993;
						result_0 = x * z * (x - 1) * (3.0 * x - 1.0) * (1.4999999999999998 * z - 0.49999999999999989) * (2.9999999999999996 * z - 1.9999999999999996) * (helper_0 * helper_1 + 60.749999999999979 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 1.4999999999999998 * z - 0.49999999999999989;
						const auto helper_1 = 2.9999999999999996 * z - 1.9999999999999996;
						result_0 = x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (20.249999999999993 * helper_0 * helper_1 + 60.749999999999972 * helper_0 * z + 30.374999999999986 * helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 56:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 273.37499999999989 * x - 182.24999999999991;
						result_0 = y * z * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 273.37499999999989 * y - 182.24999999999991;
						result_0 = x * z * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 273.37499999999989 * z - 182.24999999999991;
						result_0 = x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 57:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 273.37499999999989 * x - 182.24999999999991;
						result_0 = -y * z * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 273.37499999999989 * y - 182.24999999999991;
						result_0 = -x * z * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 273.37499999999989 * z - 91.124999999999957;
						result_0 = -x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 58:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 273.37499999999989 * x - 182.24999999999991;
						result_0 = -y * z * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 273.37499999999989 * y - 91.124999999999957;
						result_0 = -x * z * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 273.37499999999989 * z - 182.24999999999991;
						result_0 = -x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 59:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 273.37499999999989 * x - 182.24999999999991;
						result_0 = y * z * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 273.37499999999989 * y - 91.124999999999957;
						result_0 = x * z * (x - 1) * (3.0 * x - 2.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 273.37499999999989 * z - 91.124999999999957;
						result_0 = x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 60:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 273.37499999999989 * x - 91.124999999999957;
						result_0 = -y * z * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 273.37499999999989 * y - 182.24999999999991;
						result_0 = -x * z * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 273.37499999999989 * z - 182.24999999999991;
						result_0 = -x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 61:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 273.37499999999989 * x - 91.124999999999957;
						result_0 = y * z * (y - 1) * (3.0 * y - 2.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 273.37499999999989 * y - 182.24999999999991;
						result_0 = x * z * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 273.37499999999989 * z - 91.124999999999957;
						result_0 = x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 62:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 273.37499999999989 * x - 91.124999999999957;
						result_0 = y * z * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 273.37499999999989 * y - 91.124999999999957;
						result_0 = x * z * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 2.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 273.37499999999989 * z - 182.24999999999991;
						result_0 = x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				case 63:
				{
					{
						const auto helper_0 = x - 1;
						const auto helper_1 = 273.37499999999989 * x - 91.124999999999957;
						result_0 = -y * z * (y - 1) * (3.0 * y - 1.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * x + helper_1 * x);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = y - 1;
						const auto helper_1 = 273.37499999999989 * y - 91.124999999999957;
						result_0 = -x * z * (x - 1) * (3.0 * x - 1.0) * (z - 1) * (3.0 * z - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * y + helper_1 * y);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = z - 1;
						const auto helper_1 = 273.37499999999989 * z - 91.124999999999957;
						result_0 = -x * y * (x - 1) * (3.0 * x - 1.0) * (y - 1) * (3.0 * y - 1.0) * (helper_0 * helper_1 + 273.37499999999989 * helper_0 * z + helper_1 * z);
						val.col(2) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void q_3_nodes_3d(Eigen::MatrixXd &res)
			{
				res.resize(64, 3);
				res << 0, 0, 0,
					1, 0, 0,
					1, 1, 0,
					0, 1, 0,
					0, 0, 1,
					1, 0, 1,
					1, 1, 1,
					0, 1, 1,
					1.0 / 3.0, 0, 0,
					2.0 / 3.0, 0, 0,
					1, 1.0 / 3.0, 0,
					1, 2.0 / 3.0, 0,
					2.0 / 3.0, 1, 0,
					1.0 / 3.0, 1, 0,
					0, 2.0 / 3.0, 0,
					0, 1.0 / 3.0, 0,
					0, 0, 1.0 / 3.0,
					0, 0, 2.0 / 3.0,
					1, 0, 2.0 / 3.0,
					1, 0, 1.0 / 3.0,
					1, 1, 2.0 / 3.0,
					1, 1, 1.0 / 3.0,
					0, 1, 2.0 / 3.0,
					0, 1, 1.0 / 3.0,
					1.0 / 3.0, 0, 1,
					2.0 / 3.0, 0, 1,
					1, 1.0 / 3.0, 1,
					1, 2.0 / 3.0, 1,
					2.0 / 3.0, 1, 1,
					1.0 / 3.0, 1, 1,
					0, 2.0 / 3.0, 1,
					0, 1.0 / 3.0, 1,
					0, 2.0 / 3.0, 2.0 / 3.0,
					0, 2.0 / 3.0, 1.0 / 3.0,
					0, 1.0 / 3.0, 2.0 / 3.0,
					0, 1.0 / 3.0, 1.0 / 3.0,
					1, 1.0 / 3.0, 1.0 / 3.0,
					1, 1.0 / 3.0, 2.0 / 3.0,
					1, 2.0 / 3.0, 1.0 / 3.0,
					1, 2.0 / 3.0, 2.0 / 3.0,
					1.0 / 3.0, 0, 1.0 / 3.0,
					1.0 / 3.0, 0, 2.0 / 3.0,
					2.0 / 3.0, 0, 1.0 / 3.0,
					2.0 / 3.0, 0, 2.0 / 3.0,
					1.0 / 3.0, 1, 1.0 / 3.0,
					1.0 / 3.0, 1, 2.0 / 3.0,
					2.0 / 3.0, 1, 1.0 / 3.0,
					2.0 / 3.0, 1, 2.0 / 3.0,
					1.0 / 3.0, 1.0 / 3.0, 0,
					1.0 / 3.0, 2.0 / 3.0, 0,
					2.0 / 3.0, 1.0 / 3.0, 0,
					2.0 / 3.0, 2.0 / 3.0, 0,
					1.0 / 3.0, 1.0 / 3.0, 1,
					1.0 / 3.0, 2.0 / 3.0, 1,
					2.0 / 3.0, 1.0 / 3.0, 1,
					2.0 / 3.0, 2.0 / 3.0, 1,
					1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
					1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0,
					1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0,
					1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0,
					2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
					2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0,
					2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0,
					2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0;
			}

			void q_m2_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = 1.0 * (x - 1) * (y - 1) * (z - 1) * (2 * x + 2 * y + 2 * z - 1);
				}
				break;
				case 1:
				{
					result_0 = -1.0 * x * (y - 1) * (z - 1) * (-2 * x + 2 * y + 2 * z + 1);
				}
				break;
				case 2:
				{
					result_0 = -1.0 * x * y * (z - 1) * (2 * x + 2 * y - 2 * z - 3);
				}
				break;
				case 3:
				{
					result_0 = -1.0 * y * (x - 1) * (z - 1) * (2 * x - 2 * y + 2 * z + 1);
				}
				break;
				case 4:
				{
					result_0 = -1.0 * z * (x - 1) * (y - 1) * (2 * x + 2 * y - 2 * z + 1);
				}
				break;
				case 5:
				{
					result_0 = -1.0 * x * z * (y - 1) * (2 * x - 2 * y + 2 * z - 3);
				}
				break;
				case 6:
				{
					result_0 = x * y * z * (2.0 * x + 2.0 * y + 2.0 * z - 5.0);
				}
				break;
				case 7:
				{
					result_0 = 1.0 * y * z * (x - 1) * (2 * x - 2 * y - 2 * z + 3);
				}
				break;
				case 8:
				{
					result_0 = -4 * x * (x - 1) * (y - 1) * (z - 1);
				}
				break;
				case 9:
				{
					result_0 = 4 * x * y * (y - 1) * (z - 1);
				}
				break;
				case 10:
				{
					result_0 = 4 * x * y * (x - 1) * (z - 1);
				}
				break;
				case 11:
				{
					result_0 = -4 * y * (x - 1) * (y - 1) * (z - 1);
				}
				break;
				case 12:
				{
					result_0 = -4 * z * (x - 1) * (y - 1) * (z - 1);
				}
				break;
				case 13:
				{
					result_0 = 4 * x * z * (y - 1) * (z - 1);
				}
				break;
				case 14:
				{
					result_0 = -4 * x * y * z * (z - 1);
				}
				break;
				case 15:
				{
					result_0 = 4 * y * z * (x - 1) * (z - 1);
				}
				break;
				case 16:
				{
					result_0 = 4 * x * z * (x - 1) * (y - 1);
				}
				break;
				case 17:
				{
					result_0 = -4 * x * y * z * (y - 1);
				}
				break;
				case 18:
				{
					result_0 = -4 * x * y * z * (x - 1);
				}
				break;
				case 19:
				{
					result_0 = 4 * y * z * (x - 1) * (y - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void q_m2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				val.resize(uv.rows(), uv.cols());
				Eigen::ArrayXd result_0(uv.rows());
				switch (local_index)
				{
				case 0:
				{
					{
						result_0 = (y - 1) * (z - 1) * (4.0 * x + 2 * y + 2 * z - 3.0);
						val.col(0) = result_0;
					}
					{
						result_0 = (x - 1) * (z - 1) * (2.0 * x + 4.0 * y + 2.0 * z - 3.0);
						val.col(1) = result_0;
					}
					{
						result_0 = (x - 1) * (y - 1) * (2.0 * x + 2.0 * y + 4.0 * z - 3.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = -(y - 1) * (z - 1) * (-4.0 * x + 2.0 * y + 2.0 * z + 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = x * (z - 1) * (2.0 * x - 4.0 * y - 2.0 * z + 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = x * (y - 1) * (2.0 * x - 2.0 * y - 4.0 * z + 1.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						result_0 = -y * (z - 1) * (4.0 * x + 2.0 * y - 2.0 * z - 3.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * (z - 1) * (2.0 * x + 4.0 * y - 2.0 * z - 3.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -x * y * (2.0 * x + 2.0 * y - 4.0 * z - 1.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						result_0 = -y * (z - 1) * (4.0 * x - 2.0 * y + 2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -(x - 1) * (z - 1) * (2.0 * x - 4.0 * y + 2.0 * z + 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -y * (x - 1) * (2.0 * x - 2.0 * y + 4.0 * z - 1.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						result_0 = -z * (y - 1) * (4.0 * x + 2.0 * y - 2.0 * z - 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -z * (x - 1) * (2.0 * x + 4.0 * y - 2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -(x - 1) * (y - 1) * (2.0 * x + 2.0 * y - 4.0 * z + 1.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						result_0 = -z * (y - 1) * (4.0 * x - 2.0 * y + 2.0 * z - 3.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -x * z * (2.0 * x - 4.0 * y + 2.0 * z - 1.0);
						val.col(1) = result_0;
					}
					{
						result_0 = -x * (y - 1) * (2.0 * x - 2.0 * y + 4.0 * z - 3.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						result_0 = y * z * (4.0 * x + 2.0 * y + 2.0 * z - 5.0);
						val.col(0) = result_0;
					}
					{
						result_0 = x * z * (2.0 * x + 4.0 * y + 2.0 * z - 5.0);
						val.col(1) = result_0;
					}
					{
						result_0 = x * y * (2.0 * x + 2.0 * y + 4.0 * z - 5.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						result_0 = y * z * (4.0 * x - 2.0 * y - 2.0 * z + 1.0);
						val.col(0) = result_0;
					}
					{
						result_0 = z * (x - 1) * (2.0 * x - 4.0 * y - 2.0 * z + 3.0);
						val.col(1) = result_0;
					}
					{
						result_0 = y * (x - 1) * (2.0 * x - 2.0 * y - 4.0 * z + 3.0);
						val.col(2) = result_0;
					}
				}
				break;
				case 8:
				{
					{
						result_0 = -4 * (2 * x - 1) * (y - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * x * (x - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4 * x * (x - 1) * (y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 9:
				{
					{
						result_0 = 4 * y * (y - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * x * (2 * y - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 4 * x * y * (y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 10:
				{
					{
						result_0 = 4 * y * (2 * x - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * x * (x - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 4 * x * y * (x - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 11:
				{
					{
						result_0 = -4 * y * (y - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * (x - 1) * (2 * y - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4 * y * (x - 1) * (y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 12:
				{
					{
						result_0 = -4 * z * (y - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * z * (x - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4 * (x - 1) * (y - 1) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 13:
				{
					{
						result_0 = 4 * z * (y - 1) * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * x * z * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 4 * x * (y - 1) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 14:
				{
					{
						result_0 = -4 * y * z * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * x * z * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4 * x * y * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 15:
				{
					{
						result_0 = 4 * y * z * (z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * z * (x - 1) * (z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 4 * y * (x - 1) * (2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 16:
				{
					{
						result_0 = 4 * z * (2 * x - 1) * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * x * z * (x - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 4 * x * (x - 1) * (y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 17:
				{
					{
						result_0 = -4 * y * z * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * x * z * (2 * y - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4 * x * y * (y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 18:
				{
					{
						result_0 = -4 * y * z * (2 * x - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * x * z * (x - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4 * x * y * (x - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 19:
				{
					{
						result_0 = 4 * y * z * (y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * z * (x - 1) * (2 * y - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 4 * y * (x - 1) * (y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void q_m2_nodes_3d(Eigen::MatrixXd &res)
			{
				res.resize(20, 3);
				res << 0.0, 0.0, 0.0,
					1.0, 0.0, 0.0,
					1.0, 1.0, 0.0,
					0.0, 1.0, 0.0,
					0.0, 0.0, 1.0,
					1.0, 0.0, 1.0,
					1.0, 1.0, 1.0,
					0.0, 1.0, 1.0,
					0.5, 0.0, 0.0,
					1.0, 0.5, 0.0,
					0.5, 1.0, 0.0,
					0.0, 0.5, 0.0,
					0.0, 0.0, 0.5,
					1.0, 0.0, 0.5,
					1.0, 1.0, 0.5,
					0.0, 1.0, 0.5,
					0.5, 0.0, 1.0,
					1.0, 0.5, 1.0,
					0.5, 1.0, 1.0,
					0.0, 0.5, 1.0;
			}

		} // namespace

		void q_nodes_3d(const int q, Eigen::MatrixXd &val)
		{
			switch (q)
			{
			case 0:
				q_0_nodes_3d(val);
				break;
			case 1:
				q_1_nodes_3d(val);
				break;
			case 2:
				q_2_nodes_3d(val);
				break;
			case 3:
				q_3_nodes_3d(val);
				break;
			case -2:
				q_m2_nodes_3d(val);
				break;
			default:
				assert(false);
			}
		}
		void q_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			switch (q)
			{
			case 0:
				q_0_basis_value_3d(local_index, uv, val);
				break;
			case 1:
				q_1_basis_value_3d(local_index, uv, val);
				break;
			case 2:
				q_2_basis_value_3d(local_index, uv, val);
				break;
			case 3:
				q_3_basis_value_3d(local_index, uv, val);
				break;
			case -2:
				q_m2_basis_value_3d(local_index, uv, val);
				break;
			default:
				assert(false);
			}
		}

		void q_grad_basis_value_3d(const int q, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			switch (q)
			{
			case 0:
				q_0_basis_grad_value_3d(local_index, uv, val);
				break;
			case 1:
				q_1_basis_grad_value_3d(local_index, uv, val);
				break;
			case 2:
				q_2_basis_grad_value_3d(local_index, uv, val);
				break;
			case 3:
				q_3_basis_grad_value_3d(local_index, uv, val);
				break;
			case -2:
				q_m2_basis_grad_value_3d(local_index, uv, val);
				break;
			default:
				assert(false);
			}
		}

		namespace
		{

		}
	} // namespace autogen
} // namespace polyfem
