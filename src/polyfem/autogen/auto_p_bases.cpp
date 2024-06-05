#include "auto_p_bases.hpp"

namespace polyfem
{
	namespace autogen
	{
		namespace
		{
			void p_0_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
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
			void p_0_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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

			void p_0_nodes_2d(Eigen::MatrixXd &res)
			{
				res.resize(1, 2);
				res << 0.33333333333333331, 0.33333333333333331;
			}

			void p_1_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = -x - y + 1;
				}
				break;
				case 1:
				{
					result_0 = x;
				}
				break;
				case 2:
				{
					result_0 = y;
				}
				break;
				default:
					assert(false);
				}
			}
			void p_1_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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
						result_0.setConstant(-1);
						val.col(0) = result_0;
					}
					{
						result_0.setConstant(-1);
						val.col(1) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0.setOnes();
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0.setOnes();
						val.col(1) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void p_1_nodes_2d(Eigen::MatrixXd &res)
			{
				res.resize(3, 2);
				res << 0, 0,
					1, 0,
					0, 1;
			}

			void p_2_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = (x + y - 1) * (2 * x + 2 * y - 1);
				}
				break;
				case 1:
				{
					result_0 = x * (2 * x - 1);
				}
				break;
				case 2:
				{
					result_0 = y * (2 * y - 1);
				}
				break;
				case 3:
				{
					result_0 = -4 * x * (x + y - 1);
				}
				break;
				case 4:
				{
					result_0 = 4 * x * y;
				}
				break;
				case 5:
				{
					result_0 = -4 * y * (x + y - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void p_2_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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
						result_0 = 4 * x + 4 * y - 3;
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * x + 4 * y - 3;
						val.col(1) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = 4 * x - 1;
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * y - 1;
						val.col(1) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						result_0 = 4 * (-2 * x - y + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * x;
						val.col(1) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						result_0 = 4 * y;
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * x;
						val.col(1) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						result_0 = -4 * y;
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * (-x - 2 * y + 1);
						val.col(1) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void p_2_nodes_2d(Eigen::MatrixXd &res)
			{
				res.resize(6, 2);
				res << 0, 0,
					1, 0,
					0, 1,
					1.0 / 2.0, 0,
					1.0 / 2.0, 1.0 / 2.0,
					0, 1.0 / 2.0;
			}

			void p_3_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				switch (local_index)
				{
				case 0:
				{
					const auto helper_0 = pow(x, 2);
					const auto helper_1 = pow(y, 2);
					result_0 = -27.0 / 2.0 * helper_0 * y + 9 * helper_0 - 27.0 / 2.0 * helper_1 * x + 9 * helper_1 - 9.0 / 2.0 * pow(x, 3) + 18 * x * y - 11.0 / 2.0 * x - 9.0 / 2.0 * pow(y, 3) - 11.0 / 2.0 * y + 1;
				}
				break;
				case 1:
				{
					result_0 = (1.0 / 2.0) * x * (9 * pow(x, 2) - 9 * x + 2);
				}
				break;
				case 2:
				{
					result_0 = (1.0 / 2.0) * y * (9 * pow(y, 2) - 9 * y + 2);
				}
				break;
				case 3:
				{
					result_0 = (9.0 / 2.0) * x * (x + y - 1) * (3 * x + 3 * y - 2);
				}
				break;
				case 4:
				{
					result_0 = -9.0 / 2.0 * x * (3 * pow(x, 2) + 3 * x * y - 4 * x - y + 1);
				}
				break;
				case 5:
				{
					result_0 = (9.0 / 2.0) * x * y * (3 * x - 1);
				}
				break;
				case 6:
				{
					result_0 = (9.0 / 2.0) * x * y * (3 * y - 1);
				}
				break;
				case 7:
				{
					result_0 = -9.0 / 2.0 * y * (3 * x * y - x + 3 * pow(y, 2) - 4 * y + 1);
				}
				break;
				case 8:
				{
					result_0 = (9.0 / 2.0) * y * (x + y - 1) * (3 * x + 3 * y - 2);
				}
				break;
				case 9:
				{
					result_0 = -27 * x * y * (x + y - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void p_3_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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
						result_0 = -27.0 / 2.0 * pow(x, 2) - 27 * x * y + 18 * x - 27.0 / 2.0 * pow(y, 2) + 18 * y - 11.0 / 2.0;
						val.col(0) = result_0;
					}
					{
						result_0 = -27.0 / 2.0 * pow(x, 2) - 27 * x * y + 18 * x - 27.0 / 2.0 * pow(y, 2) + 18 * y - 11.0 / 2.0;
						val.col(1) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = (27.0 / 2.0) * pow(x, 2) - 9 * x + 1;
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = (27.0 / 2.0) * pow(y, 2) - 9 * y + 1;
						val.col(1) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						result_0 = 9 * ((9.0 / 2.0) * pow(x, 2) + 6 * x * y - 5 * x + (3.0 / 2.0) * pow(y, 2) - 5.0 / 2.0 * y + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * x * (6 * x + 6 * y - 5);
						val.col(1) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						result_0 = 9 * (-9.0 / 2.0 * pow(x, 2) - 3 * x * y + 4 * x + (1.0 / 2.0) * y - 1.0 / 2.0);
						val.col(0) = result_0;
					}
					{
						result_0 = -9.0 / 2.0 * x * (3 * x - 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						result_0 = (9.0 / 2.0) * y * (6 * x - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * x * (3 * x - 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						result_0 = (9.0 / 2.0) * y * (3 * y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * x * (6 * y - 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						result_0 = -9.0 / 2.0 * y * (3 * y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 9 * (-3 * x * y + (1.0 / 2.0) * x - 9.0 / 2.0 * pow(y, 2) + 4 * y - 1.0 / 2.0);
						val.col(1) = result_0;
					}
				}
				break;
				case 8:
				{
					{
						result_0 = (9.0 / 2.0) * y * (6 * x + 6 * y - 5);
						val.col(0) = result_0;
					}
					{
						result_0 = 9 * ((3.0 / 2.0) * pow(x, 2) + 6 * x * y - 5.0 / 2.0 * x + (9.0 / 2.0) * pow(y, 2) - 5 * y + 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 9:
				{
					{
						result_0 = -27 * y * (2 * x + y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -27 * x * (x + 2 * y - 1);
						val.col(1) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void p_3_nodes_2d(Eigen::MatrixXd &res)
			{
				res.resize(10, 2);
				res << 0, 0,
					1, 0,
					0, 1,
					1.0 / 3.0, 0,
					2.0 / 3.0, 0,
					2.0 / 3.0, 1.0 / 3.0,
					1.0 / 3.0, 2.0 / 3.0,
					0, 2.0 / 3.0,
					0, 1.0 / 3.0,
					1.0 / 3.0, 1.0 / 3.0;
			}

			void p_4_basis_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();

				switch (local_index)
				{
				case 0:
				{
					const auto helper_0 = pow(x, 2);
					const auto helper_1 = pow(x, 3);
					const auto helper_2 = pow(y, 2);
					const auto helper_3 = pow(y, 3);
					result_0 = 64 * helper_0 * helper_2 - 80 * helper_0 * y + (70.0 / 3.0) * helper_0 + (128.0 / 3.0) * helper_1 * y - 80.0 / 3.0 * helper_1 - 80 * helper_2 * x + (70.0 / 3.0) * helper_2 + (128.0 / 3.0) * helper_3 * x - 80.0 / 3.0 * helper_3 + (32.0 / 3.0) * pow(x, 4) + (140.0 / 3.0) * x * y - 25.0 / 3.0 * x + (32.0 / 3.0) * pow(y, 4) - 25.0 / 3.0 * y + 1;
				}
				break;
				case 1:
				{
					result_0 = (1.0 / 3.0) * x * (32 * pow(x, 3) - 48 * pow(x, 2) + 22 * x - 3);
				}
				break;
				case 2:
				{
					result_0 = (1.0 / 3.0) * y * (32 * pow(y, 3) - 48 * pow(y, 2) + 22 * y - 3);
				}
				break;
				case 3:
				{
					const auto helper_0 = pow(x, 2);
					const auto helper_1 = pow(y, 2);
					result_0 = -16.0 / 3.0 * x * (24 * helper_0 * y - 18 * helper_0 + 24 * helper_1 * x - 18 * helper_1 + 8 * pow(x, 3) - 36 * x * y + 13 * x + 8 * pow(y, 3) + 13 * y - 3);
				}
				break;
				case 4:
				{
					const auto helper_0 = 32 * pow(x, 2);
					const auto helper_1 = pow(y, 2);
					result_0 = 4 * x * (helper_0 * y - helper_0 + 16 * helper_1 * x - 4 * helper_1 + 16 * pow(x, 3) - 36 * x * y + 19 * x + 7 * y - 3);
				}
				break;
				case 5:
				{
					const auto helper_0 = pow(x, 2);
					result_0 = -16.0 / 3.0 * x * (8 * helper_0 * y - 14 * helper_0 + 8 * pow(x, 3) - 6 * x * y + 7 * x + y - 1);
				}
				break;
				case 6:
				{
					result_0 = (16.0 / 3.0) * x * y * (8 * pow(x, 2) - 6 * x + 1);
				}
				break;
				case 7:
				{
					const auto helper_0 = 4 * x;
					result_0 = helper_0 * y * (-helper_0 + 16 * x * y - 4 * y + 1);
				}
				break;
				case 8:
				{
					result_0 = (16.0 / 3.0) * x * y * (8 * pow(y, 2) - 6 * y + 1);
				}
				break;
				case 9:
				{
					const auto helper_0 = pow(y, 2);
					result_0 = -16.0 / 3.0 * y * (8 * helper_0 * x - 14 * helper_0 - 6 * x * y + x + 8 * pow(y, 3) + 7 * y - 1);
				}
				break;
				case 10:
				{
					const auto helper_0 = pow(x, 2);
					const auto helper_1 = 32 * pow(y, 2);
					result_0 = 4 * y * (16 * helper_0 * y - 4 * helper_0 + helper_1 * x - helper_1 - 36 * x * y + 7 * x + 16 * pow(y, 3) + 19 * y - 3);
				}
				break;
				case 11:
				{
					const auto helper_0 = pow(x, 2);
					const auto helper_1 = pow(y, 2);
					result_0 = -16.0 / 3.0 * y * (24 * helper_0 * y - 18 * helper_0 + 24 * helper_1 * x - 18 * helper_1 + 8 * pow(x, 3) - 36 * x * y + 13 * x + 8 * pow(y, 3) + 13 * y - 3);
				}
				break;
				case 12:
				{
					result_0 = 32 * x * y * (x + y - 1) * (4 * x + 4 * y - 3);
				}
				break;
				case 13:
				{
					result_0 = -32 * x * y * (4 * y - 1) * (x + y - 1);
				}
				break;
				case 14:
				{
					result_0 = -32 * x * y * (4 * x - 1) * (x + y - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void p_4_basis_grad_value_2d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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
						const auto helper_0 = pow(x, 2);
						const auto helper_1 = pow(y, 2);
						result_0 = 128 * helper_0 * y - 80 * helper_0 + 128 * helper_1 * x - 80 * helper_1 + (128.0 / 3.0) * pow(x, 3) - 160 * x * y + (140.0 / 3.0) * x + (128.0 / 3.0) * pow(y, 3) + (140.0 / 3.0) * y - 25.0 / 3.0;
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = pow(x, 2);
						const auto helper_1 = pow(y, 2);
						result_0 = 128 * helper_0 * y - 80 * helper_0 + 128 * helper_1 * x - 80 * helper_1 + (128.0 / 3.0) * pow(x, 3) - 160 * x * y + (140.0 / 3.0) * x + (128.0 / 3.0) * pow(y, 3) + (140.0 / 3.0) * y - 25.0 / 3.0;
						val.col(1) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = (128.0 / 3.0) * pow(x, 3) - 48 * pow(x, 2) + (44.0 / 3.0) * x - 1;
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
				}
				break;
				case 2:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = (128.0 / 3.0) * pow(y, 3) - 48 * pow(y, 2) + (44.0 / 3.0) * y - 1;
						val.col(1) = result_0;
					}
				}
				break;
				case 3:
				{
					{
						const auto helper_0 = pow(x, 2);
						const auto helper_1 = pow(y, 2);
						result_0 = -384 * helper_0 * y + 288 * helper_0 - 256 * helper_1 * x + 96 * helper_1 - 512.0 / 3.0 * pow(x, 3) + 384 * x * y - 416.0 / 3.0 * x - 128.0 / 3.0 * pow(y, 3) - 208.0 / 3.0 * y + 16;
						val.col(0) = result_0;
					}
					{
						result_0 = -16.0 / 3.0 * x * (24 * pow(x, 2) + 48 * x * y - 36 * x + 24 * pow(y, 2) - 36 * y + 13);
						val.col(1) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						const auto helper_0 = 96 * pow(x, 2);
						const auto helper_1 = pow(y, 2);
						result_0 = 4 * helper_0 * y - 4 * helper_0 + 128 * helper_1 * x - 16 * helper_1 + 256 * pow(x, 3) - 288 * x * y + 152 * x + 28 * y - 12;
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * x * (32 * pow(x, 2) + 32 * x * y - 36 * x - 8 * y + 7);
						val.col(1) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						const auto helper_0 = pow(x, 2);
						result_0 = -128 * helper_0 * y + 224 * helper_0 - 512.0 / 3.0 * pow(x, 3) + 64 * x * y - 224.0 / 3.0 * x - 16.0 / 3.0 * y + 16.0 / 3.0;
						val.col(0) = result_0;
					}
					{
						result_0 = -16.0 / 3.0 * x * (8 * pow(x, 2) - 6 * x + 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						result_0 = (16.0 / 3.0) * y * (24 * pow(x, 2) - 12 * x + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = (16.0 / 3.0) * x * (8 * pow(x, 2) - 6 * x + 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						const auto helper_0 = 4 * y;
						result_0 = helper_0 * (-helper_0 + 32 * x * y - 8 * x + 1);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 4 * x;
						result_0 = helper_0 * (-helper_0 + 32 * x * y - 8 * y + 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 8:
				{
					{
						result_0 = (16.0 / 3.0) * y * (8 * pow(y, 2) - 6 * y + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = (16.0 / 3.0) * x * (24 * pow(y, 2) - 12 * y + 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 9:
				{
					{
						result_0 = -16.0 / 3.0 * y * (8 * pow(y, 2) - 6 * y + 1);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = pow(y, 2);
						result_0 = -128 * helper_0 * x + 224 * helper_0 + 64 * x * y - 16.0 / 3.0 * x - 512.0 / 3.0 * pow(y, 3) - 224.0 / 3.0 * y + 16.0 / 3.0;
						val.col(1) = result_0;
					}
				}
				break;
				case 10:
				{
					{
						result_0 = 4 * y * (32 * x * y - 8 * x + 32 * pow(y, 2) - 36 * y + 7);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = pow(x, 2);
						const auto helper_1 = 96 * pow(y, 2);
						result_0 = 128 * helper_0 * y - 16 * helper_0 + 4 * helper_1 * x - 4 * helper_1 - 288 * x * y + 28 * x + 256 * pow(y, 3) + 152 * y - 12;
						val.col(1) = result_0;
					}
				}
				break;
				case 11:
				{
					{
						result_0 = -16.0 / 3.0 * y * (24 * pow(x, 2) + 48 * x * y - 36 * x + 24 * pow(y, 2) - 36 * y + 13);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = pow(x, 2);
						const auto helper_1 = pow(y, 2);
						result_0 = -256 * helper_0 * y + 96 * helper_0 - 384 * helper_1 * x + 288 * helper_1 - 128.0 / 3.0 * pow(x, 3) + 384 * x * y - 208.0 / 3.0 * x - 512.0 / 3.0 * pow(y, 3) - 416.0 / 3.0 * y + 16;
						val.col(1) = result_0;
					}
				}
				break;
				case 12:
				{
					{
						result_0 = 32 * y * (12 * pow(x, 2) + 16 * x * y - 14 * x + 4 * pow(y, 2) - 7 * y + 3);
						val.col(0) = result_0;
					}
					{
						result_0 = 32 * x * (4 * pow(x, 2) + 16 * x * y - 7 * x + 12 * pow(y, 2) - 14 * y + 3);
						val.col(1) = result_0;
					}
				}
				break;
				case 13:
				{
					{
						result_0 = -32 * y * (8 * x * y - 2 * x + 4 * pow(y, 2) - 5 * y + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -32 * x * (8 * x * y - x + 12 * pow(y, 2) - 10 * y + 1);
						val.col(1) = result_0;
					}
				}
				break;
				case 14:
				{
					{
						result_0 = -32 * y * (12 * pow(x, 2) + 8 * x * y - 10 * x - y + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -32 * x * (4 * pow(x, 2) + 8 * x * y - 5 * x - 2 * y + 1);
						val.col(1) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void p_4_nodes_2d(Eigen::MatrixXd &res)
			{
				res.resize(15, 2);
				res << 0, 0,
					1, 0,
					0, 1,
					1.0 / 4.0, 0,
					1.0 / 2.0, 0,
					3.0 / 4.0, 0,
					3.0 / 4.0, 1.0 / 4.0,
					1.0 / 2.0, 1.0 / 2.0,
					1.0 / 4.0, 3.0 / 4.0,
					0, 3.0 / 4.0,
					0, 1.0 / 2.0,
					0, 1.0 / 4.0,
					1.0 / 4.0, 1.0 / 4.0,
					1.0 / 4.0, 1.0 / 2.0,
					1.0 / 2.0, 1.0 / 4.0;
			}

		} // namespace

		void p_nodes_2d(const int p, Eigen::MatrixXd &val)
		{
			switch (p)
			{
			case 0:
				p_0_nodes_2d(val);
				break;
			case 1:
				p_1_nodes_2d(val);
				break;
			case 2:
				p_2_nodes_2d(val);
				break;
			case 3:
				p_3_nodes_2d(val);
				break;
			case 4:
				p_4_nodes_2d(val);
				break;
			default:
				p_n_nodes_2d(p, val);
			}
		}
		void p_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			switch (p)
			{
			case 0:
				p_0_basis_value_2d(local_index, uv, val);
				break;
			case 1:
				p_1_basis_value_2d(local_index, uv, val);
				break;
			case 2:
				p_2_basis_value_2d(local_index, uv, val);
				break;
			case 3:
				p_3_basis_value_2d(local_index, uv, val);
				break;
			case 4:
				p_4_basis_value_2d(local_index, uv, val);
				break;
			default:
				p_n_basis_value_2d(p, local_index, uv, val);
			}
		}

		void p_grad_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			switch (p)
			{
			case 0:
				p_0_basis_grad_value_2d(local_index, uv, val);
				break;
			case 1:
				p_1_basis_grad_value_2d(local_index, uv, val);
				break;
			case 2:
				p_2_basis_grad_value_2d(local_index, uv, val);
				break;
			case 3:
				p_3_basis_grad_value_2d(local_index, uv, val);
				break;
			case 4:
				p_4_basis_grad_value_2d(local_index, uv, val);
				break;
			default:
				p_n_basis_grad_value_2d(p, local_index, uv, val);
			}
		}

		namespace
		{
			void p_0_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
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
			void p_0_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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

			void p_0_nodes_3d(Eigen::MatrixXd &res)
			{
				res.resize(1, 3);
				res << 0.33333333333333331, 0.33333333333333331, 0.33333333333333331;
			}

			void p_1_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = -x - y - z + 1;
				}
				break;
				case 1:
				{
					result_0 = x;
				}
				break;
				case 2:
				{
					result_0 = y;
				}
				break;
				case 3:
				{
					result_0 = z;
				}
				break;
				default:
					assert(false);
				}
			}
			void p_1_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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
						result_0.setConstant(-1);
						val.col(0) = result_0;
					}
					{
						result_0.setConstant(-1);
						val.col(1) = result_0;
					}
					{
						result_0.setConstant(-1);
						val.col(2) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0.setOnes();
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
				case 2:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0.setOnes();
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				case 3:
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
						result_0.setOnes();
						val.col(2) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void p_1_nodes_3d(Eigen::MatrixXd &res)
			{
				res.resize(4, 3);
				res << 0, 0, 0,
					1, 0, 0,
					0, 1, 0,
					0, 0, 1;
			}

			void p_2_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				switch (local_index)
				{
				case 0:
				{
					result_0 = (x + y + z - 1) * (2 * x + 2 * y + 2 * z - 1);
				}
				break;
				case 1:
				{
					result_0 = x * (2 * x - 1);
				}
				break;
				case 2:
				{
					result_0 = y * (2 * y - 1);
				}
				break;
				case 3:
				{
					result_0 = z * (2 * z - 1);
				}
				break;
				case 4:
				{
					result_0 = -4 * x * (x + y + z - 1);
				}
				break;
				case 5:
				{
					result_0 = 4 * x * y;
				}
				break;
				case 6:
				{
					result_0 = -4 * y * (x + y + z - 1);
				}
				break;
				case 7:
				{
					result_0 = -4 * z * (x + y + z - 1);
				}
				break;
				case 8:
				{
					result_0 = 4 * x * z;
				}
				break;
				case 9:
				{
					result_0 = 4 * y * z;
				}
				break;
				default:
					assert(false);
				}
			}
			void p_2_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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
						result_0 = 4 * x + 4 * y + 4 * z - 3;
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * x + 4 * y + 4 * z - 3;
						val.col(1) = result_0;
					}
					{
						result_0 = 4 * x + 4 * y + 4 * z - 3;
						val.col(2) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = 4 * x - 1;
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
				case 2:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * y - 1;
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				case 3:
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
						result_0 = 4 * z - 1;
						val.col(2) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						result_0 = 4 * (-2 * x - y - z + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * x;
						val.col(1) = result_0;
					}
					{
						result_0 = -4 * x;
						val.col(2) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						result_0 = 4 * y;
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * x;
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						result_0 = -4 * y;
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * (-x - 2 * y - z + 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -4 * y;
						val.col(2) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						result_0 = -4 * z;
						val.col(0) = result_0;
					}
					{
						result_0 = -4 * z;
						val.col(1) = result_0;
					}
					{
						result_0 = 4 * (-x - y - 2 * z + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 8:
				{
					{
						result_0 = 4 * z;
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
					{
						result_0 = 4 * x;
						val.col(2) = result_0;
					}
				}
				break;
				case 9:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = 4 * z;
						val.col(1) = result_0;
					}
					{
						result_0 = 4 * y;
						val.col(2) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void p_2_nodes_3d(Eigen::MatrixXd &res)
			{
				res.resize(10, 3);
				res << 0, 0, 0,
					1, 0, 0,
					0, 1, 0,
					0, 0, 1,
					1.0 / 2.0, 0, 0,
					1.0 / 2.0, 1.0 / 2.0, 0,
					0, 1.0 / 2.0, 0,
					0, 0, 1.0 / 2.0,
					1.0 / 2.0, 0, 1.0 / 2.0,
					0, 1.0 / 2.0, 1.0 / 2.0;
			}

			void p_3_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				switch (local_index)
				{
				case 0:
				{
					const auto helper_0 = pow(x, 2);
					const auto helper_1 = pow(y, 2);
					const auto helper_2 = pow(z, 2);
					const auto helper_3 = (27.0 / 2.0) * x;
					const auto helper_4 = (27.0 / 2.0) * y;
					const auto helper_5 = (27.0 / 2.0) * z;
					result_0 = -helper_0 * helper_4 - helper_0 * helper_5 + 9 * helper_0 - helper_1 * helper_3 - helper_1 * helper_5 + 9 * helper_1 - helper_2 * helper_3 - helper_2 * helper_4 + 9 * helper_2 - 9.0 / 2.0 * pow(x, 3) - 27 * x * y * z + 18 * x * y + 18 * x * z - 11.0 / 2.0 * x - 9.0 / 2.0 * pow(y, 3) + 18 * y * z - 11.0 / 2.0 * y - 9.0 / 2.0 * pow(z, 3) - 11.0 / 2.0 * z + 1;
				}
				break;
				case 1:
				{
					result_0 = (1.0 / 2.0) * x * (9 * pow(x, 2) - 9 * x + 2);
				}
				break;
				case 2:
				{
					result_0 = (1.0 / 2.0) * y * (9 * pow(y, 2) - 9 * y + 2);
				}
				break;
				case 3:
				{
					result_0 = (1.0 / 2.0) * z * (9 * pow(z, 2) - 9 * z + 2);
				}
				break;
				case 4:
				{
					result_0 = (9.0 / 2.0) * x * (x + y + z - 1) * (3 * x + 3 * y + 3 * z - 2);
				}
				break;
				case 5:
				{
					const auto helper_0 = 3 * x;
					result_0 = -9.0 / 2.0 * x * (helper_0 * y + helper_0 * z + 3 * pow(x, 2) - 4 * x - y - z + 1);
				}
				break;
				case 6:
				{
					result_0 = (9.0 / 2.0) * x * y * (3 * x - 1);
				}
				break;
				case 7:
				{
					result_0 = (9.0 / 2.0) * x * y * (3 * y - 1);
				}
				break;
				case 8:
				{
					const auto helper_0 = 3 * y;
					result_0 = -9.0 / 2.0 * y * (helper_0 * x + helper_0 * z - x + 3 * pow(y, 2) - 4 * y - z + 1);
				}
				break;
				case 9:
				{
					result_0 = (9.0 / 2.0) * y * (x + y + z - 1) * (3 * x + 3 * y + 3 * z - 2);
				}
				break;
				case 10:
				{
					result_0 = (9.0 / 2.0) * z * (x + y + z - 1) * (3 * x + 3 * y + 3 * z - 2);
				}
				break;
				case 11:
				{
					const auto helper_0 = 3 * z;
					result_0 = -9.0 / 2.0 * z * (helper_0 * x + helper_0 * y - x - y + 3 * pow(z, 2) - 4 * z + 1);
				}
				break;
				case 12:
				{
					result_0 = (9.0 / 2.0) * x * z * (3 * x - 1);
				}
				break;
				case 13:
				{
					result_0 = (9.0 / 2.0) * x * z * (3 * z - 1);
				}
				break;
				case 14:
				{
					result_0 = (9.0 / 2.0) * y * z * (3 * y - 1);
				}
				break;
				case 15:
				{
					result_0 = (9.0 / 2.0) * y * z * (3 * z - 1);
				}
				break;
				case 16:
				{
					result_0 = -27 * x * y * (x + y + z - 1);
				}
				break;
				case 17:
				{
					result_0 = -27 * x * z * (x + y + z - 1);
				}
				break;
				case 18:
				{
					result_0 = 27 * x * y * z;
				}
				break;
				case 19:
				{
					result_0 = -27 * y * z * (x + y + z - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void p_3_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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
						const auto helper_0 = 27 * x;
						result_0 = -helper_0 * y - helper_0 * z - 27.0 / 2.0 * pow(x, 2) + 18 * x - 27.0 / 2.0 * pow(y, 2) - 27 * y * z + 18 * y - 27.0 / 2.0 * pow(z, 2) + 18 * z - 11.0 / 2.0;
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 27 * x;
						result_0 = -helper_0 * y - helper_0 * z - 27.0 / 2.0 * pow(x, 2) + 18 * x - 27.0 / 2.0 * pow(y, 2) - 27 * y * z + 18 * y - 27.0 / 2.0 * pow(z, 2) + 18 * z - 11.0 / 2.0;
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 27 * x;
						result_0 = -helper_0 * y - helper_0 * z - 27.0 / 2.0 * pow(x, 2) + 18 * x - 27.0 / 2.0 * pow(y, 2) - 27 * y * z + 18 * y - 27.0 / 2.0 * pow(z, 2) + 18 * z - 11.0 / 2.0;
						val.col(2) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = (27.0 / 2.0) * pow(x, 2) - 9 * x + 1;
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
				case 2:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = (27.0 / 2.0) * pow(y, 2) - 9 * y + 1;
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				case 3:
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
						result_0 = (27.0 / 2.0) * pow(z, 2) - 9 * z + 1;
						val.col(2) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						const auto helper_0 = 6 * x;
						result_0 = 9 * helper_0 * y + 9 * helper_0 * z + (81.0 / 2.0) * pow(x, 2) - 45 * x + (27.0 / 2.0) * pow(y, 2) + 27 * y * z - 45.0 / 2.0 * y + (27.0 / 2.0) * pow(z, 2) - 45.0 / 2.0 * z + 9;
						val.col(0) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * x * (6 * x + 6 * y + 6 * z - 5);
						val.col(1) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * x * (6 * x + 6 * y + 6 * z - 5);
						val.col(2) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						const auto helper_0 = 3 * x;
						result_0 = -9 * helper_0 * y - 9 * helper_0 * z - 81.0 / 2.0 * pow(x, 2) + 36 * x + (9.0 / 2.0) * y + (9.0 / 2.0) * z - 9.0 / 2.0;
						val.col(0) = result_0;
					}
					{
						result_0 = -9.0 / 2.0 * x * (3 * x - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -9.0 / 2.0 * x * (3 * x - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						result_0 = (9.0 / 2.0) * y * (6 * x - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * x * (3 * x - 1);
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						result_0 = (9.0 / 2.0) * y * (3 * y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * x * (6 * y - 1);
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				case 8:
				{
					{
						result_0 = -9.0 / 2.0 * y * (3 * y - 1);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 3 * y;
						result_0 = -9 * helper_0 * x - 9 * helper_0 * z + (9.0 / 2.0) * x - 81.0 / 2.0 * pow(y, 2) + 36 * y + (9.0 / 2.0) * z - 9.0 / 2.0;
						val.col(1) = result_0;
					}
					{
						result_0 = -9.0 / 2.0 * y * (3 * y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 9:
				{
					{
						result_0 = (9.0 / 2.0) * y * (6 * x + 6 * y + 6 * z - 5);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 6 * y;
						result_0 = 9 * helper_0 * x + 9 * helper_0 * z + (27.0 / 2.0) * pow(x, 2) + 27 * x * z - 45.0 / 2.0 * x + (81.0 / 2.0) * pow(y, 2) - 45 * y + (27.0 / 2.0) * pow(z, 2) - 45.0 / 2.0 * z + 9;
						val.col(1) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * y * (6 * x + 6 * y + 6 * z - 5);
						val.col(2) = result_0;
					}
				}
				break;
				case 10:
				{
					{
						result_0 = (9.0 / 2.0) * z * (6 * x + 6 * y + 6 * z - 5);
						val.col(0) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * z * (6 * x + 6 * y + 6 * z - 5);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 6 * z;
						result_0 = 9 * helper_0 * x + 9 * helper_0 * y + (27.0 / 2.0) * pow(x, 2) + 27 * x * y - 45.0 / 2.0 * x + (27.0 / 2.0) * pow(y, 2) - 45.0 / 2.0 * y + (81.0 / 2.0) * pow(z, 2) - 45 * z + 9;
						val.col(2) = result_0;
					}
				}
				break;
				case 11:
				{
					{
						result_0 = -9.0 / 2.0 * z * (3 * z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -9.0 / 2.0 * z * (3 * z - 1);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 3 * z;
						result_0 = -9 * helper_0 * x - 9 * helper_0 * y + (9.0 / 2.0) * x + (9.0 / 2.0) * y - 81.0 / 2.0 * pow(z, 2) + 36 * z - 9.0 / 2.0;
						val.col(2) = result_0;
					}
				}
				break;
				case 12:
				{
					{
						result_0 = (9.0 / 2.0) * z * (6 * x - 1);
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * x * (3 * x - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 13:
				{
					{
						result_0 = (9.0 / 2.0) * z * (3 * z - 1);
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * x * (6 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 14:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * z * (6 * y - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * y * (3 * y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 15:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * z * (3 * z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = (9.0 / 2.0) * y * (6 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 16:
				{
					{
						result_0 = -27 * y * (2 * x + y + z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -27 * x * (x + 2 * y + z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -27 * x * y;
						val.col(2) = result_0;
					}
				}
				break;
				case 17:
				{
					{
						result_0 = -27 * z * (2 * x + y + z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -27 * x * z;
						val.col(1) = result_0;
					}
					{
						result_0 = -27 * x * (x + y + 2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 18:
				{
					{
						result_0 = 27 * y * z;
						val.col(0) = result_0;
					}
					{
						result_0 = 27 * x * z;
						val.col(1) = result_0;
					}
					{
						result_0 = 27 * x * y;
						val.col(2) = result_0;
					}
				}
				break;
				case 19:
				{
					{
						result_0 = -27 * y * z;
						val.col(0) = result_0;
					}
					{
						result_0 = -27 * z * (x + 2 * y + z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -27 * y * (x + y + 2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void p_3_nodes_3d(Eigen::MatrixXd &res)
			{
				res.resize(20, 3);
				res << 0, 0, 0,
					1, 0, 0,
					0, 1, 0,
					0, 0, 1,
					1.0 / 3.0, 0, 0,
					2.0 / 3.0, 0, 0,
					2.0 / 3.0, 1.0 / 3.0, 0,
					1.0 / 3.0, 2.0 / 3.0, 0,
					0, 2.0 / 3.0, 0,
					0, 1.0 / 3.0, 0,
					0, 0, 1.0 / 3.0,
					0, 0, 2.0 / 3.0,
					2.0 / 3.0, 0, 1.0 / 3.0,
					1.0 / 3.0, 0, 2.0 / 3.0,
					0, 2.0 / 3.0, 1.0 / 3.0,
					0, 1.0 / 3.0, 2.0 / 3.0,
					1.0 / 3.0, 1.0 / 3.0, 0,
					1.0 / 3.0, 0, 1.0 / 3.0,
					1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
					0, 1.0 / 3.0, 1.0 / 3.0;
			}

			void p_4_basis_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &result_0)
			{

				auto x = uv.col(0).array();
				auto y = uv.col(1).array();
				auto z = uv.col(2).array();

				switch (local_index)
				{
				case 0:
				{
					const auto helper_0 = x + y + z - 1;
					const auto helper_1 = x * y;
					const auto helper_2 = pow(y, 2);
					const auto helper_3 = 9 * x;
					const auto helper_4 = pow(z, 2);
					const auto helper_5 = pow(x, 2);
					const auto helper_6 = 9 * y;
					const auto helper_7 = 9 * z;
					const auto helper_8 = 26 * helper_0;
					const auto helper_9 = helper_8 * z;
					const auto helper_10 = 13 * pow(helper_0, 2);
					const auto helper_11 = 13 * helper_0;
					result_0 = (1.0 / 3.0) * helper_0 * (3 * pow(helper_0, 3) + helper_1 * helper_8 + 18 * helper_1 * z + helper_10 * x + helper_10 * y + helper_10 * z + helper_11 * helper_2 + helper_11 * helper_4 + helper_11 * helper_5 + helper_2 * helper_3 + helper_2 * helper_7 + helper_3 * helper_4 + helper_4 * helper_6 + helper_5 * helper_6 + helper_5 * helper_7 + helper_9 * x + helper_9 * y + 3 * pow(x, 3) + 3 * pow(y, 3) + 3 * pow(z, 3));
				}
				break;
				case 1:
				{
					result_0 = (1.0 / 3.0) * x * (32 * pow(x, 3) - 48 * pow(x, 2) + 22 * x - 3);
				}
				break;
				case 2:
				{
					result_0 = (1.0 / 3.0) * y * (32 * pow(y, 3) - 48 * pow(y, 2) + 22 * y - 3);
				}
				break;
				case 3:
				{
					result_0 = (1.0 / 3.0) * z * (32 * pow(z, 3) - 48 * pow(z, 2) + 22 * z - 3);
				}
				break;
				case 4:
				{
					const auto helper_0 = 36 * x;
					const auto helper_1 = y * z;
					const auto helper_2 = pow(x, 2);
					const auto helper_3 = pow(y, 2);
					const auto helper_4 = pow(z, 2);
					const auto helper_5 = 24 * x;
					const auto helper_6 = 24 * y;
					const auto helper_7 = 24 * z;
					result_0 = -16.0 / 3.0 * x * (-helper_0 * y - helper_0 * z + 48 * helper_1 * x - 36 * helper_1 + helper_2 * helper_6 + helper_2 * helper_7 - 18 * helper_2 + helper_3 * helper_5 + helper_3 * helper_7 - 18 * helper_3 + helper_4 * helper_5 + helper_4 * helper_6 - 18 * helper_4 + 8 * pow(x, 3) + 13 * x + 8 * pow(y, 3) + 13 * y + 8 * pow(z, 3) + 13 * z - 3);
				}
				break;
				case 5:
				{
					const auto helper_0 = 2 * y;
					const auto helper_1 = 2 * z;
					const auto helper_2 = x + y + z - 1;
					const auto helper_3 = helper_2 * x;
					result_0 = 4 * helper_3 * (-helper_0 * helper_2 + helper_0 * x - helper_0 * z - helper_1 * helper_2 + helper_1 * x + 3 * pow(helper_2, 2) + 10 * helper_3 + 3 * pow(x, 2) - pow(y, 2) - pow(z, 2));
				}
				break;
				case 6:
				{
					const auto helper_0 = 6 * x;
					const auto helper_1 = pow(x, 2);
					const auto helper_2 = 8 * helper_1;
					result_0 = -16.0 / 3.0 * x * (-helper_0 * y - helper_0 * z - 14 * helper_1 + helper_2 * y + helper_2 * z + 8 * pow(x, 3) + 7 * x + y + z - 1);
				}
				break;
				case 7:
				{
					result_0 = (16.0 / 3.0) * x * y * (8 * pow(x, 2) - 6 * x + 1);
				}
				break;
				case 8:
				{
					const auto helper_0 = 4 * x;
					result_0 = helper_0 * y * (-helper_0 + 16 * x * y - 4 * y + 1);
				}
				break;
				case 9:
				{
					result_0 = (16.0 / 3.0) * x * y * (8 * pow(y, 2) - 6 * y + 1);
				}
				break;
				case 10:
				{
					const auto helper_0 = 6 * y;
					const auto helper_1 = pow(y, 2);
					const auto helper_2 = 8 * helper_1;
					result_0 = -16.0 / 3.0 * y * (-helper_0 * x - helper_0 * z - 14 * helper_1 + helper_2 * x + helper_2 * z + x + 8 * pow(y, 3) + 7 * y + z - 1);
				}
				break;
				case 11:
				{
					const auto helper_0 = 2 * y;
					const auto helper_1 = 2 * x;
					const auto helper_2 = x + y + z - 1;
					const auto helper_3 = helper_2 * y;
					result_0 = -4 * helper_3 * (-helper_0 * x - helper_0 * z + helper_1 * helper_2 + helper_1 * z - 3 * pow(helper_2, 2) + 2 * helper_2 * z - 10 * helper_3 + pow(x, 2) - 3 * pow(y, 2) + pow(z, 2));
				}
				break;
				case 12:
				{
					const auto helper_0 = 36 * x;
					const auto helper_1 = y * z;
					const auto helper_2 = pow(x, 2);
					const auto helper_3 = pow(y, 2);
					const auto helper_4 = pow(z, 2);
					const auto helper_5 = 24 * x;
					const auto helper_6 = 24 * y;
					const auto helper_7 = 24 * z;
					result_0 = -16.0 / 3.0 * y * (-helper_0 * y - helper_0 * z + 48 * helper_1 * x - 36 * helper_1 + helper_2 * helper_6 + helper_2 * helper_7 - 18 * helper_2 + helper_3 * helper_5 + helper_3 * helper_7 - 18 * helper_3 + helper_4 * helper_5 + helper_4 * helper_6 - 18 * helper_4 + 8 * pow(x, 3) + 13 * x + 8 * pow(y, 3) + 13 * y + 8 * pow(z, 3) + 13 * z - 3);
				}
				break;
				case 13:
				{
					const auto helper_0 = 36 * x;
					const auto helper_1 = y * z;
					const auto helper_2 = pow(x, 2);
					const auto helper_3 = pow(y, 2);
					const auto helper_4 = pow(z, 2);
					const auto helper_5 = 24 * x;
					const auto helper_6 = 24 * y;
					const auto helper_7 = 24 * z;
					result_0 = -16.0 / 3.0 * z * (-helper_0 * y - helper_0 * z + 48 * helper_1 * x - 36 * helper_1 + helper_2 * helper_6 + helper_2 * helper_7 - 18 * helper_2 + helper_3 * helper_5 + helper_3 * helper_7 - 18 * helper_3 + helper_4 * helper_5 + helper_4 * helper_6 - 18 * helper_4 + 8 * pow(x, 3) + 13 * x + 8 * pow(y, 3) + 13 * y + 8 * pow(z, 3) + 13 * z - 3);
				}
				break;
				case 14:
				{
					const auto helper_0 = 2 * x;
					const auto helper_1 = 2 * z;
					const auto helper_2 = x + y + z - 1;
					const auto helper_3 = helper_2 * z;
					result_0 = -4 * helper_3 * (helper_0 * helper_2 + helper_0 * y - helper_1 * x - helper_1 * y - 3 * pow(helper_2, 2) + 2 * helper_2 * y - 10 * helper_3 + pow(x, 2) + pow(y, 2) - 3 * pow(z, 2));
				}
				break;
				case 15:
				{
					const auto helper_0 = 6 * z;
					const auto helper_1 = pow(z, 2);
					const auto helper_2 = 8 * helper_1;
					result_0 = -16.0 / 3.0 * z * (-helper_0 * x - helper_0 * y - 14 * helper_1 + helper_2 * x + helper_2 * y + x + y + 8 * pow(z, 3) + 7 * z - 1);
				}
				break;
				case 16:
				{
					result_0 = (16.0 / 3.0) * x * z * (8 * pow(x, 2) - 6 * x + 1);
				}
				break;
				case 17:
				{
					const auto helper_0 = 4 * x;
					result_0 = helper_0 * z * (-helper_0 + 16 * x * z - 4 * z + 1);
				}
				break;
				case 18:
				{
					result_0 = (16.0 / 3.0) * x * z * (8 * pow(z, 2) - 6 * z + 1);
				}
				break;
				case 19:
				{
					result_0 = (16.0 / 3.0) * y * z * (8 * pow(y, 2) - 6 * y + 1);
				}
				break;
				case 20:
				{
					const auto helper_0 = 4 * y;
					result_0 = helper_0 * z * (-helper_0 + 16 * y * z - 4 * z + 1);
				}
				break;
				case 21:
				{
					result_0 = (16.0 / 3.0) * y * z * (8 * pow(z, 2) - 6 * z + 1);
				}
				break;
				case 22:
				{
					result_0 = 32 * x * y * (x + y + z - 1) * (4 * x + 4 * y + 4 * z - 3);
				}
				break;
				case 23:
				{
					result_0 = -32 * x * y * (4 * y - 1) * (x + y + z - 1);
				}
				break;
				case 24:
				{
					result_0 = -32 * x * y * (4 * x - 1) * (x + y + z - 1);
				}
				break;
				case 25:
				{
					result_0 = 32 * x * z * (x + y + z - 1) * (4 * x + 4 * y + 4 * z - 3);
				}
				break;
				case 26:
				{
					result_0 = -32 * x * z * (4 * z - 1) * (x + y + z - 1);
				}
				break;
				case 27:
				{
					result_0 = -32 * x * z * (4 * x - 1) * (x + y + z - 1);
				}
				break;
				case 28:
				{
					result_0 = 32 * x * y * z * (4 * x - 1);
				}
				break;
				case 29:
				{
					result_0 = 32 * x * y * z * (4 * z - 1);
				}
				break;
				case 30:
				{
					result_0 = 32 * x * y * z * (4 * y - 1);
				}
				break;
				case 31:
				{
					result_0 = -32 * y * z * (4 * y - 1) * (x + y + z - 1);
				}
				break;
				case 32:
				{
					result_0 = -32 * y * z * (4 * z - 1) * (x + y + z - 1);
				}
				break;
				case 33:
				{
					result_0 = 32 * y * z * (x + y + z - 1) * (4 * x + 4 * y + 4 * z - 3);
				}
				break;
				case 34:
				{
					result_0 = -256 * x * y * z * (x + y + z - 1);
				}
				break;
				default:
					assert(false);
				}
			}
			void p_4_basis_grad_value_3d(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
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
						const auto helper_0 = 160 * x;
						const auto helper_1 = y * z;
						const auto helper_2 = pow(x, 2);
						const auto helper_3 = pow(y, 2);
						const auto helper_4 = pow(z, 2);
						const auto helper_5 = 128 * x;
						const auto helper_6 = 128 * y;
						const auto helper_7 = 128 * z;
						result_0 = -helper_0 * y - helper_0 * z + 256 * helper_1 * x - 160 * helper_1 + helper_2 * helper_6 + helper_2 * helper_7 - 80 * helper_2 + helper_3 * helper_5 + helper_3 * helper_7 - 80 * helper_3 + helper_4 * helper_5 + helper_4 * helper_6 - 80 * helper_4 + (128.0 / 3.0) * pow(x, 3) + (140.0 / 3.0) * x + (128.0 / 3.0) * pow(y, 3) + (140.0 / 3.0) * y + (128.0 / 3.0) * pow(z, 3) + (140.0 / 3.0) * z - 25.0 / 3.0;
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 160 * x;
						const auto helper_1 = y * z;
						const auto helper_2 = pow(x, 2);
						const auto helper_3 = pow(y, 2);
						const auto helper_4 = pow(z, 2);
						const auto helper_5 = 128 * x;
						const auto helper_6 = 128 * y;
						const auto helper_7 = 128 * z;
						result_0 = -helper_0 * y - helper_0 * z + 256 * helper_1 * x - 160 * helper_1 + helper_2 * helper_6 + helper_2 * helper_7 - 80 * helper_2 + helper_3 * helper_5 + helper_3 * helper_7 - 80 * helper_3 + helper_4 * helper_5 + helper_4 * helper_6 - 80 * helper_4 + (128.0 / 3.0) * pow(x, 3) + (140.0 / 3.0) * x + (128.0 / 3.0) * pow(y, 3) + (140.0 / 3.0) * y + (128.0 / 3.0) * pow(z, 3) + (140.0 / 3.0) * z - 25.0 / 3.0;
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 160 * x;
						const auto helper_1 = y * z;
						const auto helper_2 = pow(x, 2);
						const auto helper_3 = pow(y, 2);
						const auto helper_4 = pow(z, 2);
						const auto helper_5 = 128 * x;
						const auto helper_6 = 128 * y;
						const auto helper_7 = 128 * z;
						result_0 = -helper_0 * y - helper_0 * z + 256 * helper_1 * x - 160 * helper_1 + helper_2 * helper_6 + helper_2 * helper_7 - 80 * helper_2 + helper_3 * helper_5 + helper_3 * helper_7 - 80 * helper_3 + helper_4 * helper_5 + helper_4 * helper_6 - 80 * helper_4 + (128.0 / 3.0) * pow(x, 3) + (140.0 / 3.0) * x + (128.0 / 3.0) * pow(y, 3) + (140.0 / 3.0) * y + (128.0 / 3.0) * pow(z, 3) + (140.0 / 3.0) * z - 25.0 / 3.0;
						val.col(2) = result_0;
					}
				}
				break;
				case 1:
				{
					{
						result_0 = (128.0 / 3.0) * pow(x, 3) - 48 * pow(x, 2) + (44.0 / 3.0) * x - 1;
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
				case 2:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = (128.0 / 3.0) * pow(y, 3) - 48 * pow(y, 2) + (44.0 / 3.0) * y - 1;
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				case 3:
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
						result_0 = (128.0 / 3.0) * pow(z, 3) - 48 * pow(z, 2) + (44.0 / 3.0) * z - 1;
						val.col(2) = result_0;
					}
				}
				break;
				case 4:
				{
					{
						const auto helper_0 = pow(x, 2);
						const auto helper_1 = pow(y, 2);
						const auto helper_2 = pow(z, 2);
						const auto helper_3 = 16 * x;
						const auto helper_4 = 24 * helper_0;
						result_0 = 288 * helper_0 - 16 * helper_1 * helper_3 - 128 * helper_1 * z + 96 * helper_1 - 16 * helper_2 * helper_3 - 128 * helper_2 * y + 96 * helper_2 - 16 * helper_4 * y - 16 * helper_4 * z - 512.0 / 3.0 * pow(x, 3) - 512 * x * y * z + 384 * x * y + 384 * x * z - 416.0 / 3.0 * x - 128.0 / 3.0 * pow(y, 3) + 192 * y * z - 208.0 / 3.0 * y - 128.0 / 3.0 * pow(z, 3) - 208.0 / 3.0 * z + 16;
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 48 * x;
						result_0 = -16.0 / 3.0 * x * (helper_0 * y + helper_0 * z + 24 * pow(x, 2) - 36 * x + 24 * pow(y, 2) + 48 * y * z - 36 * y + 24 * pow(z, 2) - 36 * z + 13);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 48 * x;
						result_0 = -16.0 / 3.0 * x * (helper_0 * y + helper_0 * z + 24 * pow(x, 2) - 36 * x + 24 * pow(y, 2) + 48 * y * z - 36 * y + 24 * pow(z, 2) - 36 * z + 13);
						val.col(2) = result_0;
					}
				}
				break;
				case 5:
				{
					{
						const auto helper_0 = 72 * x;
						const auto helper_1 = y * z;
						const auto helper_2 = 96 * pow(x, 2);
						const auto helper_3 = pow(y, 2);
						const auto helper_4 = pow(z, 2);
						const auto helper_5 = 32 * x;
						result_0 = -4 * helper_0 * y - 4 * helper_0 * z + 256 * helper_1 * x - 32 * helper_1 + 4 * helper_2 * y + 4 * helper_2 * z - 4 * helper_2 + 4 * helper_3 * helper_5 - 16 * helper_3 + 4 * helper_4 * helper_5 - 16 * helper_4 + 256 * pow(x, 3) + 152 * x + 28 * y + 28 * z - 12;
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 32 * x;
						result_0 = 4 * x * (helper_0 * y + helper_0 * z + 32 * pow(x, 2) - 36 * x - 8 * y - 8 * z + 7);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 32 * x;
						result_0 = 4 * x * (helper_0 * y + helper_0 * z + 32 * pow(x, 2) - 36 * x - 8 * y - 8 * z + 7);
						val.col(2) = result_0;
					}
				}
				break;
				case 6:
				{
					{
						const auto helper_0 = pow(x, 2);
						const auto helper_1 = 8 * helper_0;
						result_0 = 224 * helper_0 - 16 * helper_1 * y - 16 * helper_1 * z - 512.0 / 3.0 * pow(x, 3) + 64 * x * y + 64 * x * z - 224.0 / 3.0 * x - 16.0 / 3.0 * y - 16.0 / 3.0 * z + 16.0 / 3.0;
						val.col(0) = result_0;
					}
					{
						result_0 = -16.0 / 3.0 * x * (8 * pow(x, 2) - 6 * x + 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -16.0 / 3.0 * x * (8 * pow(x, 2) - 6 * x + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 7:
				{
					{
						result_0 = (16.0 / 3.0) * y * (24 * pow(x, 2) - 12 * x + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = (16.0 / 3.0) * x * (8 * pow(x, 2) - 6 * x + 1);
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				case 8:
				{
					{
						const auto helper_0 = 4 * y;
						result_0 = helper_0 * (-helper_0 + 32 * x * y - 8 * x + 1);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 4 * x;
						result_0 = helper_0 * (-helper_0 + 32 * x * y - 8 * y + 1);
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				case 9:
				{
					{
						result_0 = (16.0 / 3.0) * y * (8 * pow(y, 2) - 6 * y + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = (16.0 / 3.0) * x * (24 * pow(y, 2) - 12 * y + 1);
						val.col(1) = result_0;
					}
					{
						result_0.setZero();
						val.col(2) = result_0;
					}
				}
				break;
				case 10:
				{
					{
						result_0 = -16.0 / 3.0 * y * (8 * pow(y, 2) - 6 * y + 1);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = pow(y, 2);
						const auto helper_1 = 8 * helper_0;
						result_0 = 224 * helper_0 - 16 * helper_1 * x - 16 * helper_1 * z + 64 * x * y - 16.0 / 3.0 * x - 512.0 / 3.0 * pow(y, 3) + 64 * y * z - 224.0 / 3.0 * y - 16.0 / 3.0 * z + 16.0 / 3.0;
						val.col(1) = result_0;
					}
					{
						result_0 = -16.0 / 3.0 * y * (8 * pow(y, 2) - 6 * y + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 11:
				{
					{
						const auto helper_0 = 32 * y;
						result_0 = 4 * y * (helper_0 * x + helper_0 * z - 8 * x + 32 * pow(y, 2) - 36 * y - 8 * z + 7);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 72 * y;
						const auto helper_1 = x * z;
						const auto helper_2 = pow(x, 2);
						const auto helper_3 = 96 * pow(y, 2);
						const auto helper_4 = pow(z, 2);
						const auto helper_5 = 32 * y;
						result_0 = -4 * helper_0 * x - 4 * helper_0 * z + 256 * helper_1 * y - 32 * helper_1 + 4 * helper_2 * helper_5 - 16 * helper_2 + 4 * helper_3 * x + 4 * helper_3 * z - 4 * helper_3 + 4 * helper_4 * helper_5 - 16 * helper_4 + 28 * x + 256 * pow(y, 3) + 152 * y + 28 * z - 12;
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 32 * y;
						result_0 = 4 * y * (helper_0 * x + helper_0 * z - 8 * x + 32 * pow(y, 2) - 36 * y - 8 * z + 7);
						val.col(2) = result_0;
					}
				}
				break;
				case 12:
				{
					{
						const auto helper_0 = 48 * x;
						result_0 = -16.0 / 3.0 * y * (helper_0 * y + helper_0 * z + 24 * pow(x, 2) - 36 * x + 24 * pow(y, 2) + 48 * y * z - 36 * y + 24 * pow(z, 2) - 36 * z + 13);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = pow(x, 2);
						const auto helper_1 = pow(y, 2);
						const auto helper_2 = pow(z, 2);
						const auto helper_3 = 24 * helper_1;
						const auto helper_4 = 16 * y;
						result_0 = -16 * helper_0 * helper_4 - 128 * helper_0 * z + 96 * helper_0 + 288 * helper_1 - 16 * helper_2 * helper_4 - 128 * helper_2 * x + 96 * helper_2 - 16 * helper_3 * x - 16 * helper_3 * z - 128.0 / 3.0 * pow(x, 3) - 512 * x * y * z + 384 * x * y + 192 * x * z - 208.0 / 3.0 * x - 512.0 / 3.0 * pow(y, 3) + 384 * y * z - 416.0 / 3.0 * y - 128.0 / 3.0 * pow(z, 3) - 208.0 / 3.0 * z + 16;
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 48 * x;
						result_0 = -16.0 / 3.0 * y * (helper_0 * y + helper_0 * z + 24 * pow(x, 2) - 36 * x + 24 * pow(y, 2) + 48 * y * z - 36 * y + 24 * pow(z, 2) - 36 * z + 13);
						val.col(2) = result_0;
					}
				}
				break;
				case 13:
				{
					{
						const auto helper_0 = 48 * x;
						result_0 = -16.0 / 3.0 * z * (helper_0 * y + helper_0 * z + 24 * pow(x, 2) - 36 * x + 24 * pow(y, 2) + 48 * y * z - 36 * y + 24 * pow(z, 2) - 36 * z + 13);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 48 * x;
						result_0 = -16.0 / 3.0 * z * (helper_0 * y + helper_0 * z + 24 * pow(x, 2) - 36 * x + 24 * pow(y, 2) + 48 * y * z - 36 * y + 24 * pow(z, 2) - 36 * z + 13);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = pow(x, 2);
						const auto helper_1 = pow(y, 2);
						const auto helper_2 = pow(z, 2);
						const auto helper_3 = 24 * helper_2;
						const auto helper_4 = 16 * z;
						result_0 = -16 * helper_0 * helper_4 - 128 * helper_0 * y + 96 * helper_0 - 16 * helper_1 * helper_4 - 128 * helper_1 * x + 96 * helper_1 + 288 * helper_2 - 16 * helper_3 * x - 16 * helper_3 * y - 128.0 / 3.0 * pow(x, 3) - 512 * x * y * z + 192 * x * y + 384 * x * z - 208.0 / 3.0 * x - 128.0 / 3.0 * pow(y, 3) + 384 * y * z - 208.0 / 3.0 * y - 512.0 / 3.0 * pow(z, 3) - 416.0 / 3.0 * z + 16;
						val.col(2) = result_0;
					}
				}
				break;
				case 14:
				{
					{
						const auto helper_0 = 32 * z;
						result_0 = 4 * z * (helper_0 * x + helper_0 * y - 8 * x - 8 * y + 32 * pow(z, 2) - 36 * z + 7);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 32 * z;
						result_0 = 4 * z * (helper_0 * x + helper_0 * y - 8 * x - 8 * y + 32 * pow(z, 2) - 36 * z + 7);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = x * y;
						const auto helper_1 = 72 * z;
						const auto helper_2 = pow(x, 2);
						const auto helper_3 = pow(y, 2);
						const auto helper_4 = 96 * pow(z, 2);
						const auto helper_5 = 32 * z;
						result_0 = 256 * helper_0 * z - 32 * helper_0 - 4 * helper_1 * x - 4 * helper_1 * y + 4 * helper_2 * helper_5 - 16 * helper_2 + 4 * helper_3 * helper_5 - 16 * helper_3 + 4 * helper_4 * x + 4 * helper_4 * y - 4 * helper_4 + 28 * x + 28 * y + 256 * pow(z, 3) + 152 * z - 12;
						val.col(2) = result_0;
					}
				}
				break;
				case 15:
				{
					{
						result_0 = -16.0 / 3.0 * z * (8 * pow(z, 2) - 6 * z + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -16.0 / 3.0 * z * (8 * pow(z, 2) - 6 * z + 1);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = pow(z, 2);
						const auto helper_1 = 8 * helper_0;
						result_0 = 224 * helper_0 - 16 * helper_1 * x - 16 * helper_1 * y + 64 * x * z - 16.0 / 3.0 * x + 64 * y * z - 16.0 / 3.0 * y - 512.0 / 3.0 * pow(z, 3) - 224.0 / 3.0 * z + 16.0 / 3.0;
						val.col(2) = result_0;
					}
				}
				break;
				case 16:
				{
					{
						result_0 = (16.0 / 3.0) * z * (24 * pow(x, 2) - 12 * x + 1);
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
					{
						result_0 = (16.0 / 3.0) * x * (8 * pow(x, 2) - 6 * x + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 17:
				{
					{
						const auto helper_0 = 4 * z;
						result_0 = helper_0 * (-helper_0 + 32 * x * z - 8 * x + 1);
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 4 * x;
						result_0 = helper_0 * (-helper_0 + 32 * x * z - 8 * z + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 18:
				{
					{
						result_0 = (16.0 / 3.0) * z * (8 * pow(z, 2) - 6 * z + 1);
						val.col(0) = result_0;
					}
					{
						result_0.setZero();
						val.col(1) = result_0;
					}
					{
						result_0 = (16.0 / 3.0) * x * (24 * pow(z, 2) - 12 * z + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 19:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = (16.0 / 3.0) * z * (24 * pow(y, 2) - 12 * y + 1);
						val.col(1) = result_0;
					}
					{
						result_0 = (16.0 / 3.0) * y * (8 * pow(y, 2) - 6 * y + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 20:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 4 * z;
						result_0 = helper_0 * (-helper_0 + 32 * y * z - 8 * y + 1);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 4 * y;
						result_0 = helper_0 * (-helper_0 + 32 * y * z - 8 * z + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 21:
				{
					{
						result_0.setZero();
						val.col(0) = result_0;
					}
					{
						result_0 = (16.0 / 3.0) * z * (8 * pow(z, 2) - 6 * z + 1);
						val.col(1) = result_0;
					}
					{
						result_0 = (16.0 / 3.0) * y * (24 * pow(z, 2) - 12 * z + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 22:
				{
					{
						const auto helper_0 = 16 * x;
						result_0 = 32 * y * (helper_0 * y + helper_0 * z + 12 * pow(x, 2) - 14 * x + 4 * pow(y, 2) + 8 * y * z - 7 * y + 4 * pow(z, 2) - 7 * z + 3);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 16 * y;
						result_0 = 32 * x * (helper_0 * x + helper_0 * z + 4 * pow(x, 2) + 8 * x * z - 7 * x + 12 * pow(y, 2) - 14 * y + 4 * pow(z, 2) - 7 * z + 3);
						val.col(1) = result_0;
					}
					{
						result_0 = 32 * x * y * (8 * x + 8 * y + 8 * z - 7);
						val.col(2) = result_0;
					}
				}
				break;
				case 23:
				{
					{
						result_0 = -32 * y * (8 * x * y - 2 * x + 4 * pow(y, 2) + 4 * y * z - 5 * y - z + 1);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 8 * y;
						result_0 = -32 * x * (helper_0 * x + helper_0 * z - x + 12 * pow(y, 2) - 10 * y - z + 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -32 * x * y * (4 * y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 24:
				{
					{
						const auto helper_0 = 8 * x;
						result_0 = -32 * y * (helper_0 * y + helper_0 * z + 12 * pow(x, 2) - 10 * x - y - z + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -32 * x * (4 * pow(x, 2) + 8 * x * y + 4 * x * z - 5 * x - 2 * y - z + 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -32 * x * y * (4 * x - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 25:
				{
					{
						const auto helper_0 = 16 * x;
						result_0 = 32 * z * (helper_0 * y + helper_0 * z + 12 * pow(x, 2) - 14 * x + 4 * pow(y, 2) + 8 * y * z - 7 * y + 4 * pow(z, 2) - 7 * z + 3);
						val.col(0) = result_0;
					}
					{
						result_0 = 32 * x * z * (8 * x + 8 * y + 8 * z - 7);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 16 * z;
						result_0 = 32 * x * (helper_0 * x + helper_0 * y + 4 * pow(x, 2) + 8 * x * y - 7 * x + 4 * pow(y, 2) - 7 * y + 12 * pow(z, 2) - 14 * z + 3);
						val.col(2) = result_0;
					}
				}
				break;
				case 26:
				{
					{
						result_0 = -32 * z * (8 * x * z - 2 * x + 4 * y * z - y + 4 * pow(z, 2) - 5 * z + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -32 * x * z * (4 * z - 1);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 8 * z;
						result_0 = -32 * x * (helper_0 * x + helper_0 * y - x - y + 12 * pow(z, 2) - 10 * z + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 27:
				{
					{
						const auto helper_0 = 8 * x;
						result_0 = -32 * z * (helper_0 * y + helper_0 * z + 12 * pow(x, 2) - 10 * x - y - z + 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -32 * x * z * (4 * x - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -32 * x * (4 * pow(x, 2) + 4 * x * y + 8 * x * z - 5 * x - y - 2 * z + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 28:
				{
					{
						result_0 = 32 * y * z * (8 * x - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 32 * x * z * (4 * x - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 32 * x * y * (4 * x - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 29:
				{
					{
						result_0 = 32 * y * z * (4 * z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 32 * x * z * (4 * z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 32 * x * y * (8 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 30:
				{
					{
						result_0 = 32 * y * z * (4 * y - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = 32 * x * z * (8 * y - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = 32 * x * y * (4 * y - 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 31:
				{
					{
						result_0 = -32 * y * z * (4 * y - 1);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 8 * y;
						result_0 = -32 * z * (helper_0 * x + helper_0 * z - x + 12 * pow(y, 2) - 10 * y - z + 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -32 * y * (4 * x * y - x + 4 * pow(y, 2) + 8 * y * z - 5 * y - 2 * z + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 32:
				{
					{
						result_0 = -32 * y * z * (4 * z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -32 * z * (4 * x * z - x + 8 * y * z - 2 * y + 4 * pow(z, 2) - 5 * z + 1);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 8 * z;
						result_0 = -32 * y * (helper_0 * x + helper_0 * y - x - y + 12 * pow(z, 2) - 10 * z + 1);
						val.col(2) = result_0;
					}
				}
				break;
				case 33:
				{
					{
						result_0 = 32 * y * z * (8 * x + 8 * y + 8 * z - 7);
						val.col(0) = result_0;
					}
					{
						const auto helper_0 = 16 * y;
						result_0 = 32 * z * (helper_0 * x + helper_0 * z + 4 * pow(x, 2) + 8 * x * z - 7 * x + 12 * pow(y, 2) - 14 * y + 4 * pow(z, 2) - 7 * z + 3);
						val.col(1) = result_0;
					}
					{
						const auto helper_0 = 16 * z;
						result_0 = 32 * y * (helper_0 * x + helper_0 * y + 4 * pow(x, 2) + 8 * x * y - 7 * x + 4 * pow(y, 2) - 7 * y + 12 * pow(z, 2) - 14 * z + 3);
						val.col(2) = result_0;
					}
				}
				break;
				case 34:
				{
					{
						result_0 = -256 * y * z * (2 * x + y + z - 1);
						val.col(0) = result_0;
					}
					{
						result_0 = -256 * x * z * (x + 2 * y + z - 1);
						val.col(1) = result_0;
					}
					{
						result_0 = -256 * x * y * (x + y + 2 * z - 1);
						val.col(2) = result_0;
					}
				}
				break;
				default:
					assert(false);
				}
			}

			void p_4_nodes_3d(Eigen::MatrixXd &res)
			{
				res.resize(35, 3);
				res << 0, 0, 0,
					1, 0, 0,
					0, 1, 0,
					0, 0, 1,
					1.0 / 4.0, 0, 0,
					1.0 / 2.0, 0, 0,
					3.0 / 4.0, 0, 0,
					3.0 / 4.0, 1.0 / 4.0, 0,
					1.0 / 2.0, 1.0 / 2.0, 0,
					1.0 / 4.0, 3.0 / 4.0, 0,
					0, 3.0 / 4.0, 0,
					0, 1.0 / 2.0, 0,
					0, 1.0 / 4.0, 0,
					0, 0, 1.0 / 4.0,
					0, 0, 1.0 / 2.0,
					0, 0, 3.0 / 4.0,
					3.0 / 4.0, 0, 1.0 / 4.0,
					1.0 / 2.0, 0, 1.0 / 2.0,
					1.0 / 4.0, 0, 3.0 / 4.0,
					0, 3.0 / 4.0, 1.0 / 4.0,
					0, 1.0 / 2.0, 1.0 / 2.0,
					0, 1.0 / 4.0, 3.0 / 4.0,
					1.0 / 4.0, 1.0 / 4.0, 0,
					1.0 / 4.0, 1.0 / 2.0, 0,
					1.0 / 2.0, 1.0 / 4.0, 0,
					1.0 / 4.0, 0, 1.0 / 4.0,
					1.0 / 4.0, 0, 1.0 / 2.0,
					1.0 / 2.0, 0, 1.0 / 4.0,
					1.0 / 2.0, 1.0 / 4.0, 1.0 / 4.0,
					1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0,
					1.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0,
					0, 1.0 / 2.0, 1.0 / 4.0,
					0, 1.0 / 4.0, 1.0 / 2.0,
					0, 1.0 / 4.0, 1.0 / 4.0,
					1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0;
			}

		} // namespace

		void p_nodes_3d(const int p, Eigen::MatrixXd &val)
		{
			switch (p)
			{
			case 0:
				p_0_nodes_3d(val);
				break;
			case 1:
				p_1_nodes_3d(val);
				break;
			case 2:
				p_2_nodes_3d(val);
				break;
			case 3:
				p_3_nodes_3d(val);
				break;
			case 4:
				p_4_nodes_3d(val);
				break;
			default:
				p_n_nodes_3d(p, val);
			}
		}
		void p_basis_value_3d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			switch (p)
			{
			case 0:
				p_0_basis_value_3d(local_index, uv, val);
				break;
			case 1:
				p_1_basis_value_3d(local_index, uv, val);
				break;
			case 2:
				p_2_basis_value_3d(local_index, uv, val);
				break;
			case 3:
				p_3_basis_value_3d(local_index, uv, val);
				break;
			case 4:
				p_4_basis_value_3d(local_index, uv, val);
				break;
			default:
				p_n_basis_value_3d(p, local_index, uv, val);
			}
		}

		void p_grad_basis_value_3d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			switch (p)
			{
			case 0:
				p_0_basis_grad_value_3d(local_index, uv, val);
				break;
			case 1:
				p_1_basis_grad_value_3d(local_index, uv, val);
				break;
			case 2:
				p_2_basis_grad_value_3d(local_index, uv, val);
				break;
			case 3:
				p_3_basis_grad_value_3d(local_index, uv, val);
				break;
			case 4:
				p_4_basis_grad_value_3d(local_index, uv, val);
				break;
			default:
				p_n_basis_grad_value_3d(p, local_index, uv, val);
			}
		}

		namespace
		{

		}
	} // namespace autogen
} // namespace polyfem
