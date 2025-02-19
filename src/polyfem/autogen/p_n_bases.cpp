#include "p_n_bases.hpp"
#include <cmath>
#include <cassert>

namespace polyfem
{
	namespace autogen
	{
		Eigen::Vector3i convert_local_index_to_ijk(const int local_index, const int p)
		{
			assert(p > 1);
			Eigen::Vector3i ijk;
			ijk.setZero();
			if (local_index < 3)
			{ // vertices
				ijk((local_index + 2) % 3) = p;
			}
			else if (local_index >= 3 && local_index < 3 * p) // edges
			{
				int a = (local_index - 3) / (p - 1); // which edge
				int b = (local_index - 3) % (p - 1); // location on the edge
				ijk(a) = b + 1;
				ijk((a + 2) % 3) = p - b - 1;
			}
			else
			{ // interior
				int t0 = local_index - 3 * p + 1;
				int t1 = p - 2;
				ijk(0) = 1;
				while (t0 > t1)
				{
					t0 -= t1;
					t1 -= 1;
					ijk(0) += 1;
				}
				ijk(1) = t0;
				ijk(2) = p - ijk(0) - ijk(1);
			}
			return ijk;
		}

		Eigen::ArrayXd P(const int m, const int p, const Eigen::ArrayXd &z)
		{
			Eigen::ArrayXd result;
			result.resize(z.size());
			result.setConstant(1);
			if (m >= 1)
			{
				for (int i = 1; i <= m; ++i)
				{
					result *= (p * z - i + 1) / double(i);
				}
			}
			return result;
		}

		Eigen::ArrayXd P_prime(const int m, const int p, const Eigen::ArrayXd &z)
		{
			Eigen::ArrayXd result, tmp;
			result.resize(z.size());
			tmp.resize(z.size());
			result.setZero();
			if (m >= 1)
			{
				for (int i = 1; i <= m; ++i)
				{
					tmp.setConstant(1);
					for (int j = 1; j <= m; ++j)
					{
						if (j != i)
						{
							tmp *= (p * z - j + 1) / double(j);
						}
					}
					result += tmp * p / double(i);
				}
			}
			return result;
		}

		void p_n_nodes_3d(const int p, Eigen::MatrixXd &val)
		{
			int size = ((p + 3) * (p + 2) * (p + 1)) / 6;
			val.resize(size, 3);
			val.setZero();
			val.topRows(4) << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
			if (p > 1)
			{
				for (int i = 1; i < p; i++)
				{
					double x = i / (double)p;
					double y = 1 - x;
					val.row(i + 3 + (p - 1) * 0) << x, 0, 0;
					val.row(i + 3 + (p - 1) * 1) << y, x, 0;
					val.row(i + 3 + (p - 1) * 2) << 0, y, 0;
					val.row(i + 3 + (p - 1) * 3) << 0, 0, x;
					val.row(i + 3 + (p - 1) * 4) << y, 0, x;
					val.row(i + 3 + (p - 1) * 5) << 0, y, x;
				}
			}
			if (p > 2)
			{
				int start_id = 4 + 6 * (p - 1);
				int n_face_nodes = ((p - 1) * (p - 2)) / 2;
				for (int i = 1, idx = 0; i < p; i++)
					for (int j = 1; i + j < p; j++)
					{
						double x = i / (double)p, y = j / (double)p;
						double z = 1 - x - y;
						val.row(start_id + idx + n_face_nodes * 0) << x, y, 0;
						val.row(start_id + idx + n_face_nodes * 1) << x, 0, y;
						val.row(start_id + idx + n_face_nodes * 2) << z, x, y;
						val.row(start_id + idx + n_face_nodes * 3) << 0, z, y;
						idx++;
					}
			}
			if (p > 3)
			{
				int start_id = 4 + 6 * (p - 1) + 2 * (p - 1) * (p - 2);
				for (int i = 1, idx = 0; i < p; i++)
					for (int j = 1; i + j < p; j++)
						for (int k = 1; i + j + k < p; k++)
						{
							val.row(start_id + idx) << i / (double)p, j / (double)p, k / (double)p;
							idx++;
						}
			}
		}

		void p_n_nodes_2d(const int p, Eigen::MatrixXd &val)
		{
			int size = (p + 1) * (p + 2) / 2;
			val.resize(size, 2);
			val.setZero();
			val.topRows(3) << 0, 0, 1, 0, 0, 1;
			if (p > 1)
			{
				for (int i = 0; i < p - 1; i++)
				{
					val.row(i + 3) << (i + 1) / double(p), 0;                             // 3+i
					val.row(i + p + 2) << 1.0 - (i + 1) / double(p), (i + 1) / double(p); // 3+(p-1)+i
					val.row(i + p * 2 + 1) << 0, 1.0 - (i + 1) / double(p);               // 3+2(p-1)+i
				}
			}
			if (p > 2)
			{
				int inner_node_id = 3 * p;
				for (int i = 0; i < p - 2; i++)
					for (int j = 0; j < p - i - 2; j++)
					{
						val.row(inner_node_id) << (i + 1) / double(p), (j + 1) / double(p);
						inner_node_id++;
					}
			}
		}

		void p_n_basis_value_3d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			auto x = uv.col(0).array();
			auto y = uv.col(1).array();
			auto z = uv.col(2).array();
			p_n_nodes_3d(p, val);
			int i = round(val(local_index, 0) * p), j = round(val(local_index, 1) * p), k = round(val(local_index, 2) * p);
			val = P(i, p, x) * P(j, p, y) * P(k, p, z) * P(p - i - j - k, p, 1 - x - y - z);
		}

		void p_n_basis_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			auto x = uv.col(0).array();
			auto y = uv.col(1).array();
			Eigen::Vector3i ijk;
			ijk = convert_local_index_to_ijk(local_index, p);
			val = P(ijk(0), p, x) * P(ijk(1), p, y) * P(ijk(2), p, 1 - x - y);
		}

		void p_n_basis_grad_value_3d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			auto x = uv.col(0).array();
			auto y = uv.col(1).array();
			auto z = uv.col(2).array();
			auto w = 1 - x - y - z;
			p_n_nodes_3d(p, val);
			int i = round(val(local_index, 0) * p), j = round(val(local_index, 1) * p), k = round(val(local_index, 2) * p);
			int l = p - i - j - k;
			val.resize(uv.rows(), uv.cols());
			auto value_x = P(i, p, x), value_y = P(j, p, y), value_z = P(k, p, z);
			auto value_w = P(l, p, w), derivative_w = P_prime(l, p, w);
			val.col(0) = value_y * value_z * (P_prime(i, p, x) * value_w - value_x * derivative_w);
			val.col(1) = value_x * value_z * (P_prime(j, p, y) * value_w - value_y * derivative_w);
			val.col(2) = value_x * value_y * (P_prime(k, p, z) * value_w - value_z * derivative_w);
		}

		void p_n_basis_grad_value_2d(const int p, const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
		{
			auto x = uv.col(0).array();
			auto y = uv.col(1).array();
			Eigen::Vector3i ijk;
			ijk = convert_local_index_to_ijk(local_index, p);
			val.resize(uv.rows(), uv.cols());
			val.col(0) = P(ijk(1), p, y) * (P_prime(ijk(0), p, x) * P(ijk(2), p, 1 - x - y) - P(ijk(0), p, x) * P_prime(ijk(2), p, 1 - x - y));
			val.col(1) = P(ijk(0), p, x) * (P_prime(ijk(1), p, y) * P(ijk(2), p, 1 - x - y) - P(ijk(1), p, y) * P_prime(ijk(2), p, 1 - x - y));
		}
	} // namespace autogen
} // namespace polyfem
