#pragma once

#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <functional>
#include <vector>

// non linear NeoHookean material model
namespace polyfem::assembler
{
	class MooneyRivlinElasticity
	{
	public:
		// sets material params
		void add_multimaterial(const int index, const json &params);

		void stress_from_disp_grad(const int size,
								   const RowVectorNd &p,
								   const int el_id,
								   const Eigen::MatrixXd &displacement_grad,
								   Eigen::MatrixXd &stress_tensor) const;

		template <typename T>
		T elastic_energy(const int size,
						 const RowVectorNd &p,
						 const int el_id,
						 const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &disp_grad) const
		{
			assert(c1_.size() == 1 || el_id < c1_.size());

			const double x = p(0);
			const double y = p(1);
			const double z = size == 3 ? p(2) : 0;
			const double t = 0; // TODO

			const auto &tmp_c1 = c1_.size() == 1 ? c1_[0] : c1_[el_id];
			const auto &tmp_c2 = c2_.size() == 1 ? c2_[0] : c2_[el_id];
			const auto &tmp_k = k_.size() == 1 ? k_[0] : k_[el_id];

			const double c1 = tmp_c1(x, y, z, t, el_id);
			const double c2 = tmp_c2(x, y, z, t, el_id);
			const double k = tmp_k(x, y, z, t, el_id);

			const T log_det_j = log(polyfem::utils::determinant(disp_grad));
			const T val = c1 * ((disp_grad.transpose() * disp_grad).trace() - size) + c2 * ((disp_grad.transpose() * disp_grad * disp_grad.transpose() * disp_grad).trace() - size) + k / 2 * (log_det_j * log_det_j);

			return val;
		}

	private:
		std::vector<utils::ExpressionValue> c1_;
		std::vector<utils::ExpressionValue> c2_;
		std::vector<utils::ExpressionValue> k_;
	};
} // namespace polyfem::assembler