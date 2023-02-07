#pragma once

#include "MatParams.hpp"
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <functional>
#include <iostream>
#include <vector>

// non linear MooneyRivlin material model
namespace polyfem::assembler
{
	class MooneyRivlinElasticity
	{
	public:
		MooneyRivlinElasticity();

		// sets material params
		void add_multimaterial(const int index, const json &params, const int size);

		template <typename T>
		T elastic_energy(const int size,
						 const RowVectorNd &p,
						 const int el_id,
						 const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &disp_grad) const
		{
			const double t = 0; // TODO

			const double c1 = c1_(p, t, el_id);
			const double c2 = c2_(p, t, el_id);
			const double k = k_(p, t, el_id);

			auto def_grad = disp_grad;
			for (int d = 0; d < size; ++d)
				def_grad(d, d) += T(1);

			const T det_j = polyfem::utils::determinant(def_grad);
			const T log_det_j = log(det_j);

			const auto F_tilde = def_grad / pow(det_j, 1 / 3.0);
			const auto C_tilde = F_tilde * F_tilde.transpose();
			const auto I1_tilde = first_invariant(C_tilde);
			const auto I2_tilde = second_invariant(C_tilde);

			const T val = c1 * (I1_tilde - size) + c2 * (I2_tilde - size) + k / 2 * (log_det_j * log_det_j);

			return val;
		}

		const GenericMatParam &c1() const { return c1_; }
		const GenericMatParam &c2() const { return c2_; }
		const GenericMatParam &k() const { return k_; }

	private:
		GenericMatParam c1_;
		GenericMatParam c2_;
		GenericMatParam k_;
	};
} // namespace polyfem::assembler