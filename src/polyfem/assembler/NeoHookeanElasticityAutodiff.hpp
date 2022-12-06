#pragma once

#include "MatParams.hpp"
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <functional>
#include <iostream>
#include <vector>

/// Used for test only
namespace polyfem::assembler
{
	class NeoHookeanAutodiff
	{
	public:
		NeoHookeanAutodiff();

		// sets material params
		void add_multimaterial(const int index, const json &params, const int size);

		template <typename T>
		T elastic_energy(const int size,
						 const RowVectorNd &p,
						 const int el_id,
						 const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &disp_grad) const
		{
			const double t = 0; // TODO

			// Id + grad d
			auto def_grad = disp_grad;
			for (int d = 0; d < size; ++d)
				def_grad(d, d) += T(1);

			double lambda, mu;
			params_.lambda_mu(p, p, el_id, lambda, mu);

			const T log_det_j = log(polyfem::utils::determinant(def_grad));
			const T val = mu / 2 * ((def_grad.transpose() * def_grad).trace() - size - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			return val;
		}

	private:
		LameParameters params_;
	};
} // namespace polyfem::assembler