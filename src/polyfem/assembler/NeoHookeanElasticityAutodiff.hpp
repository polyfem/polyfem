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
						 const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &def_grad) const
		{
			const double t = 0; // TODO

			double lambda, mu;
			params_.lambda_mu(p, p, el_id, lambda, mu);

			const T log_det_j = log(polyfem::utils::determinant(def_grad));
			const T val = mu / 2 * ((def_grad * def_grad.transpose()).trace() - size - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			return val;
		}

	private:
		LameParameters params_;
	};
} // namespace polyfem::assembler