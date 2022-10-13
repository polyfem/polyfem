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
			const double t = 0; // TODO

			const double c1 = c1_(p, t, el_id);
			const double c2 = c2_(p, t, el_id);
			const double k = k_(p, t, el_id);

			auto def_grad = disp_grad;
			for (int d = 0; d < size; ++d)
				def_grad(d, d) += T(1);

			const T log_det_j = log(polyfem::utils::determinant(def_grad));
			const T val = c1 * ((def_grad.transpose() * def_grad).trace() - size) + c2 * ((def_grad.transpose() * def_grad * def_grad.transpose() * def_grad).trace() - size) + k / 2 * (log_det_j * log_det_j);

			return val;
		}

	private:
		GenericMatParam c1_;
		GenericMatParam c2_;
		GenericMatParam k_;
	};
} // namespace polyfem::assembler