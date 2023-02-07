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
	class AMIPSEnergy
	{
	public:
		AMIPSEnergy();

		// sets material params
		void add_multimaterial(const int index, const json &params, const int size);

		template <typename T>
		T elastic_energy(const int size,
						 const RowVectorNd &p,
						 const int el_id,
						 const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &disp_grad) const
		{
			using std::pow;

			auto def_grad = disp_grad;
			for (int d = 0; d < size; ++d)
				def_grad(d, d) += T(1);

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> J(size, size);
			for (int i = 0; i < size; ++i)
				for (int j = 0; j < size; ++j)
				{
					J(i, j) = T(0);
					for (int k = 0; k < size; ++k)
						J(i, j) += def_grad(i, k) * canonical_transformation_[el_id](k, j);
				}

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> JtJ;
			JtJ = J.transpose() * J;

			const T val = JtJ.diagonal().sum() / pow(polyfem::utils::determinant(J), 2. / size);

			return val;
		}

	private:
		std::vector<Eigen::MatrixXd> canonical_transformation_;
	};
} // namespace polyfem::assembler