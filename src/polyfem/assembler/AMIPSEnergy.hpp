#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <functional>
#include <iostream>
#include <vector>

namespace polyfem::assembler
{
	class AMIPSEnergy : public GenericElastic<AMIPSEnergy>
	{
	public:
		AMIPSEnergy();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		std::string name() const override { return "AMIPS"; }
		std::map<std::string, ParamFunc> parameters() const override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			using std::pow;
			using MatrixNT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>;

			MatrixNT F = MatrixNT::Zero(size(), size());
			for (int i = 0; i < size(); ++i)
			{
				for (int j = 0; j < size(); ++j)
				{
					for (int k = 0; k < size(); ++k)
					{
						F(i, j) += def_grad(i, k) * canonical_transformation_[el_id](k, j);
					}
				}
			}

			const T J = polyfem::utils::determinant(F);
			return (F.transpose() * F).trace() / pow(J, 2. / size());
		}

	private:
		std::vector<Eigen::MatrixXd> canonical_transformation_;
	};
} // namespace polyfem::assembler