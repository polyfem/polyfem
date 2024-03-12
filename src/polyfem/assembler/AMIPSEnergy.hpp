#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <functional>
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
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			MatrixN<T> F = MatrixN<T>::Zero(domain_size(), codomain_size());
			for (int i = 0; i < domain_size(); ++i)
			{
				for (int j = 0; j < codomain_size(); ++j)
				{
					for (int k = 0; k < codomain_size(); ++k)
					{
						F(i, j) += def_grad(i, k) * canonical_transformation_[el_id](k, j);
					}
				}
			}

			const T J = polyfem::utils::determinant(F);
			return (F.transpose() * F).trace() / pow(J, 2. / domain_size());
		}

	private:
		std::vector<MatrixNd> canonical_transformation_;
	};
} // namespace polyfem::assembler