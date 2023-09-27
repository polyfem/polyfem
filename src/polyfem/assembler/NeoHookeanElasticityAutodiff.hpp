#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

/// Used for test only
namespace polyfem::assembler
{
	class NeoHookeanAutodiff : public GenericElastic<NeoHookeanAutodiff>
	{
	public:
		std::string name() const override { return "NeoHookeanAutodiff"; }
		// Used only for testing, no need to output
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

		NeoHookeanAutodiff();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			double lambda, mu;
			params_.lambda_mu(p, p, el_id, lambda, mu);

			const T log_det_j = log(polyfem::utils::determinant(def_grad));
			const T val = mu / 2 * ((def_grad * def_grad.transpose()).trace() - size() - 2 * log_det_j) + lambda / 2 * log_det_j * log_det_j;

			return val;
		}

	private:
		LameParameters params_;
	};
} // namespace polyfem::assembler