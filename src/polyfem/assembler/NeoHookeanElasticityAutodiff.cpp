#include "NeoHookeanElasticityAutodiff.hpp"

namespace polyfem::assembler
{
	NeoHookeanAutodiff::NeoHookeanAutodiff()
	{
	}

	void NeoHookeanAutodiff::add_multimaterial(const int index, const json &params)
	{
		params_.add_multimaterial(index, params, size() == 3);
	}

	template <typename T>
	T NeoHookeanAutodiff::elastic_energy_T(
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

	// This macro defines the template specializations for NeoHookeanAutodiff::elastic_energy_T
	POLYFEM_TEMPLATE_SPECIALIZE_ELASTIC_ENERGY(NeoHookeanAutodiff)
} // namespace polyfem::assembler