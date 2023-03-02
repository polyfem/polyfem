#include "MooneyRivlinElasticity.hpp"

#include <polyfem/basis/Basis.hpp>

namespace polyfem::assembler
{
	MooneyRivlinElasticity::MooneyRivlinElasticity()
		: c1_("c1"), c2_("c2"), k_("k")
	{
	}

	void MooneyRivlinElasticity::add_multimaterial(const int index, const json &params)
	{
		c1_.add_multimaterial(index, params);
		c2_.add_multimaterial(index, params);
		k_.add_multimaterial(index, params);
	}

	template <typename T>
	T MooneyRivlinElasticity::elastic_energy_T(
		const RowVectorNd &p,
		const int el_id,
		const DefGradMatrix<T> &def_grad) const
	{
		const double t = 0; // TODO

		const double c1 = c1_(p, t, el_id);
		const double c2 = c2_(p, t, el_id);
		const double k = k_(p, t, el_id);

		const T J = polyfem::utils::determinant(def_grad);
		const T log_J = log(J);

		const auto F_tilde = def_grad / pow(J, 1 / size());
		const auto C_tilde = F_tilde * F_tilde.transpose();
		const auto I1_tilde = first_invariant(C_tilde);
		const auto I2_tilde = second_invariant(C_tilde);

		const T val = c1 * (I1_tilde - size()) + c2 * (I2_tilde - size()) + k / 2 * (log_J * log_J);

		return val;
	}

	// This macro defines the template specializations for MooneyRivlinElasticity::elastic_energy_T
	POLYFEM_TEMPLATE_SPECIALIZE_ELASTIC_ENERGY(MooneyRivlinElasticity)
} // namespace polyfem::assembler