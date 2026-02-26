#include "WeightedVolumeForm.hpp"
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	double WeightedVolumeForm::value_unweighted_with_param(const Eigen::VectorXd &x) const
	{
		assert(x.size() == state_.mesh->n_elements());

		double val = 0;
		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < state_.bases.size(); e++)
		{
			state_.ass_vals_cache.compute(e, state_.mesh->is_volume(), state_.bases[e], state_.geom_bases()[e], vals);
			val += (vals.det.array() * vals.quadrature.weights.array()).sum() * x(e);
		}
		return val;
	}

	void WeightedVolumeForm::compute_partial_gradient_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(x.size() == state_.mesh->n_elements());

		gradv.setZero(x.size());
		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < state_.bases.size(); e++)
		{
			state_.ass_vals_cache.compute(e, state_.mesh->is_volume(), state_.bases[e], state_.geom_bases()[e], vals);
			gradv(e) = (vals.det.array() * vals.quadrature.weights.array()).sum();
		}
		gradv *= weight();
	}
} // namespace polyfem::solver
