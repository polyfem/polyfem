#pragma once

#include <polyfem/solver/forms/adjoint_forms/ParametrizationForm.hpp>
#include "VariableToSimulation.hpp"

namespace polyfem::solver
{
	class WeightedVolumeForm : public ParametrizationForm
	{
	public:
		WeightedVolumeForm(const CompositeParametrization &parametrizations, const State &state) : ParametrizationForm(parametrizations), state_(state)
		{
		}

		inline double value_unweighted_with_param(const Eigen::VectorXd &x) const override
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

		inline void first_derivative_unweighted_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
			assert(x.size() == state_.mesh->n_elements());

			gradv.setZero(x.size());
			assembler::ElementAssemblyValues vals;
			for (int e = 0; e < state_.bases.size(); e++)
			{
				state_.ass_vals_cache.compute(e, state_.mesh->is_volume(), state_.bases[e], state_.geom_bases()[e], vals);
				gradv(e) = (vals.det.array() * vals.quadrature.weights.array()).sum();
			}
		}

	private:
		const State &state_;
	};
} // namespace polyfem::solver