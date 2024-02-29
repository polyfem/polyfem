#pragma once

#include <polyfem/solver/forms/adjoint_forms/ParametrizationForm.hpp>
#include "VariableToSimulation.hpp"

namespace polyfem::solver
{
	/// @brief Computes the dot product of the input x (after parametrization) and the volume of each element on the mesh
	class WeightedVolumeForm : public ParametrizationForm
	{
	public:
		WeightedVolumeForm(const CompositeParametrization &parametrizations, const State &state)
			: ParametrizationForm(parametrizations), state_(state)
		{
		}

	protected:
		/// @param x The input vector, after parametrization, same size as the number of elements on the mesh
		double value_unweighted_with_param(const Eigen::VectorXd &x) const override;

		/// @brief Computes the gradient of this form wrt. x, assuming that the volume of elements doesn't depend on x
		void compute_partial_gradient_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	private:
		const State &state_;
	};
} // namespace polyfem::solver