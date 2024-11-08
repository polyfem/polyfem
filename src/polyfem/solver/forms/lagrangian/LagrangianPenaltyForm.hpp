#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Form of the penalty in augmented lagrangian
	class LagrangianPenaltyForm : public Form
	{

	public:
		// StiffnessMatrix &mask() { return mask_; }
		// const StiffnessMatrix &mask() const { return mask_; }
		// Eigen::VectorXd target() const { return target_x_; }

		virtual double compute_error(const Eigen::VectorXd &x) const = 0;

	protected:
		// StiffnessMatrix mask_; ///< identity matrix masked by the AL dofs
	};
} // namespace polyfem::solver
