#pragma once

#include <polyfem/solver/forms/Form.hpp>

namespace polyfem::solver
{
	/// @brief Form of the lagrangian in augmented lagrangian
	class LagrangianForm : public Form
	{

	public:
		virtual void update_lagrangian(const Eigen::VectorXd &x, const double k_al) = 0;

	protected:
		Eigen::VectorXd lagr_mults_; ///< vector of lagrange multipliers
	};
} // namespace polyfem::solver
