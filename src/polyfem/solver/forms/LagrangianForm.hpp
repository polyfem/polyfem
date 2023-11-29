#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Form of the lagrangian in augmented lagrangian
	class LagrangianForm : public Form
	{
	public:
		/// @brief Construct a new LagrangianForm object
		LagrangianForm() {}

	public:
		virtual void update_lagrangian(const Eigen::VectorXd &x, const double k_al) = 0;
	};
} // namespace polyfem::solver
