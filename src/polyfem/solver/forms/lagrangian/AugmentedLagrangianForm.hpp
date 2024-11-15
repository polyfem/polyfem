#pragma once

#include <polyfem/solver/forms/Form.hpp>

namespace polyfem::solver
{
	/// @brief Form of the augmented lagrangian
	class AugmentedLagrangianForm : public Form
	{

	public:
		virtual void update_lagrangian(const Eigen::VectorXd &x, const double k_al) = 0;

		virtual double compute_error(const Eigen::VectorXd &x) const = 0;
		virtual Eigen::VectorXd target(const Eigen::VectorXd &x) const { return Eigen::VectorXd{}; };

		inline void set_initial_weight(const double k_al) { k_al_ = k_al; }

	protected:
		Eigen::VectorXd lagr_mults_; ///< vector of lagrange multipliers
		double k_al_;                ///< penalty parameter
	};
} // namespace polyfem::solver
