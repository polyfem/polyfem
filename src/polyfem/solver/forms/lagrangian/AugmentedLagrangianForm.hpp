#pragma once

#include <polyfem/solver/forms/Form.hpp>

namespace polyfem::solver
{
	/// @brief Form of the augmented lagrangian
	class AugmentedLagrangianForm : public Form
	{

	public:
		AugmentedLagrangianForm(const std::vector<int> &constraint_nodes) : constraint_nodes_(constraint_nodes) {}
		virtual ~AugmentedLagrangianForm() {}

		virtual void update_lagrangian(const Eigen::VectorXd &x, const double k_al) = 0;

		virtual double compute_error(const Eigen::VectorXd &x) const = 0;
		virtual Eigen::VectorXd target(const Eigen::VectorXd &x) const { return Eigen::VectorXd{}; };

		inline void set_initial_weight(const double k_al) { k_al_ = k_al; }

		inline const std::vector<int> &constraint_nodes() const { return constraint_nodes_; }

	protected:
		Eigen::VectorXd lagr_mults_;              ///< vector of lagrange multipliers
		double k_al_;                             ///< penalty parameter
		const std::vector<int> constraint_nodes_; ///< constraint nodes
	};
} // namespace polyfem::solver
