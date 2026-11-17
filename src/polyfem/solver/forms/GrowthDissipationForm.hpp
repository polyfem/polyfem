#pragma once

#include "Form.hpp"

namespace polyfem::solver
{
	/// Rate-independent dissipation of the growth law, on the growth block of
	/// the stacked (displacement, growth) system:
	///
	///     D(theta) = sum_a w_a (theta_a - theta_n,a),
	///     w_a      = sum_e  vn_e phi_e  int_e N_a dV,
	///
	/// with phi_e the per-element threshold (Pa), vn_e the prescribed normal
	/// growth, N_a the growth shape functions and theta_n the converged growth of
	/// the previous increment (data). The vn_e factor makes stationarity read
	/// f_I = phi_e with f_I = -(1/vn) d(J^g psi_e)/d theta, the driving force
	/// per the formulation's convention, so a threshold field computed as f_I
	/// (V0b, route B) enters unchanged.
	///
	/// Linear in theta: constant gradient w, zero Hessian, so it never changes
	/// convexity or symmetry. Irreversibility (theta >= theta_n) is NOT imposed
	/// here: it is the hard constraint of the active-set loop in
	/// GrowthElasticVarForm (pinned DOFs are Dirichlet at theta_n). This form's
	/// max_step_size is only the positivity filter for det F^g > 0
	/// (theta >= floor_fraction * theta_n), which the assembler requires.
	class GrowthDissipationForm : public Form
	{
	public:
		/// @param weights      nodal weights w_a (energy per unit growth), size n_growth
		/// @param theta_n      previous converged growth, size n_growth
		/// @param floor_fraction  positivity filter: iterates keep theta >= floor_fraction * theta_n
		GrowthDissipationForm(const Eigen::VectorXd &weights,
							  const Eigen::VectorXd &theta_n,
							  const double floor_fraction);

		std::string name() const override { return "growth-dissipation"; }

		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		const Eigen::VectorXd &weights() const { return weights_; }
		const Eigen::VectorXd &theta_n() const { return theta_n_; }

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		Eigen::VectorXd weights_;
		Eigen::VectorXd theta_n_;
		Eigen::VectorXd floor_;
	};
} // namespace polyfem::solver
