#include "SpatialIntegralForms.hpp"

using namespace polyfem::utils;

namespace polyfem::solver
{
	double SpatialIntegralForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		assert(time_step_ < state_.diff_cached.size());
		return AdjointTools::integrate_objective(state_, get_integral_functional(), state_.diff_cached[time_step_].u, ids_, spatial_integral_type_, time_step_);
	}

	void SpatialIntegralForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(time_step_ < state_.diff_cached.size());
		gradv.setZero(x.size());
		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &parametrization = param_map->get_parameterization();
			const auto &state = param_map->get_state();
			const auto &param_type = param_map->get_parameter_type();

			if (&state != &state_)
				continue;

			Eigen::VectorXd term;
			if (param_type == ParameterType::Shape)
				AdjointTools::compute_shape_derivative_functional_term(state_, state_.diff_cached[time_step_].u, get_integral_functional(), ids_, spatial_integral_type_, term, time_step_);
			else if (param_type == ParameterType::MacroStrain)
				AdjointTools::compute_macro_strain_derivative_functional_term(state_, state_.diff_cached[time_step_].u, get_integral_functional(), ids_, spatial_integral_type_, term, time_step_);
      
      if (term.size() > 0)
        gradv += parametrization.apply_jacobian(term, x);
		}
	}

	Eigen::MatrixXd SpatialIntegralForm::compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state)
	{
		if (&state != &state_)
			return Eigen::VectorXd::Zero(state.ndof(), state.diff_cached.size());

		assert(time_step_ < state_.diff_cached.size());

		Eigen::VectorXd rhs;
		AdjointTools::dJ_du_step(state, get_integral_functional(), state.diff_cached[time_step_].u, ids_, spatial_integral_type_, time_step_, rhs);

		return rhs;
	}

	// TODO: call local assemblers instead
	IntegrableFunctional StressForm::get_integral_functional() const
	{
		IntegrableFunctional j;

		const std::string formulation = state_.formulation();
		const int power = in_power_;

		j.set_j([formulation, power](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			Eigen::MatrixXd grad_u_q, stress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				if (formulation == "Laplacian")
				{
					stress = grad_u.row(q);
				}
				else if (formulation == "LinearElasticity")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else if (formulation == "NeoHookean")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
				}
				else
					log_and_throw_error("Unknown formulation!");
				val(q) = pow(stress.squaredNorm(), power / 2.);
			}
		});

		j.set_dj_dgradu([formulation, power](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			const int dim = sqrt(grad_u.cols());
			const int actual_dim = (formulation == "Laplacian") ? 1 : dim;
			Eigen::MatrixXd grad_u_q, stress, stress_dstress;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				if (formulation == "Laplacian")
				{
					stress = grad_u.row(q);
					stress_dstress = 2 * stress;
				}
				else if (formulation == "LinearElasticity")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
					stress_dstress = mu(q) * (stress + stress.transpose()) + lambda(q) * stress.trace() * Eigen::MatrixXd::Identity(stress.rows(), stress.cols());
				}
				else if (formulation == "NeoHookean")
				{
					vector2matrix(grad_u.row(q), grad_u_q);
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
					stress_dstress = mu(q) * stress + FmT * stress.transpose() * FmT * (mu(q) - lambda(q) * std::log(def_grad.determinant())) + (lambda(q) * (FmT.array() * stress.array()).sum()) * FmT;
				}
				else
					log_and_throw_error("Unknown formulation!");

				const double coef = power * pow(stress.squaredNorm(), power / 2. - 1.);
				for (int i = 0; i < actual_dim; i++)
					for (int l = 0; l < dim; l++)
						val(q, i * dim + l) = coef * stress_dstress(i, l);
			}
		});

		return j;
	}

  void StressForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
  {
    SpatialIntegralForm::compute_partial_gradient(x, gradv);
		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &parametrization = param_map->get_parameterization();
			const auto &state = param_map->get_state();
			const auto &param_type = param_map->get_parameter_type();

			if (&state != &state_)
				continue;

			Eigen::VectorXd term;
			if (param_type == ParameterType::Material)
				log_and_throw_error("Doesn't support stress derivative wrt. material!");

      if (term.size() > 0)
        gradv += parametrization.apply_jacobian(term, x);
		}
  }
} // namespace polyfem::solver