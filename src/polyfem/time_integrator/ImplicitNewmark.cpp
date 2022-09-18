#include "ImplicitNewmark.hpp"

namespace polyfem::time_integrator
{
	void ImplicitNewmark::set_parameters(const nlohmann::json &params)
	{
		m_gamma = params.at("gamma");
		m_beta = params.at("beta");
	}

	void ImplicitNewmark::update_quantities(const Eigen::VectorXd &x)
	{
		const Eigen::VectorXd v = compute_velocity(x);
		a_prev() = compute_acceleration(v);
		v_prev() = v;
		x_prev() = x;
	}

	Eigen::VectorXd ImplicitNewmark::x_tilde() const
	{
		return x_prev() + dt() * (v_prev() + dt() * (0.5 - beta()) * a_prev());
	}

	Eigen::VectorXd ImplicitNewmark::compute_velocity(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd tmp = x_prev() + dt() * (v_prev() + dt() * (0.5 - beta()) * a_prev());
		const Eigen::VectorXd a = (x - tmp) / (beta() * dt() * dt());
		return v_prev() + dt() * ((1 - gamma()) * a_prev() + gamma() * a);
	}

	Eigen::VectorXd ImplicitNewmark::compute_acceleration(const Eigen::VectorXd &v) const
	{
		return (v - v_prev() - (1 - gamma()) * dt() * a_prev()) / (gamma() * dt());
	}

	double ImplicitNewmark::acceleration_scaling() const
	{
		return beta() * dt() * dt();
	}
} // namespace polyfem::time_integrator
