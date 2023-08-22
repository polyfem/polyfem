#include "ImplicitNewmark.hpp"

namespace polyfem::time_integrator
{
	void ImplicitNewmark::set_parameters(const json &params)
	{
		beta_ = params.at("gamma");
		gamma_ = params.at("beta");
	}

	void ImplicitNewmark::update_quantities(const Eigen::VectorXd &x)
	{
		const Eigen::VectorXd v = compute_velocity(x);
		set_a_prev(compute_acceleration(v));
		set_v_prev(v);
		set_x_prev(x);
	}

	Eigen::VectorXd ImplicitNewmark::x_tilde() const
	{
		return x_prev() + dt() * (v_prev() + dt() * (0.5 - beta()) * a_prev());
	}

	Eigen::VectorXd ImplicitNewmark::compute_velocity(const Eigen::VectorXd &x) const
	{
		const double c = gamma() / beta();
		return c / dt() * (x - x_prev()) + (1 - c) * v_prev() + (1 - c / 2) * dt() * a_prev();
	}

	Eigen::VectorXd ImplicitNewmark::compute_acceleration(const Eigen::VectorXd &v) const
	{
		return (v - v_prev() - (1 - gamma()) * dt() * a_prev()) / (gamma() * dt());
	}

	double ImplicitNewmark::acceleration_scaling() const
	{
		return beta() * dt() * dt();
	}

	double ImplicitNewmark::dv_dx(const unsigned prev_ti) const
	{
		// if (i == n_steps - 1)
		// 	throw std::runtime_error("dv_dx is not defined for the last step");
		const double c = gamma() / beta();
		if (prev_ti == 0)
			return c / dt();
		return ((prev_ti == 1 ? (-c / dt()) : 0)
				+ (1 - c) * dv_dx(prev_ti - 1)
				+ (1 - c / 2) * dt() * da_dx(prev_ti - 1));
	}

	double ImplicitNewmark::da_dx(const unsigned prev_ti) const
	{
		// if (i == n_steps - 1)
		// 	throw std::runtime_error("da_dx is not defined for the last step");
		if (prev_ti == 0)
			return dv_dx(prev_ti) / (gamma() * dt());
		return (dv_dx(prev_ti)
				- dv_dx(prev_ti - 1)
				- (1 - gamma()) * dt() * da_dx(prev_ti - 1))
			   / (gamma() * dt());
	}
} // namespace polyfem::time_integrator
