#include "ImplicitEuler.hpp"

namespace polyfem::time_integrator
{
	void ImplicitEuler::update_quantities(const Eigen::VectorXd &x)
	{
		const Eigen::VectorXd v = compute_velocity(x);
		set_a_prev(compute_acceleration(v));
		set_v_prev(v);
		set_x_prev(x);
	}

	Eigen::VectorXd ImplicitEuler::x_tilde() const
	{
		return x_prev() + dt() * v_prev();
	}

	Eigen::VectorXd ImplicitEuler::compute_velocity(const Eigen::VectorXd &x) const
	{
		return (x - x_prev()) / dt();
	}

	Eigen::VectorXd ImplicitEuler::compute_acceleration(const Eigen::VectorXd &v) const
	{
		return (v - v_prev()) / dt();
	}

	double ImplicitEuler::acceleration_scaling() const
	{
		return dt() * dt();
	}

	double ImplicitEuler::dv_dx(const unsigned prev_ti) const
	{
		if (prev_ti > 1)
			return 0;
		return (prev_ti == 0 ? 1 : -1) / dt();
	}
} // namespace polyfem::time_integrator
