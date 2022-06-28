#include "ImplicitEuler.hpp"

namespace polyfem::time_integrator
{
	void ImplicitEuler::update_quantities(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd v = (x - x_prev()) / dt();
		a_prev() = (v - v_prev()) / dt();
		v_prev() = v;
		x_prev() = x;
	}

	Eigen::VectorXd ImplicitEuler::x_tilde() const
	{
		return x_prev() + dt() * v_prev();
	}

	double ImplicitEuler::acceleration_scaling() const
	{
		return dt() * dt();
	}
} // namespace polyfem::time_integrator
