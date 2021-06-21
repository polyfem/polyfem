#include <polyfem/ImplicitNewmark.hpp>

namespace polyfem
{

	void ImplicitNewmark::set_parameters(const nlohmann::json &args)
	{
		gamma = args["gamma"];
		beta = args["beta"];
	}

	void ImplicitNewmark::update_quantities(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd tmp = x_prev + dt() * (v_prev + dt() * (0.5 - beta) * a_prev);
		v_prev += dt() * (1 - gamma) * a_prev;	 // vᵗ + h(1 - γ)aᵗ
		a_prev = (x - tmp) / (beta * dt() * dt()); // aᵗ⁺¹ = ...
		v_prev += dt() * gamma * a_prev;		   // hγaᵗ⁺¹
		x_prev = x;
	}

	Eigen::VectorXd ImplicitNewmark::x_tilde() const
	{
		return x_prev + dt() * (v_prev + dt() * (0.5 - beta) * a_prev);
	}

	double ImplicitNewmark::acceleration_scaling() const
	{
		return beta * dt() * dt();
	}

} // namespace polyfem
