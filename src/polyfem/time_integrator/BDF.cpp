#include "BDF.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::time_integrator
{
	BDF::BDF(const int order)
	{
		if (order < 1 || order > 6)
			log_and_throw_error("BDF order must be 1 ≤ n ≤ 6");
		max_steps_ = order;
	}

	void BDF::set_parameters(const json &params)
	{
		max_steps_ = params.at("steps");
		if (max_steps_ < 1 || max_steps_ > 6)
			log_and_throw_error("BDF steps must be 1 ≤ n ≤ 6");
	}

	const std::vector<double> &BDF::alphas(const int i)
	{
		static const std::array<std::vector<double>, 6> _alphas = {{
			{1},
			{4.0 / 3.0, -1.0 / 3.0},
			{18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0},
			{48.0 / 25.0, -36.0 / 25.0, 16.0 / 25.0, -3.0 / 25.0},
			{300.0 / 137.0, -300.0 / 137.0, 200.0 / 137.0, -75.0 / 137.0, 12.0 / 137.0},
			{360.0 / 147.0, -450.0 / 147.0, 400.0 / 147.0, -225.0 / 147.0, 72.0 / 147.0, -10.0 / 147.0},
		}};
		assert(i >= 0 && i < _alphas.size());
		return _alphas[i];
	}

	double BDF::betas(const int i)
	{
		static const std::array<double, 6> _betas = {{
			1.0,
			2.0 / 3.0,
			6.0 / 11.0,
			12.0 / 25.0,
			60.0 / 137.0,
			60.0 / 147.0,
		}};
		assert(i >= 0 && i < _betas.size());
		return _betas[i];
	}

	Eigen::VectorXd BDF::weighted_sum_x_prevs() const
	{
		const std::vector<double> &alpha = alphas(steps() - 1);

		Eigen::VectorXd sum = Eigen::VectorXd::Zero(x_prev().size());
		for (int i = 0; i < steps(); i++)
		{
			sum += alpha[i] * x_prevs_[i];
		}

		return sum;
	}

	Eigen::VectorXd BDF::weighted_sum_v_prevs() const
	{
		const std::vector<double> &alpha = alphas(steps() - 1);

		Eigen::VectorXd sum = Eigen::VectorXd::Zero(v_prev().size());
		for (int i = 0; i < steps(); i++)
		{
			sum += alpha[i] * v_prevs_[i];
		}

		return sum;
	}

	void BDF::update_quantities(const Eigen::VectorXd &x)
	{
		const Eigen::VectorXd v = compute_velocity(x);
		const Eigen::VectorXd a = compute_acceleration(v);

		x_prevs_.push_front(x);
		v_prevs_.push_front(v);
		a_prevs_.push_front(a);

		if (steps() > max_steps())
		{
			x_prevs_.pop_back();
			v_prevs_.pop_back();
			a_prevs_.pop_back();
		}
		assert(x_prevs_.size() <= max_steps());
		assert(x_prevs_.size() == v_prevs_.size());
		assert(x_prevs_.size() == a_prevs_.size());
	}

	Eigen::VectorXd BDF::x_tilde() const
	{
		return weighted_sum_x_prevs() + betas(steps() - 1) * dt() * weighted_sum_v_prevs();
	}

	Eigen::VectorXd BDF::compute_velocity(const Eigen::VectorXd &x) const
	{
		return (x - weighted_sum_x_prevs()) / beta_dt();
	}

	Eigen::VectorXd BDF::compute_acceleration(const Eigen::VectorXd &v) const
	{
		return (v - weighted_sum_v_prevs()) / beta_dt();
	}

	double BDF::acceleration_scaling() const
	{
		const double beta = betas(steps() - 1);
		return beta * beta * dt() * dt();
	}

	double BDF::dv_dx(const unsigned i) const
	{
		if (i == 0)
			return 1 / beta_dt();
		if (i >= steps())
			return 0;
		return -alphas(steps() - 1)[i] / beta_dt();
	}

	double BDF::beta_dt() const
	{
		return betas(steps() - 1) * dt();
	}
} // namespace polyfem::time_integrator
