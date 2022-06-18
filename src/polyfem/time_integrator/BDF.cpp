#include "BDF.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::time_integrator
{
	using namespace utils;
	void BDFTimeIntegrator::set_parameters(const nlohmann::json &params)
	{
		num_steps = params.value("num_steps", 1);
		if (num_steps < 1 || num_steps > 6)
		{
			logger().warn("BDFTimeIntegrator num_steps must be 1 ≤ n ≤ 6}; using default of 1");
			num_steps = 1;
		}
	}

	void BDFTimeIntegrator::init(const std::vector<Eigen::VectorXd> &x_prevs,
								 const std::vector<Eigen::VectorXd> &v_prevs,
								 const std::vector<Eigen::VectorXd> &a_prevs,
								 double dt)
	{
		assert(x_prevs.size() > 0 && x_prevs.size() <= 6);
		assert(x_prevs.size() == v_prevs.size());
		assert(x_prevs.size() == a_prevs.size());

		this->x_prevs.clear();
		this->v_prevs.clear();
		this->a_prevs.clear();

		const int n = std::min(int(x_prevs.size()), num_steps);
		for (int i = 0; i < n; i++)
		{
			this->x_prevs.push_back(x_prevs[i]);
			this->v_prevs.push_back(v_prevs[i]);
			this->a_prevs.push_back(a_prevs[i]);
		}
	}

	const std::vector<double> &BDFTimeIntegrator::alphas(const int i)
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

	double BDFTimeIntegrator::betas(const int i)
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

	Eigen::VectorXd BDFTimeIntegrator::weighted_sum_x_prevs() const
	{
		const std::vector<double> &alpha = alphas(x_prevs.size() - 1);

		Eigen::VectorXd sum = Eigen::VectorXd::Zero(x_prev().size());
		for (int i = 0; i < x_prevs.size(); i++)
		{
			sum += alpha[i] * x_prevs[i];
		}

		return sum;
	}

	Eigen::VectorXd BDFTimeIntegrator::weighted_sum_v_prevs() const
	{
		const std::vector<double> &alpha = alphas(v_prevs.size() - 1);

		Eigen::VectorXd sum = Eigen::VectorXd::Zero(v_prev().size());
		for (int i = 0; i < v_prevs.size(); i++)
		{
			sum += alpha[i] * v_prevs[i];
		}

		return sum;
	}

	void BDFTimeIntegrator::update_quantities(const Eigen::VectorXd &x)
	{
		const Eigen::VectorXd sum_x_prev = weighted_sum_x_prevs();
		const Eigen::VectorXd sum_v_prev = weighted_sum_v_prevs();
		const double beta = betas(x_prevs.size() - 1);

		x_prevs.push_front(x); // x_prev() = x
		v_prevs.push_front((x_prev() - sum_x_prev) / (beta * dt()));
		a_prevs.push_front((v_prev() - sum_v_prev) / (beta * dt()));

		if (x_prevs.size() > num_steps)
		{
			x_prevs.pop_back();
			v_prevs.pop_back();
			a_prevs.pop_back();
		}
		assert(x_prevs.size() <= num_steps);
		assert(x_prevs.size() == v_prevs.size());
		assert(x_prevs.size() == a_prevs.size());
	}

	Eigen::VectorXd BDFTimeIntegrator::x_tilde() const
	{
		const double beta = betas(x_prevs.size() - 1);
		return weighted_sum_x_prevs() + beta * dt() * weighted_sum_v_prevs();
	}

	double BDFTimeIntegrator::acceleration_scaling() const
	{
		const double beta = betas(x_prevs.size() - 1);
		return beta * beta * dt() * dt();
	}

	double BDFTimeIntegrator::beta_dt() const
	{
		const double beta = betas(x_prevs.size() - 1);
		return beta * dt();
	}
} // namespace polyfem::time_integrator
