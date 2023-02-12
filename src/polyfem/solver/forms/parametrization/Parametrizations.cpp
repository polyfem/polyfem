#include "Parametrizations.hpp"

#include <polyfem/utils/ElasticityUtils.hpp>

namespace polyfem::solver
{
	std::vector<std::shared_ptr<Parametrization>> ParametrizationFactory::build(const json &params, const int full_size)
	{
		return std::vector<std::shared_ptr<Parametrization>>();
	}

	ExponentialMap::ExponentialMap(const int from, const int to)
		: from_(from), to_(to)
	{
		assert(from_ < to_);
	}

	Eigen::VectorXd ExponentialMap::inverse_eval(const Eigen::VectorXd &y) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = y;
			res.segment(from_, to_ - from_) = y.segment(from_, to_ - from_).array().log();
			return res;
		}
		else
			return y.array().log();
	}

	Eigen::VectorXd ExponentialMap::eval(const Eigen::VectorXd &x) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = x;
			res.segment(from_, to_ - from_) = x.segment(from_, to_ - from_).array().exp();
			return res;
		}
		else
			return x.array().exp();
	}

	Eigen::VectorXd ExponentialMap::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = grad.array();
			res.segment(from_, to_ - from_) = x.segment(from_, to_ - from_).array().exp() * grad.segment(from_, to_ - from_).array();
			return res;
		}
		else
			return x.array().exp() * grad.array();
	}


	Eigen::VectorXd PowerMap::inverse_eval(const Eigen::VectorXd &y) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = y;
			res.segment(from_, to_ - from_) = y.segment(from_, to_ - from_).array().pow(1. / power_);
			return res;
		}
		else
			return y.array().pow(1. / power_);
	}

	Eigen::VectorXd PowerMap::eval(const Eigen::VectorXd &x) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = x;
			res.segment(from_, to_ - from_) = x.segment(from_, to_ - from_).array().pow(power_);
			return res;
		}
		else
			return x.array().pow(power_);
	}

	Eigen::VectorXd PowerMap::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		if (from_ >= 0)
		{
			Eigen::VectorXd res = grad;
			res.segment(from_, to_ - from_) = grad.segment(from_, to_ - from_).array() * x.segment(from_, to_ - from_).array().pow(power_ - 1) * power_;
			return res;
		}
		else
			return grad.array() * x.array().pow(power_ - 1) * power_;
	}

	ENu2LambdaMu::ENu2LambdaMu(const bool is_volume)
		: is_volume_(is_volume)
	{
	}

	Eigen::VectorXd ENu2LambdaMu::inverse_eval(const Eigen::VectorXd &y) const
	{
		const int size = y.size() / 2;
		assert(size * 2 == y.size());

		Eigen::VectorXd x(y.size());
		for (int i = 0; i < size; i++)
		{
			x(i) = convert_to_E(is_volume_, y(i), y(i + size));
			x(i + size) = convert_to_nu(is_volume_, y(i), y(i + size));
		}

		return x;
	}

	Eigen::VectorXd ENu2LambdaMu::eval(const Eigen::VectorXd &x) const
	{
		const int size = x.size() / 2;
		assert(size * 2 == x.size());

		Eigen::VectorXd y(x.size());
		for (int i = 0; i < size; i++)
		{
			y(i) = convert_to_lambda(is_volume_, x(i), x(i + size));
			y(i + size) = convert_to_mu(x(i), x(i + size));
		}

		return y;
	}

	Eigen::VectorXd ENu2LambdaMu::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		const int size = grad.size() / 2;
		assert(size * 2 == grad.size());
		assert(size * 2 == x.size());

		Eigen::VectorXd grad_E_nu;
		grad_E_nu.setZero(grad.size());
		for (int i = 0; i < size; i++)
		{
			const Eigen::Matrix2d jacobian = d_lambda_mu_d_E_nu(is_volume_, x(i), x(i + size));
			grad_E_nu(i) = grad(i) * jacobian(0, 0) + grad(i + size) * jacobian(1, 0);
			grad_E_nu(i + size) = grad(i) * jacobian(0, 1) + grad(i + size) * jacobian(1, 1);
		}

		return grad_E_nu;
	}

	Eigen::VectorXd PerBody::eval(const Eigen::VectorXd &x) const
	{
		assert(x.size() == reduced_size_);
		Eigen::VectorXd y(full_size_);

		for (int e = 0; e < n_elem_; e++)
		{
			const int body_id = mesh_.get_body_id(e);
			const auto &entry = body_id_map_.at(body_id);
			y(e) = x(entry[1]);                                 // lambda or E
			y(e + n_elem_) = x(entry[1] + body_id_map_.size()); // mu or nu
		}

		return y;
	}

	Eigen::VectorXd PerBody::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		assert(grad.size() == full_size_);
		Eigen::VectorXd grad_body;
		grad_body.setZero(reduced_size_);

		for (int e = 0; e < n_elem_; e++)
		{
			const int body_id = mesh_.get_body_id(e);
			const auto &entry = body_id_map_.at(body_id);
			grad_body(entry[1]) += grad(e);
			grad_body(entry[1] + body_id_map_.size()) += grad(e + n_elem_);
		}

		return grad_body;
	}

	Eigen::VectorXd PerBody::inverse_eval(const Eigen::VectorXd &y) const
	{
		assert(y.size() == full_size_);
		Eigen::VectorXd x(reduced_size_);

		for (auto i : body_id_map_)
		{
			x(i.second[1]) = y(i.second[0]);
			x(i.second[1] + body_id_map_.size()) = y(i.second[0] + n_elem_);
		}

		return x;
	}

	AppendConstantMap::AppendConstantMap(const int size, const double val): size_(size), val_(val)
	{
		if (size_ <= 0)
			log_and_throw_error("Invalid AppendConstantMap input!");
	}

	int AppendConstantMap::size(const int x_size) const 
	{ 
		return x_size + size_; 
	}

	Eigen::VectorXd AppendConstantMap::inverse_eval(const Eigen::VectorXd &y) const
	{
		return y.head(y.size() - size_);
	}

	Eigen::VectorXd AppendConstantMap::eval(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd y(size(x.size()));
		y << x, Eigen::VectorXd::Ones(size_) * val_;

		return y;
	}
	Eigen::VectorXd AppendConstantMap::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
	{
		return grad.head(grad.size() - size_);
	}
} // namespace polyfem::solver
