#pragma once

#include <polyfem/utils/Logger.hpp>

#include <Eigen/Core>

namespace polyfem::solver
{
	/** This parameterize a function f : x -> y
	 * and provides the chain rule with respect to previous gradients
	 */
	class Parametrization
	{
	public:
		virtual ~Parametrization() {}

		virtual Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const
		{
			log_and_throw_error("Not supported");
			return Eigen::VectorXd();
		}

		virtual int size(const int x_size) const = 0; // just for verification
		virtual Eigen::VectorXd eval(const Eigen::VectorXd &x) const = 0;
		virtual Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const = 0;
	};

	class CompositeParametrization
	{
	public:
		CompositeParametrization() {}
		CompositeParametrization(const std::vector<std::shared_ptr<Parametrization>> &parametrizations): parametrizations_(parametrizations) {}
		virtual ~CompositeParametrization() {}

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const
		{
			log_and_throw_error("Not supported");
			return Eigen::VectorXd();
		}

		Eigen::VectorXd eval(const Eigen::VectorXd &x) const
		{
			if (parametrizations_.empty())
				return x;

			Eigen::VectorXd y = x;
			for (const auto &p : parametrizations_)
			{
				y = p->eval(y);
			}

			return y;
		}
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const
		{
			if (parametrizations_.empty())
				return grad_full;

			std::vector<Eigen::VectorXd> ys;
			auto y = x;
			for (const auto &p : parametrizations_)
			{
				ys.emplace_back(y);
				y = p->eval(y);
			}

			Eigen::VectorXd gradv = grad_full;
			for (int i = parametrizations_.size() - 1; i >= 0; --i)
			{
				gradv = parametrizations_[i]->apply_jacobian(gradv, ys[i]);
			}

			return gradv;
		}
	
	private:
		std::vector<std::shared_ptr<Parametrization>> parametrizations_;
	};
} // namespace polyfem::solver
