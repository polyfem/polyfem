#pragma once

#include <polyfem/utils/Logger.hpp>

#include <Eigen/Core>

namespace polyfem::solver
{
	/** This parameterize a function f : x -> y
	 * and provides the chain rule with respect to previous gradients
	 */
	class Parameterization
	{
	public:
		virtual ~Parameterization() {}

		virtual Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const
		{
			log_and_throw_error("Not supported");
			return Eigen::VectorXd();
		}

		virtual int size(const int x_size) const = 0; // just for verification
		virtual Eigen::VectorXd eval(const Eigen::VectorXd &x) const = 0;
		virtual Eigen::VectorXd chain_rule(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const = 0;
	};

	class CompositeParameterization
	{
	public:
		virtual ~CompositeParameterization() {}

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) const
		{
			log_and_throw_error("Not supported");
			return Eigen::VectorXd();
		}

		Eigen::VectorXd eval(const Eigen::VectorXd &x) const
		{
			if (parameterizations_.empty())
				return x;

			Eigen::VectorXd y = x;
			for (const auto &p : parameterizations_)
			{
				y = p->eval(y);
			}

			return y;
		}
		Eigen::VectorXd chain_rule(const Eigen::VectorXd &x, const Eigen::VectorXd &grad_full) const
		{
			Eigen::VectorXd gradv = grad_full;

			std::vector<Eigen::VectorXd> ys;
			for (const auto &p : parameterizations_)
			{
				ys.emplace_back(y);
				y = p->eval(y);
			}

			for (int i = parameterizations_.size() - 1; i >= 0; --i)
			{
				gradv = parameterizations_[i]->chain_rule(gradv, ys[i]);
			}
		}
	
	private:
		std::vector<std::shared_ptr<Parameterization>> parameterizations_;
	};
} // namespace polyfem::solver
