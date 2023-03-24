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
		Parametrization() {}
		virtual ~Parametrization() {}

		virtual Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y)
		{
			log_and_throw_error("Not supported");
			return Eigen::VectorXd();
		}

		virtual int size(const int x_size) const = 0; // just for verification
		virtual Eigen::VectorXd eval(const Eigen::VectorXd &x) const = 0;
		virtual Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const = 0;
	};

	class IndexedParametrization : public Parametrization
	{
	public:
		IndexedParametrization() {}
		virtual ~IndexedParametrization() {}

		void set_output_indexing(const Eigen::VectorXi &output_indexing) { output_indexing_ = output_indexing; }
		Eigen::VectorXi get_output_indexing(const Eigen::VectorXd &x) const
		{
			const int out_size = size(x.size());
			if (output_indexing_.size() == out_size)
				return output_indexing_;
			else if (output_indexing_.size() == 0)
			{
				Eigen::VectorXi ind;
				ind.setLinSpaced(out_size, 0, out_size - 1);
				return ind;
			}
			else
				log_and_throw_error(fmt::format("Indexing size and output size of the Parametrization do not match! {} vs {}", output_indexing_.size(), out_size));
			return Eigen::VectorXi();
		}

	protected:
		Eigen::VectorXi output_indexing_;
	};

	class CompositeParametrization : public IndexedParametrization
	{
	public:
		CompositeParametrization() {}
		CompositeParametrization(const std::vector<std::shared_ptr<Parametrization>> &parametrizations) : parametrizations_(parametrizations) {}
		virtual ~CompositeParametrization() {}

		int size(const int x_size) const override
		{
			int cur_size = x_size;
			for (const auto &p : parametrizations_)
				cur_size = p->size(cur_size);

			return cur_size;
		}

		Eigen::VectorXd inverse_eval(const Eigen::VectorXd &y) override
		{
			if (parametrizations_.empty())
				return y;

			Eigen::VectorXd x = y;
			for (int i = parametrizations_.size() - 1; i >= 0; i--)
			{
				x = parametrizations_[i]->inverse_eval(x);
			}

			return x;
		}

		Eigen::VectorXd eval(const Eigen::VectorXd &x) const override
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
		Eigen::VectorXd apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const override
		{
			Eigen::VectorXd gradv = grad_full(get_output_indexing(x));

			if (parametrizations_.empty())
				return gradv;

			std::vector<Eigen::VectorXd> ys;
			auto y = x;
			for (const auto &p : parametrizations_)
			{
				ys.emplace_back(y);
				y = p->eval(y);
			}

			for (int i = parametrizations_.size() - 1; i >= 0; --i)
				gradv = parametrizations_[i]->apply_jacobian(gradv, ys[i]);

			return gradv;
		}

	private:
		std::vector<std::shared_ptr<Parametrization>> parametrizations_;
	};
} // namespace polyfem::solver