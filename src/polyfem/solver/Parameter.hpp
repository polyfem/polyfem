#pragma once

#include <polyfem/State.hpp>

namespace polyfem
{
	class Parameter
	{
	public:
		Parameter(std::vector<std::shared_ptr<State>> states_ptr) : states_ptr_(states_ptr){};
		virtual ~Parameter() = default;

		virtual void update() = 0;

		virtual void map(const Eigen::MatrixXd &x, Eigen::MatrixXd &q) = 0;

		virtual void smoothing(const Eigen::VectorXd &x, Eigen::VectorXd &new_x) = 0;
		virtual bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) = 0;
		virtual bool is_intersection_free(const Eigen::VectorXd &x) = 0;
		virtual bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) { return true; }
		virtual double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) { return 1; }

		virtual void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) = 0;
		virtual void line_search_end(bool failed) = 0;
		virtual void post_step(const int iter_num, const Eigen::VectorXd &x0) = 0;

		inline virtual void set_optimization_dim(const int optimization_dim) final { optimization_dim_ = optimization_dim; }
		inline virtual int optimization_dim() final { return optimization_dim_; }

		inline virtual std::string name() final { return parameter_name_; }

		virtual bool pre_solve(const Eigen::VectorXd &newX) = 0;
		virtual void post_solve(const Eigen::VectorXd &newX) = 0;

		inline virtual bool remesh(Eigen::VectorXd &x) { return true; };

		virtual Eigen::VectorXd force_inequality_constraint(const Eigen::VectorXd &x0, const Eigen::VectorXd &dx) { return x0 + dx; }
		virtual int n_inequality_constraints() { return 0; }
		virtual double inequality_constraint_val(const Eigen::VectorXd &x, const int index)
		{
			assert(false);
			return std::nan("");
		}
		virtual Eigen::VectorXd inequality_constraint_grad(const Eigen::VectorXd &x, const int index)
		{
			assert(false);
			return Eigen::VectorXd();
		}

		virtual Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const
		{
			Eigen::VectorXd min(x.size());
			min.setConstant(std::numeric_limits<double>::min());
			return min;
		}
		virtual Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const
		{
			Eigen::VectorXd max(x.size());
			max.setConstant(std::numeric_limits<double>::max());
			return max;
		}

	private:
		std::vector<std::shared_ptr<State>> states_ptr_;
		int optimization_dim_;
		const std::string parameter_name_;
	};
} // namespace polyfem