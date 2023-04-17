#pragma once

#include "AdjointForm.hpp"

namespace polyfem::solver
{
	class TransientForm : public AdjointForm
	{
	public:
		TransientForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const int time_steps, const double dt, const std::string &transient_integral_type, const std::vector<int> &steps, const std::shared_ptr<StaticForm> &obj) : AdjointForm(variable_to_simulations), time_steps_(time_steps), dt_(dt), transient_integral_type_(transient_integral_type), steps_(steps), obj_(obj) {}
		virtual ~TransientForm() = default;

		Eigen::MatrixXd compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const override;
		void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		void init(const Eigen::VectorXd &x) override;
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		void line_search_end() override;
		void post_step(const int iter_num, const Eigen::VectorXd &x) override;
		void solution_changed(const Eigen::VectorXd &new_x) override;
		void update_quantities(const double t, const Eigen::VectorXd &x) override;
		void init_lagging(const Eigen::VectorXd &x) override;
		void update_lagging(const Eigen::VectorXd &x, const int iter_num) override;
		void set_apply_DBC(const Eigen::VectorXd &x, bool apply_DBC) override;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

	protected:
		std::vector<double> get_transient_quadrature_weights() const;
		double value_unweighted(const Eigen::VectorXd &x) const override;

		int time_steps_;
		double dt_;
		std::string transient_integral_type_;
		std::vector<int> steps_;
		mutable std::shared_ptr<StaticForm> obj_;
	};
}