#pragma once

#include "AdjointForm.hpp"

namespace polyfem::solver
{
	class TransientForm : public AdjointForm
	{
	public:
		TransientForm(const VariableToSimulationGroup &variable_to_simulations, const int time_steps, const double dt, const std::string &transient_integral_type, const std::vector<int> &steps, const std::shared_ptr<StaticForm> &obj) : AdjointForm(variable_to_simulations), time_steps_(time_steps), dt_(dt), steps_(steps), obj_(obj), transient_integral_type_(transient_integral_type) {}
		virtual ~TransientForm() = default;

		virtual Eigen::MatrixXd compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const override;
		virtual void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		void init(const Eigen::VectorXd &x) override;
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		void line_search_end() override;
		void post_step(const polysolve::nonlinear::PostStepData &data) override;
		void solution_changed(const Eigen::VectorXd &new_x) override;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

	protected:
		virtual double value_unweighted(const Eigen::VectorXd &x) const override;

		int time_steps_;
		double dt_;
		std::vector<int> steps_;
		std::shared_ptr<StaticForm> obj_;

	private:
		std::vector<double> get_transient_quadrature_weights() const;
		std::string transient_integral_type_;
	};

	class ProxyTransientForm : public TransientForm
	{
	public:
		ProxyTransientForm(const VariableToSimulationGroup &variable_to_simulations, const int time_steps, const double dt, const std::string &transient_integral_type, const std::vector<int> &steps, const std::shared_ptr<StaticForm> &obj) : TransientForm(variable_to_simulations, time_steps, dt, transient_integral_type, steps, obj) {}
		virtual ~ProxyTransientForm() = default;

		Eigen::MatrixXd compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const override;
		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;

	private:
		double eval(const Eigen::VectorXd &y) const;
		Eigen::VectorXd eval_grad(const Eigen::VectorXd &y) const;
	};
} // namespace polyfem::solver