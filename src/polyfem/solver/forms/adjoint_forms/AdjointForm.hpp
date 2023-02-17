#pragma once

#include <polyfem/solver/forms/ParametrizationForm.hpp>
#include "VariableToSimulation.hpp"

namespace polyfem::solver
{
	class AdjointForm : public ParametrizationForm
	{
	public:
		AdjointForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations) : ParametrizationForm(parametrizations), variable_to_simulations_(variable_to_simulations) {}
		virtual ~AdjointForm() {}

		virtual Eigen::MatrixXd compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) override;
		virtual void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const final override;

		std::vector<std::shared_ptr<VariableToSimulation>> variable_to_simulations_;
	};

	class StaticForm : public AdjointForm
	{
	public:
		using AdjointForm::AdjointForm;
		virtual ~StaticForm() = default;

		virtual void set_time_step(int time_step) { time_step_ = time_step; }
		int get_time_step() const { return time_step_; }

		virtual Eigen::MatrixXd compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) final override;
		virtual Eigen::VectorXd compute_adjoint_rhs_unweighted_step(const Eigen::VectorXd &x, const State &state) = 0;
		virtual double value_unweighted(const Eigen::VectorXd &x) const = 0;

	protected:
		int time_step_ = 0; // time step to integrate
	};

	class TransientForm : public AdjointForm
	{
	public:
		TransientForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const int time_steps, const double dt, const std::string &transient_integral_type, const std::shared_ptr<StaticForm> &obj) : AdjointForm(variable_to_simulations, parametrizations), time_steps_(time_steps), dt_(dt), transient_integral_type_(transient_integral_type), obj_(obj) {}
		virtual ~TransientForm() = default;

		Eigen::MatrixXd compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) override;
		void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		std::vector<double> get_transient_quadrature_weights() const;
		double value_unweighted(const Eigen::VectorXd &x) const override;

		int time_steps_;
		double dt_;
		std::string transient_integral_type_;
		std::shared_ptr<StaticForm> obj_;
	};

	class NodeTargetForm : public StaticForm
	{
	public:
		NodeTargetForm(const State &state, const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const json &args);
		NodeTargetForm(const State &state, const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const std::vector<int> &active_nodes_, const Eigen::MatrixXd &target_vertex_positions_);
		~NodeTargetForm() = default;

		Eigen::VectorXd compute_adjoint_rhs_unweighted_step(const Eigen::VectorXd &x, const State &state) override;
		double value_unweighted(const Eigen::VectorXd &x) const override;
		void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
	
	protected:
		const State &state_;

		Eigen::MatrixXd target_vertex_positions;
		std::vector<int> active_nodes;
	};
} // namespace polyfem::solver