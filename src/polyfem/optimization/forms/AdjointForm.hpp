#pragma once

#include <polyfem/solver/forms/Form.hpp>
#include "VariableToSimulation.hpp"

namespace polyfem::solver
{
	class AdjointForm : public Form
	{
	public:
		AdjointForm(const VariableToSimulationGroup &variable_to_simulations) : variable_to_simulations_(variable_to_simulations) {}
		virtual ~AdjointForm() {}

		virtual std::string name() const override { return "adjoint"; }
		void enable_energy_print(const std::string &print_energy_keyword);

		double value(const Eigen::VectorXd &x) const override;
		virtual void solution_changed(const Eigen::VectorXd &new_x) override;

		const VariableToSimulationGroup &get_variable_to_simulations() const { return variable_to_simulations_; }

		virtual Eigen::MatrixXd compute_reduced_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const;
		virtual void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const;
		virtual void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const final override;
		virtual Eigen::MatrixXd compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const;

		// not used functions from base class
		virtual void update_quantities(const double t, const Eigen::VectorXd &x) final override;
		virtual void init_lagging(const Eigen::VectorXd &x) final override;
		virtual void update_lagging(const Eigen::VectorXd &x, const int iter_num) final override;

	protected:
		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const final override;
		virtual void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const final override;

		const VariableToSimulationGroup variable_to_simulations_;

		enum class PrintStage
		{
			Inactive,
			AlreadyPrinted,
			ToPrint
		};

		mutable PrintStage print_energy_ = PrintStage::Inactive;
		std::string print_energy_keyword_;
	};

	class StaticForm : public AdjointForm
	{
	public:
		using AdjointForm::AdjointForm;
		virtual ~StaticForm() = default;

		virtual std::string name() const override { return "static"; }

		Eigen::MatrixXd compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const final override;
		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const final override;
		double value_unweighted(const Eigen::VectorXd &x) const final override;

		virtual Eigen::VectorXd compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const = 0;
		virtual Eigen::VectorXd compute_adjoint_rhs_step_prev(const int time_step, const Eigen::VectorXd &x, const State &state) const;
		virtual void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const = 0;
		virtual double value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const = 0;
		virtual void solution_changed(const Eigen::VectorXd &new_x) final override;
		virtual void solution_changed_step(const int time_step, const Eigen::VectorXd &new_x) {}
		virtual bool depends_on_step_prev() const final { return depends_on_step_prev_; }

	protected:
		bool depends_on_step_prev_ = false;
	};

	class MaxStressForm : public StaticForm
	{
	public:
		MaxStressForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : StaticForm(variable_to_simulations), state_(state)
		{
			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		std::string name() const override { return "max_stress"; }

		Eigen::VectorXd compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const override;
		double value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const override;
		void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	private:
		std::set<int> interested_ids_;
		const State &state_;
	};
} // namespace polyfem::solver