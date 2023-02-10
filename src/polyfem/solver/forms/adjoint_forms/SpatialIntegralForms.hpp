#pragma once

#include "AdjointForm.hpp"

namespace polyfem::solver
{
	class SpatialIntegralForm : public StaticForm
	{
	public:
		SpatialIntegralForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const State &state, const json &args) : StaticForm(variable_to_simulations, parametrizations), state_(state)
		{
		}

		const State &get_state() { return state_; }

		void set_integral_type(const SpatialIntegralType type) { spatial_integral_type_ = type; }

		double value_unweighted(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd compute_adjoint_rhs_step(const Eigen::VectorXd &x, const State &state) override;
		virtual void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		virtual IntegrableFunctional get_integral_functional() const = 0;

		const State &state_;
		SpatialIntegralType spatial_integral_type_;
		std::set<int> ids_;
	};

	class StressForm : public SpatialIntegralForm
	{
	public:
		StressForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, parametrizations, state, args)
		{
			set_integral_type(SpatialIntegralType::VOLUME);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		int in_power_ = 2;
	};

	class PositionForm : public SpatialIntegralForm
	{
	public:
		PositionForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, parametrizations, state, args)
		{
			set_integral_type(SpatialIntegralType::VOLUME);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		void set_dim(const int dim) { dim_ = dim; }

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		int dim_ = 0;
	};
} // namespace polyfem::solver