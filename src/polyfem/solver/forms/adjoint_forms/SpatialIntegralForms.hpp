#pragma once

#include "AdjointForm.hpp"

namespace polyfem::solver
{
	class SpatialIntegralForm : public StaticForm
	{
	public:
		SpatialIntegralForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : StaticForm(variable_to_simulations), state_(state)
		{
		}

		const State &get_state() { return state_; }

		void set_integral_type(const SpatialIntegralType type) { spatial_integral_type_ = type; }

		Eigen::VectorXd compute_adjoint_rhs_unweighted_step(const int time_step, const Eigen::VectorXd &x, const State &state) const override;
		double value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const override;
		void compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		virtual IntegrableFunctional get_integral_functional() const = 0;

		const State &state_;
		SpatialIntegralType spatial_integral_type_;
		std::set<int> ids_;
	};

	class ElasticEnergyForm : public SpatialIntegralForm
	{
	public:
		ElasticEnergyForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		void compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};

	class StressNormForm : public SpatialIntegralForm
	{
	public:
		StressNormForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			if (args["power"] > 0)
				in_power_ = args["power"];
		}

		void compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		int in_power_ = 2;
	};

	class ComplianceForm : public SpatialIntegralForm
	{
	public:
		ComplianceForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		void compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};

	class PositionForm : public SpatialIntegralForm
	{
	public:
		PositionForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			set_dim(args["dim"]);
		}

		void set_dim(const int dim) { dim_ = dim; }

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		int dim_ = 0;
	};

	class AccelerationForm : public SpatialIntegralForm
	{
	public:
		AccelerationForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			set_dim(args["dim"].get<int>());
		}

		void set_dim(const int dim) { dim_ = dim; }

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		int dim_ = 0;
	};

	class KineticForm : public SpatialIntegralForm
	{
	public:
		KineticForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};

	class StressForm : public SpatialIntegralForm
	{
	public:
		StressForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			dimensions_ = args["dimensions"].get<std::vector<int>>();
		}

		void compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;

		std::vector<int> dimensions_;
	};

	// Integral of one entry of displacement gradient
	class DispGradForm : public SpatialIntegralForm
	{
	public:
		DispGradForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			dimensions_ = args["dimensions"].get<std::vector<int>>();
		}

		void compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;

		std::vector<int> dimensions_;
	};

	class VolumeForm : public SpatialIntegralForm
	{
	public:
		VolumeForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};
} // namespace polyfem::solver