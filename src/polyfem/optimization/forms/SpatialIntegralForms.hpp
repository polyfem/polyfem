#pragma once

#include "AdjointForm.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	class SpatialIntegralForm : public StaticForm
	{
	public:
		SpatialIntegralForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : StaticForm(variable_to_simulations), state_(state)
		{
		}

		std::string name() const override { return "spatial_integral"; }

		const State &get_state() { return state_; }

		Eigen::VectorXd compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const override;
		double value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const override;
		void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		void set_integral_type(const SpatialIntegralType type) { spatial_integral_type_ = type; }

		virtual IntegrableFunctional get_integral_functional() const = 0;

		const State &state_;
		SpatialIntegralType spatial_integral_type_;
		std::set<int> ids_;
	};

	class ElasticEnergyForm : public SpatialIntegralForm
	{
	public:
		ElasticEnergyForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		std::string name() const override { return "elastic_energy"; }

		void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};

	class StressNormForm : public SpatialIntegralForm
	{
	public:
		StressNormForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			if (args["power"] > 0)
				in_power_ = args["power"];
		}

		std::string name() const override { return "stress_norm"; }

		void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		int in_power_ = 2;
	};

	class DirichletEnergyForm : public SpatialIntegralForm
	{
	public:
		DirichletEnergyForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)

		{
			std::string formulation = state.formulation();
			if (!(formulation == "Laplacian" || formulation == "Electrostatics"))
				log_and_throw_adjoint_error("DirichletEnergyForm can only be used with Laplacian or Electrostatics problems!");

			set_integral_type(SpatialIntegralType::Volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		std::string name() const override { return "dirichlet_energy"; }

		void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};

	class ComplianceForm : public SpatialIntegralForm
	{
	public:
		ComplianceForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		std::string name() const override { return "compliance"; }

		void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};

	class PositionForm : public SpatialIntegralForm
	{
	public:
		PositionForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Volume);

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
		AccelerationForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Volume);

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
		KineticForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};

	class StressForm : public SpatialIntegralForm
	{
	public:
		StressForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			dimensions_ = args["dimensions"].get<std::vector<int>>();
		}

		std::string name() const override { return "stress"; }

		void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;

		std::vector<int> dimensions_;
	};

	class VolumeForm : public SpatialIntegralForm
	{
	public:
		VolumeForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Volume);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};
} // namespace polyfem::solver