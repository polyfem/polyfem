#pragma once

#include "AdjointForm.hpp"
#include "IntegralObjective.hpp"

#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/assembler/Multiscale.hpp>

#include <shared_mutex>
#include <array>

namespace polyfem::solver
{
	class HomogenizedEnergyObjective : public SpatialIntegralObjective
	{
	public:
		HomogenizedEnergyObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> macro_strain_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args);
		~HomogenizedEnergyObjective() = default;

		double value() override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;
		IntegrableFunctional get_integral_functional() override;

	protected:
		std::vector<int> id;
		std::string formulation_;

		std::shared_ptr<const Parameter> elastic_param_; // stress depends on elastic param
	};

	class HomogenizedStressObjective : public SpatialIntegralObjective
	{
	public:
		HomogenizedStressObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> macro_strain_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args);
		~HomogenizedStressObjective() = default;

		double value() override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;
		IntegrableFunctional get_integral_functional() override;

	protected:
		std::vector<int> id;
		std::string formulation_;

		double microstructure_volume;

		std::shared_ptr<const Parameter> elastic_param_; // stress depends on elastic param
	};

	class CompositeHomogenizedStressObjective : public Objective
	{
	public:
		CompositeHomogenizedStressObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> macro_strain_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args);
		~CompositeHomogenizedStressObjective() = default;

		double value() override;
		Eigen::MatrixXd compute_adjoint_rhs(const State &state) override;
		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;

	protected:
		std::array<std::shared_ptr<HomogenizedStressObjective>, 4> js;
	};

	class StrainObjective : public SpatialIntegralObjective
	{
	public:
		StrainObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args) : SpatialIntegralObjective(state, shape_param, args)
		{
			spatial_integral_type_ = AdjointForm::SpatialIntegralType::VOLUME;
			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}
		~StrainObjective() = default;

		IntegrableFunctional get_integral_functional() override;
	};

	class HomogenizedStiffnessObjective : public StaticObjective
	{
	public:
		HomogenizedStiffnessObjective(std::shared_ptr<State> state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args);
		~HomogenizedStiffnessObjective() = default;

		double value() override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;

		void set_time_step(int time_step) { time_step_ = time_step; }

	protected:
		std::vector<int> id;
		std::string formulation_;

		std::shared_ptr<assembler::Multiscale> multiscale_assembler;
		std::shared_ptr<const Parameter> elastic_param_, shape_param_;
	};

} // namespace polyfem::solver