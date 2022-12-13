#pragma once

#include "AdjointForm.hpp"
#include "Objective.hpp"

#include <polyfem/solver/forms/ContactForm.hpp>

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
} // namespace polyfem::solver