#pragma once

#include "VariableToSimulation.hpp"
#include <polyfem/io/ResourceIO.hpp>
#include <polyfem/optimization/parametrization/PeriodicMeshToMesh.hpp>
#include <inflator/Inflator.hpp>

namespace polyfem::solver
{
	/// Regenerate periodic meshes from [cell width, inflator parameters] and map shape derivatives back to design variables.
	class InflatedPeriodicShapeVariableToSimulation final : public VariableToSimulation
	{
	public:
		InflatedPeriodicShapeVariableToSimulation(
			std::vector<std::shared_ptr<varform::DifferentiableVarForm>> varforms,
			std::vector<std::shared_ptr<DiffCache>> diff_caches,
			CompositeParametrization parametrization, const json &settings, const io::ResourceIO &resources);

		std::string name() const override { return "inflated-periodic-shape"; }
		ParameterType parameter_type() const override { return ParameterType::PeriodicShape; }
		bool affects_varform(const varform::DifferentiableVarForm &target) const override;
		void update(const Eigen::VectorXd &variables) override;
		void update_state_variables(const Eigen::VectorXd &variables, Eigen::VectorXd &state_variables) const override;
		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &variables) const override;
		int inverse_dof() const override;
		Eigen::VectorXd inverse_eval() const override;
		Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &variables) const override;

	private:
		std::vector<std::shared_ptr<varform::DifferentiableVarForm>> varforms_;
		std::vector<std::shared_ptr<DiffCache>> diff_caches_;
		CompositeParametrization parametrization_;
		int dimension_;
		std::filesystem::path wire_;
		std::string work_directory_;
		json meshing_;
		inflator::Options options_;
		int generation_ = 0;
		Eigen::VectorXd initial_;
		Eigen::VectorXd last_parameters_;
		Eigen::VectorXd periodic_parameters_;
		/// Rows are design parameters; columns are interleaved physical vertex coordinates.
		Eigen::MatrixXd shape_jacobian_;
		/// Rows are design parameters; columns are periodic coordinates followed by the cell matrix.
		Eigen::MatrixXd periodic_jacobian_;
		std::unique_ptr<PeriodicMeshToMesh> periodic_map_;
	};
} // namespace polyfem::solver
