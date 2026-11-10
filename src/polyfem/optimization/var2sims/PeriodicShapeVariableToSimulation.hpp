#pragma once

#include <polyfem/optimization/parametrization/PeriodicMeshToMesh.hpp>
#include <polyfem/optimization/var2sims/VariableToSimulation.hpp>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

namespace polyfem::solver
{
	class PeriodicShapeVariableToSimulation : public VariableToSimulation
	{
	public:
		using VarFormPtrs = std::vector<std::shared_ptr<varform::DifferentiableVarForm>>;
		using DiffCachePtrs = std::vector<std::shared_ptr<DiffCache>>;

		/// @brief Construct PeriodicShapeVariableToSimulation.
		/// @param[in] varforms Shared ptr to all forward sim varforms.
		/// @param[in] diff_caches Shared ptr to all diff caches.
		/// @param[in] parametrizations Parametrizations.
		/// @throw std::runtime_error Throw if input is invalid.
		PeriodicShapeVariableToSimulation(VarFormPtrs varforms,
										  DiffCachePtrs diff_caches,
										  CompositeParametrization parametrizations);

		std::string name() const override;
		ParameterType parameter_type() const override;
		bool affects_varform(const varform::DifferentiableVarForm &target) const override;
		void update(const Eigen::VectorXd &x) override;
		void update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const override;
		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		int inverse_dof() const override;
		Eigen::VectorXd inverse_eval() const override;
		Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const override;

	private:
		int dim_;
		int vertex_num_;
		VarFormPtrs varforms_;
		DiffCachePtrs diff_caches_;
		CompositeParametrization parametrization_;

		// Helper to map a cell to periodic structure.
		std::unique_ptr<PeriodicMeshToMesh> periodic_mesh_map_;

		int para_out_dof() const;
	};

} // namespace polyfem::solver
