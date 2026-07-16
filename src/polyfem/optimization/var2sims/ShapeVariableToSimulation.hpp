#pragma once

#include <polyfem/optimization/var2sims/VariableToSimulation.hpp>

#include <Eigen/Core>

#include <string>

namespace polyfem::solver
{
	/// @brief Maps to mesh vertex position.
	///
	/// Expect optimization variable layout:
	/// [vertex 0 position] [vertex 1 position] ...
	class ShapeVariableToSimulation : public VariableToSimulation
	{
	public:
		using StatePtrs = std::vector<std::shared_ptr<legacy::State>>;
		using DiffCachePtrs = std::vector<std::shared_ptr<DiffCache>>;

		/// @brief Construct ShapeVariableToSimulation.
		/// @param[in] states Shared ptr to all forward sim states.
		/// @param[in] diff_caches Shared ptr to all diff caches.
		/// @param[in] parametrizations Parametrizations.
		/// @param[in] active_dimensions Vector of active dimensions per mesh node. Empty implies all active.
		/// @param[in] active_geom_nodes Vector of active nodes. Empty implies all active.
		/// @throw std::runtime_error Throw if input is invalid.
		ShapeVariableToSimulation(StatePtrs states,
								  DiffCachePtrs diff_caches,
								  CompositeParametrization parametrizations,
								  Eigen::VectorXi active_dimensions,
								  Eigen::VectorXi active_geom_nodes);

		std::string name() const override;
		ParameterType parameter_type() const override;
		bool affect_state(const legacy::State &target) const override;
		void update(const Eigen::VectorXd &x) override;
		void update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const override;
		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const override;
		int inverse_dof() const override;
		Eigen::VectorXd inverse_eval() const override;
		Eigen::VectorXd apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const override;

	private:
		int dim_;
		int vertex_num_;
		StatePtrs states_;
		DiffCachePtrs diff_caches_;
		CompositeParametrization parametrization_;
		Eigen::VectorXi active_dimensions_;
		Eigen::VectorXi active_geom_nodes_;

		/// @brief Return variable dof after parametrization mapping.
		int para_out_dof() const;
	};

} // namespace polyfem::solver
