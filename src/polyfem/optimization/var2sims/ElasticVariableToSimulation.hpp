#pragma once

#include <polyfem/optimization/var2sims/VariableToSimulation.hpp>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

namespace polyfem::solver
{
	/// @brief Maps to lame coefficients per element.
	///
	/// Ex. If a tet mesh has 100 tets, we maps 100 pairs of lame coefficients to their
	/// respective tets.
	///
	/// Expect optimization variables with layout:
	/// [ lambda for all elements ...] [ mu for all elements ... ]
	class ElasticVariableToSimulation : public VariableToSimulation
	{
	public:
		using StatePtrs = std::vector<std::shared_ptr<legacy::State>>;
		using DiffCachePtrs = std::vector<std::shared_ptr<DiffCache>>;

		/// @brief Construct ElasticVariableToSimulation.
		/// @param[in] states Shared ptr to all forward sim states.
		/// @param[in] diff_caches Shared ptr to all diff caches.
		/// @param[in] parametrizations Parametrizations.
		/// @throw std::runtime_error Throw if input is invalid.
		ElasticVariableToSimulation(StatePtrs states,
									DiffCachePtrs diff_caches,
									CompositeParametrization parametrizations);

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
		int elem_num_;
		StatePtrs states_;
		DiffCachePtrs diff_caches_;
		CompositeParametrization parametrization_;

		int para_out_dof() const;
	};

} // namespace polyfem::solver
