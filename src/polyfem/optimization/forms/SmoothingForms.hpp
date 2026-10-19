#pragma once

#include <polyfem/optimization/forms/AdjointForm.hpp>

#include <Eigen/Core>

#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace polyfem::solver
{
	class BoundarySmoothingForm : public AdjointForm
	{
	public:
		BoundarySmoothingForm(
			const VariableToSimulationGroup &variable_to_simulations,
			std::shared_ptr<const varform::DifferentiableVarForm> varform,
			const bool scale_invariant,
			const int power,
			const std::vector<int> &surface_selections,
			const std::vector<int> &active_dims);

		double value_unweighted(const Eigen::VectorXd &x) const override;
		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	private:
		std::shared_ptr<const varform::DifferentiableVarForm> varform_;
		const bool scale_invariant_;
		const int power_; // only if scale_invariant_ is true
		Eigen::SparseMatrix<bool, Eigen::RowMajor> adj;
		Eigen::SparseMatrix<double, Eigen::RowMajor> L;
		std::set<int> surface_ids_;
		std::vector<int> active_dims_;
	};

	/// @brief Penalizes relative jumps in per-element Lamé parameters.
	///
	/// J = weight / (neighboring pair num) ∑ᵢⱼ [(1 − λᵢ/λⱼ)² + (1 − μᵢ/μⱼ)²].
	/// ij denotes neighboring elements.
	/// λ, μ denotes lame parameters.
	///
	/// For penalty of the form f = (1 − pᵢ/pⱼ)², the first derivatives are
	/// ∂f/∂pᵢ = 2(pᵢ/pⱼ − 1)/pⱼ
	/// ∂f/∂pⱼ = 2(1 − pᵢ/pⱼ)pᵢ/pⱼ²
	class ElasticMaterialSmoothingForm : public AdjointForm
	{
	public:
		/// @brief Create an elastic-material (Lamé-parameter) smoothing term.
		/// @param volume_selections Active body ids. Empty implies all active.
		ElasticMaterialSmoothingForm(
			const VariableToSimulationGroup &variable_to_simulations,
			std::shared_ptr<const varform::DifferentiableVarForm> varform,
			const std::vector<int> &volume_selections);

		std::string name() const override { return "elastic_material_smoothing"; }

		double value_unweighted(const Eigen::VectorXd &x) const override;
		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	private:
		Eigen::VectorXd lame_parameters(const Eigen::VectorXd &x) const;

		std::shared_ptr<const varform::DifferentiableVarForm> varform_;
		/// Adjacent global element id. Contains both direction Ex. (elem a, elem b) and (elem b, elem a)
		std::vector<std::pair<int, int>> adjacent_elements_;
	};
} // namespace polyfem::solver
