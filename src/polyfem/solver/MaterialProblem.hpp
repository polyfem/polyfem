#pragma once

#include "OptimizationProblem.hpp"

namespace polyfem
{
	class MaterialProblem : public OptimizationProblem
	{
	public:
		MaterialProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_);

		using OptimizationProblem::gradient;
		using OptimizationProblem::value;

		double target_value(const TVector &x);
		double smooth_value(const TVector &x);

		void target_gradient(const TVector &x, TVector &gradv);
		void smooth_gradient(const TVector &x, TVector &gradv);

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;

		bool is_step_valid(const TVector &x0, const TVector &x1);

		int optimization_dim() override { return 0; }

		void line_search_end(bool failed);
		bool remesh(TVector &x) { return false; }

		bool solution_changed_pre(const TVector &newX) override;

		// map x (optimization variables) to parameters (lambda, mu, friction, damping)
		std::function<void(const TVector &x, State &state)> x_to_param;
		// map parameters to x
		std::function<void(TVector &x, State &state)> param_to_x;
		// compute gradient wrt. x given: gradient wrt. parameters, values of parameters
		std::function<void(TVector &dx, const Eigen::VectorXd &dparams, State &state)> dparam_to_dx;

	private:
		double min_mu, min_lambda;
		double max_mu, max_lambda;

		bool has_material_smoothing = false;
		json material_params;
		json smoothing_params;

		double smoothing_weight;
		double target_weight = 1;

		Eigen::SparseMatrix<bool> tt_adjacency;
	};
} // namespace polyfem
