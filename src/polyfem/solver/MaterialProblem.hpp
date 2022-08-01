#pragma once

#include <polyfem/solver/OptimizationProblem.hpp>

namespace polyfem
{
	class MaterialProblem : public OptimizationProblem
	{
	public:
		MaterialProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_, const json &args);

		double target_value(const TVector &x);
		double smooth_value(const TVector &x);

		void target_gradient(const TVector &x, TVector &gradv);
		void smooth_gradient(const TVector &x, TVector &gradv);

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;

		double value(const TVector &x, const bool only_elastic) { return value(x); };
		void gradient(const TVector &x, TVector &gradv, const bool only_elastic) { gradient(x, gradv); };

		bool is_step_valid(const TVector &x0, const TVector &x1);
		bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; };
		double max_step_size(const TVector &x0, const TVector &x1);

		void line_search_end(bool failed);
		bool remesh(TVector &x) { return false; };

		void solution_changed(const TVector &newX) override;

		// map x (optimization variables) to parameters (lambda, mu, friction, damping)
		std::function<void(const TVector &x, State &state)> x_to_param;
		// map parameters to x
		std::function<void(TVector &x, State &state)> param_to_x;
		// compute gradient wrt. x given: gradient wrt. parameters, values of parameters
		std::function<void(TVector &dx, const Eigen::VectorXd &dparams, State &state)> dparam_to_dx;

	private:
		double min_mu, min_lambda;
		double max_mu, max_lambda;

		double smoothing_weight;
		double target_weight = 1;

		Eigen::SparseMatrix<bool> tt_adjacency;
	};
} // namespace polyfem
