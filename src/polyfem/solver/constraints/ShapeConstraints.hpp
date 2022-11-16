#pragma once

#include "Constraints.hpp"

namespace polyfem
{
	class ShapeConstraints : public Constraints
	{
	public:
		void update_state(std::shared_ptr<State> state, const Eigen::VectorXd &reduced) override
		{
			Eigen::MatrixXd V_rest, V_full;
			Eigen::MatrixXi F;
			state->get_vf(V_rest, F);
			reduced_to_full(reduced, V_rest, V_full);
			state->set_v(V_full);
		}

		ShapeConstraints(const json &constraint_params, int full_size, int reduced_size) : Constraints(constraint_params, full_size, reduced_size)
		{
			if (constraint_params["type"] == "identity")
			{
				// full_to_reduced_diff_ = [this](const DiffMatrix &full) {
				// 	DiffVector reduced(reduced_size_);
				// 	for (int i = 0; i < reduced_size_; ++i)
				// 		reduced(i) = full(i);
				// 	return reduced;
				// };
				reduced_to_full_ = [this](const Eigen::VectorXd &reduced, const Eigen::MatrixXd &V_rest) {
					Eigen::MatrixXd full(full_size_, 1);
					for (int i = 0; i < full_size_; ++i)
						full(i) = reduced(i);
					return full;
				};
				full_to_reduced_ = [this](const Eigen::MatrixXd &V_full) {
					Eigen::VectorXd reduced(reduced_size_);
					for (int i = 0; i < reduced_size_; ++i)
						reduced(i) = V_full(i);
					return reduced;
				};
				dfull_to_dreduced_ = [this](const Eigen::VectorXd &dV_full) {
					Eigen::VectorXd dreduced(reduced_size_);
					for (int i = 0; i < reduced_size_; ++i)
						dreduced(i) = dV_full(i);
					return dreduced;
				};
			}
		}

		void reduced_to_full(const Eigen::VectorXd &reduced, const Eigen::MatrixXd &V_rest, Eigen::MatrixXd &V_full) const
		{
			V_full = reduced_to_full_(reduced, V_rest);
		}

		void full_to_reduced(const Eigen::VectorXd &V_full, Eigen::VectorXd &reduced) const
		{
			if (full_to_reduced_ != nullptr)
			{
				reduced = full_to_reduced_(V_full);
				return;
			}

			// DiffScalarBase::setVariableCount(full_size_); // size of input vector
			// assert(full.size() == full_size_);
			// DiffVector full_diff(full_size_);
			// for (int i = 0; i < full.size(); ++i)
			// 	full_diff(i) = DiffScalar(i, full(i));
			// auto reduced_diff = full_to_reduced_diff_(full_diff);
			// assert(reduced_diff.size() == reduced_size_);
			// reduced.resize(reduced_size_);
			// for (int i = 0; i < reduced_size_; ++i)
			// 	reduced(i) = reduced_diff(i).getValue();
		}

		void dfull_to_dreduced(const Eigen::VectorXd &V_full, const Eigen::VectorXd &dV_full, Eigen::VectorXd &dreduced) const
		{
			if (dfull_to_dreduced_ != nullptr)
			{
				dreduced = dfull_to_dreduced_(dV_full);
				return;
			}

			// DiffScalarBase::setVariableCount(full_size_); // size of input vector
			// assert(full.size() == full_size_);
			// DiffVector full_diff(full_size_);
			// for (int i = 0; i < full.size(); ++i)
			// 	full_diff(i) = DiffScalar(i, full(i));
			// auto reduced_diff = full_to_reduced_diff_(full_diff);
			// assert(reduced_diff.size() == reduced_size_);
			// Eigen::MatrixXd grad(reduced_size_, full_size_);
			// for (int i = 0; i < reduced_size_; ++i)
			// {
			// 	for (int j = 0; j < full.size(); ++j)
			// 		grad(i, j) = reduced_diff(i).getGradient()(j);
			// }

			// dreduced = grad * dV_full;
		}

	private:
		const Eigen::MatrixXd V_rest;

		// Must define the inverse function with Eigen types, differentiability is not needed.
		std::function<Eigen::MatrixXd(const Eigen::VectorXd &, const Eigen::MatrixXd &)> reduced_to_full_;

		// For differentiability, either define the function with differentiable types.
		// std::function<DiffVector(const DiffMatrix &)> full_to_reduced_diff_;
		// Or explicitly define the function and gradient with Eigen types.
		std::function<Eigen::VectorXd(const Eigen::MatrixXd &)> full_to_reduced_;
		std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> dfull_to_dreduced_;
	};
} // namespace polyfem