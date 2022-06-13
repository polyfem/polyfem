#pragma once

#include "ProblemWithSolution.hpp"

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	namespace problem
	{
		class TestProblem : public ProblemWithSolution
		{
		public:
			TestProblem(const std::string &name);

			VectorNd eval_fun(const VectorNd &pt, const double t) const override { return eval_impl(pt); }
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override { return eval_impl(pt); }
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override { return eval_impl(pt); }

			bool is_scalar() const override { return params_["is_scalar"]; }

			void set_parameters(const json &params) override;

		private:
			template <typename T>
			T eval_impl(const T &pt) const;

		private:
			json params_;
		};
	} // namespace problem
} // namespace polyfem
