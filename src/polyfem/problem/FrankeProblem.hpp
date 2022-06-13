#pragma once

#include "ProblemWithSolution.hpp"

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	namespace problem
	{
		class FrankeProblem : public ProblemWithSolution
		{
		public:
			FrankeProblem(const std::string &name);

			VectorNd eval_fun(const VectorNd &pt, const double t) const override;
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

			bool is_scalar() const override { return true; }
		};

		class FrankeProblemOld : public ProblemWithSolution
		{
		public:
			FrankeProblemOld(const std::string &name);

			VectorNd eval_fun(const VectorNd &pt, const double t) const override;
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

			bool is_scalar() const override { return true; }
		};
	} // namespace problem
} // namespace polyfem
