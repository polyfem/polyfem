#pragma once

#include "ProblemWithSolution.hpp"

#include <vector>
#include <Eigen/Dense>

namespace poly_fem
{
	class LinearProblem: public ProblemWithSolution
	{
	public:
		LinearProblem(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const override;

		bool is_scalar() const override { return true; }
	};

	class QuadraticProblem : public ProblemWithSolution
	{
	public:
		QuadraticProblem(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const override;

		bool is_scalar() const override { return true; }
	};

	class ZeroBCProblem: public ProblemWithSolution
	{
	public:
		ZeroBCProblem(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const override;

		bool is_scalar() const override { return true; }
	};
}


