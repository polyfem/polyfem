#pragma once

#include <polyfem/ProblemWithSolution.hpp>

#include <Eigen/Dense>

#include <vector>
#include <string>


namespace polyfem
{
	class KernelProblem: public ProblemWithSolution
	{
	public:
		KernelProblem(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const override { assert(false); return AutodiffHessianPt(1); }
		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		void set_parameters(const json &params) override;
		bool is_scalar() const override;

	private:
		std::string formulation_ = "Laplacian";
		int n_kernels_ = 5;
		double kernel_distance_ = 0.05;
	};
}


