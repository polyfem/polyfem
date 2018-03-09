#pragma once

#include "Problem.hpp"

#include "AutodiffTypes.hpp"

namespace poly_fem
{
	class ProblemWithSolution : public Problem
	{
	public:
		ProblemWithSolution(const std::string &name);

		virtual void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const;
		virtual void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const;

		virtual void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const;
		virtual void exact_grad(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const;


		virtual bool has_exact_sol() const { return true; }
		virtual bool has_gradient() const { return true; }


		virtual ~ProblemWithSolution() { }
	protected:
		virtual VectorNd eval_fun(const VectorNd &pt) const = 0;
		virtual AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const = 0;
		virtual AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const = 0;

		virtual int size_for(const Eigen::MatrixXd &pts) const { return is_scalar() ? 1 : pts.cols(); }
	};
}


