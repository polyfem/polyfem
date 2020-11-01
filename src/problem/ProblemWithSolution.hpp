#pragma once

#include <polyfem/Problem.hpp>
#include <polyfem/AssemblerUtils.hpp>

#include <polyfem/AutodiffTypes.hpp>

namespace polyfem
{
	class ProblemWithSolution : public Problem
	{
	public:
		ProblemWithSolution(const std::string &name);

		virtual void rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		virtual void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		virtual void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		virtual void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		virtual bool has_exact_sol() const override { return true; }
		virtual bool is_rhs_zero() const override { return false; }

		virtual ~ProblemWithSolution() {}

	protected:
		virtual VectorNd eval_fun(const VectorNd &pt, double t) const = 0;
		virtual AutodiffGradPt eval_fun(const AutodiffGradPt &pt, double t) const = 0;
		virtual AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, double t) const = 0;

		virtual int size_for(const Eigen::MatrixXd &pts) const { return is_scalar() ? 1 : pts.cols(); }
	};

	class BilaplacianProblemWithSolution : public Problem
	{
	public:
		BilaplacianProblemWithSolution(const std::string &name);

		void rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return true; }
		bool is_rhs_zero() const override { return false; }
		bool is_scalar() const override { return true; }

		virtual ~BilaplacianProblemWithSolution() {}
	};
} // namespace polyfem
