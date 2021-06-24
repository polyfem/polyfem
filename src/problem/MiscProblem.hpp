#pragma once

#include <polyfem/ProblemWithSolution.hpp>

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	class LinearProblem : public ProblemWithSolution
	{
	public:
		LinearProblem(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt, const double t) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

		bool is_scalar() const override { return true; }
	};

	class QuadraticProblem : public ProblemWithSolution
	{
	public:
		QuadraticProblem(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt, const double t) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

		bool is_scalar() const override { return true; }
	};

	class CubicProblem : public ProblemWithSolution
	{
	public:
		CubicProblem(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt, const double t) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

		bool is_scalar() const override { return true; }
	};

	class SineProblem : public ProblemWithSolution
	{
	public:
		SineProblem(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt, const double t) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

		bool is_scalar() const override { return true; }
	};

	class ZeroBCProblem : public ProblemWithSolution
	{
	public:
		ZeroBCProblem(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt, const double t) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

		bool is_scalar() const override { return true; }
	};

	class MinSurfProblem : public Problem
	{
	public:
		MinSurfProblem(const std::string &name);

		void rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return false; }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		bool is_scalar() const override { return true; }
		bool has_exact_sol() const override { return false; }
	};

	class TimeDependentProblem : public Problem
	{
	public:
		TimeDependentProblem(const std::string &name);

		void rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return false; }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return true; }
		bool is_time_dependent() const override { return true; }
	};

	class GenericScalarProblemExact : public ProblemWithSolution
	{
	public:
		GenericScalarProblemExact(const std::string &name);

		bool is_scalar() const override { return true; }
		bool is_time_dependent() const override { return func_ <= 1; }
		bool is_constant_in_time() const override { return false; }

		void initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		void set_parameters(const json &params) override;

		VectorNd eval_fun(const VectorNd &pt, double t) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt, double t) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, double t) const override;

		void rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

	private:
		int func_;
	};
} // namespace polyfem
