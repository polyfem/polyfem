#pragma once

#include "ProblemWithSolution.hpp"

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	namespace problem
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

		class MinSurfProblem : public assembler::Problem
		{
		public:
			MinSurfProblem(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return false; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			bool is_scalar() const override { return true; }
			bool has_exact_sol() const override { return false; }
		};

		class TimeDependentProblem : public assembler::Problem
		{
		public:
			TimeDependentProblem(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return false; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

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

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

			VectorNd eval_fun(const VectorNd &pt, double t) const override;
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, double t) const override;
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, double t) const override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		private:
			int func_;
		};
	} // namespace problem
} // namespace polyfem
