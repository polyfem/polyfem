#pragma once

#include <polyfem/assembler/Problem.hpp>
#include "ProblemWithSolution.hpp"

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	namespace problem
	{
		class TimeDepentendStokesProblem : public assembler::Problem
		{
		public:
			TimeDepentendStokesProblem(const std::string &name);

			virtual bool has_exact_sol() const override { return false; }
			bool is_scalar() const override { return false; }

			bool is_time_dependent() const override { return is_time_dependent_; }
			bool is_constant_in_time() const override { return !is_time_dependent_; }
			virtual void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			virtual void set_parameters(const json &params) override;

		protected:
			bool is_time_dependent_;
		};

		class ConstantVelocity : public TimeDepentendStokesProblem
		{
		public:
			ConstantVelocity(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		};

		class TwoSpheres : public TimeDepentendStokesProblem
		{
		public:
			TwoSpheres(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
			void initial_density(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		};

		class DrivenCavity : public TimeDepentendStokesProblem
		{
		public:
			DrivenCavity(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		};

		class DrivenCavityC0 : public TimeDepentendStokesProblem
		{
		public:
			DrivenCavityC0(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		};

		class DrivenCavitySmooth : public TimeDepentendStokesProblem
		{
		public:
			DrivenCavitySmooth(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		};

		class Flow : public TimeDepentendStokesProblem
		{
		public:
			Flow(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

		private:
			int inflow_;
			int outflow_;

			int flow_dir_;

			double inflow_amout_;
			double outflow_amout_;
		};

		class FlowWithObstacle : public TimeDepentendStokesProblem
		{
		public:
			FlowWithObstacle(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

		private:
			double U_;
		};

		class Kovnaszy : public assembler::Problem
		{
		public:
			Kovnaszy(const std::string &name);

			bool has_exact_sol() const override { return true; }
			bool is_rhs_zero() const override { return true; }
			bool is_scalar() const override { return false; }
			bool is_time_dependent() const override { return is_time_dependent_; }

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

			void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		public:
			double viscosity_;
			bool is_time_dependent_;
		};

		class CornerFlow : public TimeDepentendStokesProblem
		{
		public:
			CornerFlow(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

		private:
			double U_;
		};

		class Lshape : public TimeDepentendStokesProblem
		{
		public:
			Lshape(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

		private:
			double U_;
		};

		class UnitFlowWithObstacle : public TimeDepentendStokesProblem
		{
		public:
			UnitFlowWithObstacle(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

		private:
			double U_;
			int inflow_;
			int dir_;
		};

		class TaylorGreenVortexProblem : public assembler::Problem
		{
		public:
			TaylorGreenVortexProblem(const std::string &name);

			bool has_exact_sol() const override { return true; }
			bool is_rhs_zero() const override { return true; }
			bool is_scalar() const override { return false; }
			bool is_time_dependent() const override { return true; }

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

			void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		public:
			double viscosity_;
		};

		class StokesLawProblem : public assembler::Problem
		{
		public:
			StokesLawProblem(const std::string &name);

			bool has_exact_sol() const override { return true; }
			bool is_rhs_zero() const override { return true; }
			bool is_scalar() const override { return false; }
			bool is_time_dependent() const override { return is_time_dependent_; }

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

			void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		public:
			double viscosity_, radius;
			bool is_time_dependent_;
		};

		class Airfoil : public assembler::Problem
		{
		public:
			Airfoil(const std::string &name);

			bool has_exact_sol() const override { return true; }
			bool is_rhs_zero() const override { return true; }
			bool is_scalar() const override { return false; }
			bool is_time_dependent() const override { return is_time_dependent_; }

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

			void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		public:
			bool is_time_dependent_;
		};

		class SimpleStokeProblemExact : public ProblemWithSolution
		{
		public:
			SimpleStokeProblemExact(const std::string &name);

			VectorNd eval_fun(const VectorNd &pt, const double t) const override;
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

			bool is_scalar() const override { return false; }

			void set_parameters(const json &params) override;

		private:
			int func_;
		};

		class SineStokeProblemExact : public ProblemWithSolution
		{
		public:
			SineStokeProblemExact(const std::string &name);

			VectorNd eval_fun(const VectorNd &pt, const double t) const override;
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

			bool is_scalar() const override { return false; }
		};

		class TransientStokeProblemExact : public assembler::Problem
		{
		public:
			TransientStokeProblemExact(const std::string &name);

			bool has_exact_sol() const override { return true; }
			bool is_rhs_zero() const override { return false; }
			bool is_scalar() const override { return false; }
			bool is_time_dependent() const override { return true; }
			bool is_constant_in_time() const override { return false; }

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

			void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		private:
			int func_;
			double viscosity_;
		};
	} // namespace problem
} // namespace polyfem