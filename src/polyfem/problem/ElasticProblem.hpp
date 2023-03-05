#pragma once

#include <polyfem/assembler/Problem.hpp>
#include "ProblemWithSolution.hpp"

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	namespace problem
	{
		class ElasticProblem : public assembler::Problem
		{
		public:
			ElasticProblem(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			bool has_exact_sol() const override { return false; }
			bool is_scalar() const override { return false; }
		};

		class TorsionElasticProblem : public assembler::Problem
		{
		public:
			TorsionElasticProblem(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			bool has_exact_sol() const override { return false; }
			bool is_scalar() const override { return false; }
			bool is_constant_in_time() const override { return false; }

			void set_parameters(const json &params) override;

		private:
			double n_turns_ = 0.5;
			int coordiante_0_ = 0;
			int coordiante_1_ = 1;
			RowVectorNd trans_;
		};

		class DoubleTorsionElasticProblem : public assembler::Problem
		{
		public:
			DoubleTorsionElasticProblem(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
			void initial_velocity(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
			void initial_acceleration(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			bool has_exact_sol() const override { return false; }
			bool is_scalar() const override { return false; }
			bool is_constant_in_time() const override { return false; }
			bool is_time_dependent() const override { return true; }

			void set_parameters(const json &params) override;

		private:
			double angular_v0_ = 0.5;
			double angular_v1_ = -0.5;
			std::array<int, 2> coordiante_0_ = {{0, 1}};
			std::array<int, 2> coordiante_1_ = {{0, 1}};
			RowVectorNd trans_0_;
			RowVectorNd trans_1_;
		};

		class ElasticProblemZeroBC : public assembler::Problem
		{
		public:
			ElasticProblemZeroBC(const std::string &name);
			bool is_rhs_zero() const override { return false; }

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			bool has_exact_sol() const override { return false; }
			bool is_scalar() const override { return false; }
		};

		class ElasticProblemExact : public ProblemWithSolution
		{
		public:
			ElasticProblemExact(const std::string &name);

			VectorNd eval_fun(const VectorNd &pt, const double t) const override;
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

			bool is_scalar() const override { return false; }
		};

		class CompressionElasticProblemExact : public ProblemWithSolution
		{
		public:
			CompressionElasticProblemExact(const std::string &name);

			VectorNd eval_fun(const VectorNd &pt, const double t) const override;
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

			bool is_scalar() const override { return false; }
		};

		class QuadraticElasticProblemExact : public ProblemWithSolution
		{
		public:
			QuadraticElasticProblemExact(const std::string &name);

			VectorNd eval_fun(const VectorNd &pt, const double t) const override;
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

			bool is_scalar() const override { return false; }
		};

		class LinearElasticProblemExact : public ProblemWithSolution
		{
		public:
			LinearElasticProblemExact(const std::string &name);

			VectorNd eval_fun(const VectorNd &pt, const double t) const override;
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const override;
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const override;

			bool is_scalar() const override { return false; }
		};

		class GravityProblem : public assembler::Problem
		{
		public:
			GravityProblem(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return false; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
			void initial_velocity(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
			void initial_acceleration(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			bool has_exact_sol() const override { return false; }
			bool is_scalar() const override { return false; }
			bool is_time_dependent() const override { return true; }

			void set_parameters(const json &params) override;

		private:
			double force_ = 0.1;
		};

		class WalkProblem : public assembler::Problem
		{
		public:
			WalkProblem(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return true; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
			void initial_velocity(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
			void initial_acceleration(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			bool has_exact_sol() const override { return false; }
			bool is_scalar() const override { return false; }
			bool is_time_dependent() const override { return true; }
			bool is_constant_in_time() const override { return false; }
		};

		class ElasticCantileverExact : public assembler::Problem
		{
		public:
			ElasticCantileverExact(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return false; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const override;

			bool has_exact_sol() const override { return true; }
			void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_scalar() const override { return false; }

			void set_parameters(const json &params) override;

		private:
			VectorNd eval_fun(const VectorNd &pt, const double t) const;
			AutodiffGradPt eval_fun(const AutodiffGradPt &pt, const double t) const;
			AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt, const double t) const;
			int size_for(const Eigen::MatrixXd &pts) const { return is_scalar() ? 1 : pts.cols(); }

			double singular_point_displacement;
			double E;
			double nu;
			std::string formulation;
			double length;
			double width;
		};
	} // namespace problem
} // namespace polyfem
