#pragma once

#include <polyfem/Problem.hpp>
#include <polyfem/ProblemWithSolution.hpp>

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	class ElasticProblem: public Problem
	{
	public:
		ElasticProblem(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return true; }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }

		int n_incremental_load_steps(const double diag) const override { return 1/diag; }
	};

	class TorsionElasticProblem: public Problem
	{
	public:
		TorsionElasticProblem(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return true; }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }
		bool is_linear_in_time() const override { return false; }

		void set_parameters(const json &params) override;

		int n_incremental_load_steps(const double diag) const override { return 10 * n_turns_; }

	private:
		double n_turns_ = 0.5;
		int coordiante_0_ = 0;
		int coordiante_1_ = 1;
		RowVectorNd trans_;
	};

	class ElasticProblemZeroBC: public Problem
	{
	public:
		ElasticProblemZeroBC(const std::string &name);
		bool is_rhs_zero() const override { return false; }


		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }

		int n_incremental_load_steps(const double diag) const override { return 2/diag; }
	};


	class ElasticProblemExact: public ProblemWithSolution
	{
	public:
		ElasticProblemExact(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const override;

		bool is_scalar() const override { return false; }
	};


	class CompressionElasticProblemExact: public ProblemWithSolution
	{
	public:
		CompressionElasticProblemExact(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const override;

		bool is_scalar() const override { return false; }
	};



	class QuadraticElasticProblemExact: public ProblemWithSolution
	{
	public:
		QuadraticElasticProblemExact(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const override;

		bool is_scalar() const override { return false; }
	};


	class LinearElasticProblemExact: public ProblemWithSolution
	{
	public:
		LinearElasticProblemExact(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const override;

		bool is_scalar() const override { return false; }
	};

	class GravityProblem: public Problem
	{
	public:
		GravityProblem(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return false; }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;
		void velocity_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void acceleration_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void initial_velocity(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void initial_acceleration(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }
		bool is_time_dependent() const override { return true; }

		void set_parameters(const json &params) override;

	private:
			double force_ = 0.1;
	};
}

