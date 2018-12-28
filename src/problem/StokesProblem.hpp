#pragma once

#include <polyfem/Problem.hpp>
#include <polyfem/ProblemWithSolution.hpp>

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	class DrivenCavity: public Problem
	{
	public:
		DrivenCavity(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return true; }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }
		// bool is_mixed() const override { return true; }
	};


	class Flow: public Problem
	{
	public:
		Flow(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return true; }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }
		// bool is_mixed() const override { return true; }
	};


	class TimeDependentFlow: public Problem
	{
	public:
		TimeDependentFlow(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return true; }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;

		void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		bool is_time_dependent() const override { return true; }
		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }
		// bool is_mixed() const override { return true; }
	};
}
