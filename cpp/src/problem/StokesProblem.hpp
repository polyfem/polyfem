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
		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }
		bool is_stokes() const override { return true; }
	};


	class Flow: public Problem
	{
	public:
		Flow(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;
		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }
		bool is_stokes() const override { return true; }
	};
}
