#pragma once

#include <polyfem/Problem.hpp>
#include <polyfem/ProblemWithSolution.hpp>

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
class ConstantVelocity : public Problem
{
public:
	ConstantVelocity(const std::string &name);

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

	bool has_exact_sol() const override { return false; }
	bool is_scalar() const override { return false; }

	bool is_time_dependent() const override { return is_time_depetend_; }
	void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	void set_parameters(const json &params) override;

private:
	bool is_time_depetend_;
};
class DrivenCavity : public Problem
{
public:
	DrivenCavity(const std::string &name);

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

	bool has_exact_sol() const override { return false; }
	bool is_scalar() const override { return false; }
};

class DrivenCavitySmooth : public Problem
{
public:
	DrivenCavitySmooth(const std::string &name);

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

	bool has_exact_sol() const override { return false; }
	bool is_scalar() const override { return false; }
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

		void set_parameters(const json &params) override;
	private:
		int inflow_;
		int outflow_;

		int flow_dir_;

		double inflow_amout_;
		double outflow_amout_;
	};

	class FlowWithObstacle : public Problem
	{
	public:
		FlowWithObstacle(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return true; }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }
		bool is_time_dependent() const override { return is_time_depetend_; }

		void set_parameters(const json &params) override;

		void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	private:
		double U_;
		bool is_time_depetend_;
	};

	class TimeDependentFlow: public Flow
	{
	public:
		TimeDependentFlow(const std::string &name);

		void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		bool is_time_dependent() const override { return true; }
	};
}
