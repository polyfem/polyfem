#pragma once

#include <polyfem/Problem.hpp>
#include <polyfem/ProblemWithSolution.hpp>

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
class TimeDepentendStokesProblem : public Problem
{
public:
	TimeDepentendStokesProblem(const std::string &name);

	bool has_exact_sol() const override { return false; }
	bool is_scalar() const override { return false; }

	bool is_time_dependent() const override { return is_time_dependent_; }
	void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	void set_parameters(const json &params) override;

protected:
	bool is_time_dependent_;
};

class ConstantVelocity : public TimeDepentendStokesProblem
{
public:
	ConstantVelocity(const std::string &name);

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
};

class DrivenCavity : public TimeDepentendStokesProblem
{
public:
	DrivenCavity(const std::string &name);

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
};

class DrivenCavitySmooth : public TimeDepentendStokesProblem
{
public:
	DrivenCavitySmooth(const std::string &name);

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
};

class Flow : public TimeDepentendStokesProblem
{
public:
	Flow(const std::string &name);

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

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

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

	void set_parameters(const json &params) override;

private:
	double U_;
};
} // namespace polyfem
