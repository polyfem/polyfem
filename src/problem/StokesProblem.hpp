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

	virtual bool has_exact_sol() const override { return false; }
	bool is_scalar() const override { return false; }

	bool is_time_dependent() const override { return is_time_dependent_; }
	virtual void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	virtual void set_parameters(const json &params) override;

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

class CollidingBalls : public TimeDepentendStokesProblem
{
public:
	CollidingBalls(const std::string &name);

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

	void set_parameters(const json &params) override;
};

class CornerFlow : public TimeDepentendStokesProblem
{
public:
	CornerFlow(const std::string &name);

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

	void set_parameters(const json &params) override;

private:
	double U_;
};

class UnitFlowWithObstacle : public TimeDepentendStokesProblem
{
public:
	UnitFlowWithObstacle(const std::string &name);

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	bool is_rhs_zero() const override { return true; }

	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

	void set_parameters(const json &params) override;

private:
	double U_;
	int inflow_;
	int dir_;
};

class TaylorGreenVortexProblem : public Problem
{
public:
	TaylorGreenVortexProblem(const std::string &name);

	bool has_exact_sol() const override { return true; }
	bool is_rhs_zero() const override { return true; }
	bool is_scalar() const override { return false; }
	bool is_time_dependent() const override { return true; }

	void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	void set_parameters(const json &params) override;

	void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
public:
	double viscosity_;
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

class TransientStokeProblemExact : public Problem
{
public:
	TransientStokeProblemExact(const std::string &name);

	bool has_exact_sol() const override { return true; }
	bool is_rhs_zero() const override { return false; }
	bool is_scalar() const override { return false; }
	bool is_time_dependent() const override { return true; }
	bool is_linear_in_time() const override { return false; }

	void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	void set_parameters(const json &params) override;

	void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

	void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
	void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

private:
	int func_;
	double viscosity_;
};

} // namespace polyfem
