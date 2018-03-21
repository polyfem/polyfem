#pragma once

#include "Problem.hpp"
#include "InterpolatedFunction.hpp"


#include <vector>
#include <Eigen/Dense>

namespace poly_fem
{
	class CustomProblem: public Problem
	{
	public:
		CustomProblem(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }

		void set_parameters(const json &params) override;
	private:
		double rhs_;
		double scaling_;
		Eigen::Vector3d translation_;
		std::vector<Eigen::Vector3d> bc_;
		std::vector<InterpolatedFunction2d> funcs_;
		std::vector<bool> val_bc_;

	};
}

