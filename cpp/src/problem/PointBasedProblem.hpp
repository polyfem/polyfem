#pragma once

#include "Problem.hpp"
#include "InterpolatedFunction.hpp"


#include <vector>
#include <Eigen/Dense>

namespace poly_fem
{
	class PointBasedTensorProblem: public Problem
	{
	public:
		PointBasedTensorProblem(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }

		void set_parameters(const json &params) override;

		void init(const std::vector<int> &b_id);

		void set_constant(const int index, const Eigen::Vector3d &value);
		void set_function(const int index, const Eigen::MatrixXd &func, const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tri);
	private:
		bool initialized_ = false;
		double rhs_;
		double scaling_;
		Eigen::Vector3d translation_;
		std::vector<Eigen::Vector3d> bc_;
		std::vector<InterpolatedFunction2d> funcs_;
		std::vector<bool> val_bc_;

	};
}

