#ifndef FRANKE_2D_PROBLEM_HPP
#define FRANKE_2D_PROBLEM_HPP

#include "Problem.hpp"

#include <vector>
#include <Eigen/Dense>

namespace poly_fem
{
	class Franke2dProblem: public Problem
	{
	public:
		Franke2dProblem(const std::string &name);

		void rhs(const std::string &formulation, const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void bc(const std::string &formulation, const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		void exact(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void exact_grad(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return true; }
		bool is_scalar() const override { return true; }
		bool has_gradient() const override { return true; }
	};
}

#endif //FRANKE_2D_PROBLEM_HPP

