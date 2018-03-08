#ifndef FRANKE_3D_PROBLEM_HPP
#define FRANKE_3D_PROBLEM_HPP

#include "Problem.hpp"

#include <vector>
#include <Eigen/Dense>

namespace poly_fem
{
	class Franke3dProblem: public Problem
	{
	public:
		Franke3dProblem(const std::string &name);

		void rhs(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void exact_grad(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return true; }
		bool is_scalar() const override { return true; }
		bool has_gradient() const override { return true; }
	};
}

#endif //FRANKE_3D_PROBLEM_HPP

