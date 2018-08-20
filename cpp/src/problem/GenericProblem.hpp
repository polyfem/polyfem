#pragma once

#include <polyfem/Problem.hpp>
#include <polyfem/ExpressionValue.hpp>

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	class GenericTensorProblem: public Problem
	{
	public:
		GenericTensorProblem(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }

		void set_parameters(const json &params) override;

		bool is_dimention_dirichet(const int tag, const int dim) const override;
		bool all_dimentions_dirichelt() const override { return all_dimentions_dirichelt_; }
	private:
		bool all_dimentions_dirichelt_ = true;

		std::vector<Eigen::Matrix<ExpressionValue, 1, 3, Eigen::RowMajor>> forces_;
		std::vector<Eigen::Matrix<ExpressionValue, 1, 3, Eigen::RowMajor>> displacements_;

		std::vector<Eigen::Matrix<bool, 1, 3, Eigen::RowMajor>> dirichelt_dimentions_;


	};
}

