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
		bool is_rhs_zero() const override
		{
			for (int i = 0; i < 3; ++i)
			{
				if (!rhs_(i).is_zero())
					return false;
			}
			return true;
		}

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return has_exact_; }
		bool is_scalar() const override { return false; }

		void set_parameters(const json &params) override;

		bool is_dimention_dirichet(const int tag, const int dim) const override;
		bool all_dimentions_dirichelt() const override { return all_dimentions_dirichelt_; }

		void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void exact_grad(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		// bool is_mixed() const override { return is_mixed_; }
		int n_incremental_load_steps(const double diag) const override;

	private:
		bool all_dimentions_dirichelt_ = true;
		bool has_exact_ = false;
		bool has_exact_grad_ = false;
		// bool is_mixed_ = false;

		std::vector<Eigen::Matrix<ExpressionValue, 1, 3, Eigen::RowMajor>> forces_;
		std::vector<Eigen::Matrix<ExpressionValue, 1, 3, Eigen::RowMajor>> displacements_;

		std::vector<Eigen::Matrix<bool, 1, 3, Eigen::RowMajor>> dirichelt_dimentions_;

		Eigen::Matrix<ExpressionValue, 1, 3, Eigen::RowMajor> rhs_;
		Eigen::Matrix<ExpressionValue, 1, 3, Eigen::RowMajor> exact_;
		Eigen::Matrix<ExpressionValue, 1, 9, Eigen::RowMajor> exact_grad_;
		bool is_all_;
	};


	class GenericScalarProblem: public Problem
	{
	public:
		GenericScalarProblem(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return rhs_.is_zero(); }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return has_exact_; }
		bool is_scalar() const override { return true; }

		void set_parameters(const json &params) override;

		void exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void exact_grad(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

	private:
		std::vector<Eigen::Matrix<ExpressionValue, 1, 1, Eigen::RowMajor>> neumann_;
		std::vector<Eigen::Matrix<ExpressionValue, 1, 1, Eigen::RowMajor>> dirichlet_;

		ExpressionValue rhs_;
		ExpressionValue exact_;
		Eigen::Matrix<ExpressionValue, 1, 3, Eigen::RowMajor> exact_grad_;
		bool is_all_;
		bool has_exact_ = false;
		bool has_exact_grad_ = false;
	};
}

