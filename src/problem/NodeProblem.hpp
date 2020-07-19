#pragma once

#include <polyfem/Problem.hpp>
#include <polyfem/InterpolatedFunction.hpp>
#include <polyfem/RBFInterpolation.hpp>


#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	class NodeValues
	{
	public:

		NodeValues();

		void load(const std::string &path);
		void init(const Mesh &mesh);

		double dirichlet_interpolate(const int p_id, const Eigen::MatrixXd &uv) const
		{
			return interpolate(p_id, uv, true);
		}
		double neumann_interpolate(const int p_id, const Eigen::MatrixXd &uv) const
		{
			return interpolate(p_id, uv, false);
		}
	private:
		double interpolate(const int p_id, const Eigen::MatrixXd &uv, bool is_dirichelt) const;

		std::vector<int> raw_ids_;
		std::vector<std::vector<double>> raw_data_;
		std::vector<bool> raw_dirichelt_;

		std::vector<Eigen::VectorXd> data_;
		std::vector<bool> dirichelt_;
	};


	class NodeProblem: public Problem
	{
	public:
		NodeProblem(const std::string &name);
		void init(const Mesh &mesh) override;

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return abs(rhs_) < 1e-10; }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return true; }

		void set_parameters(const json &params) override;

		bool is_dimention_dirichet(const int tag, const int dim) const override;
		bool all_dimentions_dirichelt() const override { return all_dimentions_dirichelt_; }
	private:
		bool all_dimentions_dirichelt_ = true;
		std::vector<Eigen::Matrix<bool, 1, 3, Eigen::RowMajor>> dirichelt_dimentions_;
		double rhs_;
		NodeValues values_;
		bool is_all_;
	};
}

