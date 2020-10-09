#pragma once

#include <polyfem/Problem.hpp>
#include <polyfem/ExpressionValue.hpp>

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace polyfem
{
	class GenericTensorProblem : public Problem
	{
	public:
		GenericTensorProblem(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override
		{
			for (int i = 0; i < 3; ++i)
			{
				if (!rhs_[i].is_zero())
					return false;
			}
			return true;
		}

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return has_exact_; }
		bool is_scalar() const override { return false; }
		bool is_time_dependent() const override { return is_time_dept_; }
		bool is_linear_in_time() const override { return !is_time_dept_; }

		void velocity_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void acceleration_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		void initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void initial_velocity(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void initial_acceleration(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		void set_parameters(const json &params) override;

		bool is_dimention_dirichet(const int tag, const int dim) const override;
		bool all_dimentions_dirichelt() const override { return all_dimentions_dirichelt_; }

		void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		// bool is_mixed() const override { return is_mixed_; }
		int n_incremental_load_steps(const double diag) const override;

		void add_dirichlet_boundary(const int id, const Eigen::RowVector3d &val, const bool isx, const bool isy, const bool isz);
		void add_neumann_boundary(const int id, const Eigen::RowVector3d &val);
		void add_pressure_boundary(const int id, const double val);

		void add_dirichlet_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z)> &func, const bool isx, const bool isy, const bool isz);
		void add_neumann_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z)> &func);
		void add_pressure_boundary(const int id, const std::function<double(double x, double y, double z)> &func);

		void set_rhs(double x, double y, double z);

		void clear() override;

	private:
		bool all_dimentions_dirichelt_ = true;
		bool has_exact_ = false;
		bool has_exact_grad_ = false;
		bool is_time_dept_ = false;
		// bool is_mixed_ = false;

		std::vector<std::array<ExpressionValue, 3>> forces_;
		std::vector<std::array<ExpressionValue, 3>> displacements_;
		std::vector<ExpressionValue> pressures_;

		std::array<ExpressionValue, 3> initial_position_;
		std::array<ExpressionValue, 3> initial_velocity_;
		std::array<ExpressionValue, 3> initial_acceleration_;

		std::vector<Eigen::Matrix<bool, 1, 3>> dirichelt_dimentions_;

		std::array<ExpressionValue, 3> rhs_;
		std::array<ExpressionValue, 3> exact_;
		std::array<ExpressionValue, 9> exact_grad_;
		bool is_all_;
	};

	class GenericScalarProblem : public Problem
	{
	public:
		GenericScalarProblem(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return rhs_.is_zero(); }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return has_exact_; }
		bool is_scalar() const override { return true; }
		bool is_time_dependent() const override { return is_time_dept_; }
		bool is_linear_in_time() const override { return !is_time_dept_; }

		void set_parameters(const json &params) override;

		void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		void add_dirichlet_boundary(const int id, const double val);
		void add_neumann_boundary(const int id, const double val);

		void add_dirichlet_boundary(const int id, const std::function<double(double x, double y, double z)> &func);
		void add_neumann_boundary(const int id, const std::function<double(double x, double y, double z)> &func);

		void clear() override;

	private:
		std::vector<ExpressionValue> neumann_;
		std::vector<ExpressionValue> dirichlet_;

		ExpressionValue rhs_;
		ExpressionValue exact_;
		std::array<ExpressionValue, 3> exact_grad_;
		bool is_all_;
		bool has_exact_ = false;
		bool has_exact_grad_ = false;
		bool is_time_dept_ = false;
	};
} // namespace polyfem
