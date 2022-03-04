#pragma once

#include <polyfem/Problem.hpp>
#include <polyfem/ExpressionValue.hpp>

#include <Eigen/Dense>

#include <array>
#include <vector>

namespace polyfem
{
	class Interpolation
	{
	public:
		virtual ~Interpolation() {}
		virtual double eval(const double t) const = 0;
		virtual void init(const json &params) {}

		static std::shared_ptr<Interpolation> build(const json &params);
	};

	class NoInterpolation : public Interpolation
	{
	public:
		double eval(const double t) const override { return 1; };
	};

	class LinearInterpolation : public Interpolation
	{
	public:
		double eval(const double t) const override { return t; }
	};

	class LinearRamp : public Interpolation
	{
	public:
		double eval(const double t) const override;
		void init(const json &params) override;

	private:
		double to_;
		double form_;
	};

	class GenericTensorProblem : public Problem
	{
	public:
		GenericTensorProblem(const std::string &name);

		void rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
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
		void set_time_dependent(const bool val) { is_time_dept_ = val; }
		bool is_constant_in_time() const override { return !is_time_dept_; }
		bool might_have_no_dirichlet() override { return !is_all_; }

		void velocity_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void acceleration_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		void initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void initial_velocity(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
		void initial_acceleration(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

		void set_parameters(const json &params) override;

		bool is_dimension_dirichet(const int tag, const int dim) const override;
		bool all_dimensions_dirichlet() const override { return all_dimensions_dirichlet_; }

		void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		// bool is_mixed() const override { return is_mixed_; }
		int n_incremental_load_steps(const double diag) const override;

		void add_dirichlet_boundary(const int id, const Eigen::RowVector3d &val, const bool isx, const bool isy, const bool isz, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void add_neumann_boundary(const int id, const Eigen::RowVector3d &val, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void add_pressure_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());

		void update_dirichlet_boundary(const int id, const Eigen::RowVector3d &val, const bool isx, const bool isy, const bool isz, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void update_neumann_boundary(const int id, const Eigen::RowVector3d &val, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void update_pressure_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());

		void add_dirichlet_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const bool isx, const bool isy, const bool isz, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void add_neumann_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void add_pressure_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());

		void update_dirichlet_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const bool isx, const bool isy, const bool isz, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void update_neumann_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void update_pressure_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());

		void add_dirichlet_boundary(const int id, const json &val, const bool isx, const bool isy, const bool isz, const std::string &interpolation = "");
		void add_neumann_boundary(const int id, const json &val, const std::string &interpolation = "");
		void add_pressure_boundary(const int id, json val, const std::string &interpolation = "");

		void update_dirichlet_boundary(const int id, const json &val, const bool isx, const bool isy, const bool isz, const std::string &interpolation = "");
		void update_neumann_boundary(const int id, const json &val, const std::string &interpolation = "");
		void update_pressure_boundary(const int id, json val, const std::string &interpolation = "");

		void set_rhs(double x, double y, double z);

		void clear() override;

	private:
		bool all_dimensions_dirichlet_ = true;
		bool has_exact_ = false;
		bool has_exact_grad_ = false;
		bool is_time_dept_ = false;
		// bool is_mixed_ = false;

		std::vector<std::array<ExpressionValue, 3>> forces_;
		std::vector<std::shared_ptr<Interpolation>> forces_interpolation_;
		std::vector<std::array<ExpressionValue, 3>> displacements_;
		std::vector<std::shared_ptr<Interpolation>> displacements_interpolation_;
		std::vector<ExpressionValue> pressures_;
		std::vector<std::shared_ptr<Interpolation>> pressure_interpolation_;

		std::vector<std::pair<int, std::array<ExpressionValue, 3>>> initial_position_;
		std::vector<std::pair<int, std::array<ExpressionValue, 3>>> initial_velocity_;
		std::vector<std::pair<int, std::array<ExpressionValue, 3>>> initial_acceleration_;

		std::vector<Eigen::Matrix<bool, 1, 3>> dirichlet_dimensions_;

		std::array<ExpressionValue, 3> rhs_;
		std::array<ExpressionValue, 3> exact_;
		std::array<ExpressionValue, 9> exact_grad_;
		bool is_all_;
	};

	class GenericScalarProblem : public Problem
	{
	public:
		GenericScalarProblem(const std::string &name);

		void rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		bool is_rhs_zero() const override { return rhs_.is_zero(); }

		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void neumann_bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return has_exact_; }
		bool is_scalar() const override { return true; }
		bool is_time_dependent() const override { return is_time_dept_; }
		void set_time_dependent(const bool val) { is_time_dept_ = val; }
		bool is_constant_in_time() const override { return !is_time_dept_; }
		bool might_have_no_dirichlet() override { return !is_all_; }

		void set_parameters(const json &params) override;

		void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		void add_dirichlet_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void add_neumann_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());

		void update_dirichlet_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void update_neumann_boundary(const int id, const double val, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());

		void add_dirichlet_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void add_neumann_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());

		void update_dirichlet_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());
		void update_neumann_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<Interpolation> &interp = std::make_shared<NoInterpolation>());

		void add_dirichlet_boundary(const int id, const json &val, const std::string &interp = "");
		void add_neumann_boundary(const int id, const json &val, const std::string &interp = "");

		void update_dirichlet_boundary(const int id, const json &val, const std::string &interp = "");
		void update_neumann_boundary(const int id, const json &val, const std::string &interp = "");

		void clear() override;

	private:
		std::vector<ExpressionValue> neumann_;
		std::vector<ExpressionValue> dirichlet_;

		std::vector<std::shared_ptr<Interpolation>> neumann_interpolation_;
		std::vector<std::shared_ptr<Interpolation>> dirichlet_interpolation_;

		ExpressionValue rhs_;
		ExpressionValue exact_;
		std::array<ExpressionValue, 3> exact_grad_;
		bool is_all_;
		bool has_exact_ = false;
		bool has_exact_grad_ = false;
		bool is_time_dept_ = false;
	};
} // namespace polyfem
