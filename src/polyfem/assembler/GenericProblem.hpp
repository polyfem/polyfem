#pragma once

#include <polyfem/assembler/Problem.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/Interpolation.hpp>

namespace polyfem
{
	namespace assembler
	{
		struct TensorBCValue
		{
			std::array<utils::ExpressionValue, 3> value;
			std::vector<std::shared_ptr<utils::Interpolation>> interpolation;
			Eigen::Matrix<bool, 1, 3> dirichlet_dimension;

			void set_unit_type(const std::string &unit_type)
			{
				for (auto &v : value)
					v.set_unit_type(unit_type);
			}

			double eval(const RowVectorNd &pts, const int dim, const double t, const int el_id = -1) const;

		};

		struct ScalarBCValue
		{
			utils::ExpressionValue value;
			std::shared_ptr<utils::Interpolation> interpolation;

			void set_unit_type(const std::string &unit_type)
			{
				value.set_unit_type(unit_type);
			}
      
			double eval(const RowVectorNd &pts, const double t) const;
		};

		class GenericTensorProblem : public Problem
		{
		public:
			GenericTensorProblem(const std::string &name);
			void set_units(const assembler::Assembler &assembler, const Units &units) override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override
			{
				for (int i = 0; i < 3; ++i)
				{
					if (!rhs_[i].is_zero())
						return false;
				}
				return true;
			}

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const override;

			void dirichlet_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const double t, Eigen::MatrixXd &val) const override;
			void neumann_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const Eigen::MatrixXd &normal, const double t, Eigen::MatrixXd &val) const override;
			bool is_nodal_dirichlet_boundary(const int n_id, const int tag) override;
			bool is_nodal_neumann_boundary(const int n_id, const int tag) override;
			bool has_nodal_dirichlet() override;
			bool has_nodal_neumann() override;
			bool is_nodal_dimension_dirichlet(const int n_id, const int tag, const int dim) const override;
			void update_nodes(const Eigen::VectorXi &in_node_to_node) override;

			bool has_exact_sol() const override { return has_exact_; }
			bool is_scalar() const override { return false; }
			bool is_time_dependent() const override { return is_time_dept_; }
			void set_time_dependent(const bool val) { is_time_dept_ = val; }
			bool is_constant_in_time() const override { return !is_time_dept_; }
			bool might_have_no_dirichlet() override { return !is_all_; }

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
			void initial_velocity(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;
			void initial_acceleration(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			void set_parameters(const json &params) override;

			bool is_dimension_dirichet(const int tag, const int dim) const override;
			bool all_dimensions_dirichlet() const override { return all_dimensions_dirichlet_; }

			void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void add_dirichlet_boundary(const int id, const Eigen::RowVector3d &val, const bool isx, const bool isy, const bool isz, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void add_neumann_boundary(const int id, const Eigen::RowVector3d &val, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void add_pressure_boundary(const int id, const double val, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());

			void update_dirichlet_boundary(const int id, const Eigen::RowVector3d &val, const bool isx, const bool isy, const bool isz, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void update_neumann_boundary(const int id, const Eigen::RowVector3d &val, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void update_pressure_boundary(const int id, const double val, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());

			void add_dirichlet_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const bool isx, const bool isy, const bool isz, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void add_neumann_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void add_pressure_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());

			void update_dirichlet_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const bool isx, const bool isy, const bool isz, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void update_neumann_boundary(const int id, const std::function<Eigen::MatrixXd(double x, double y, double z, double t)> &func, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void update_pressure_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());

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

			std::vector<TensorBCValue> forces_;
			std::vector<TensorBCValue> displacements_;
			std::vector<ScalarBCValue> pressures_;

			std::vector<std::pair<int, std::array<utils::ExpressionValue, 3>>> initial_position_;
			std::vector<std::pair<int, std::array<utils::ExpressionValue, 3>>> initial_velocity_;
			std::vector<std::pair<int, std::array<utils::ExpressionValue, 3>>> initial_acceleration_;

			std::array<utils::ExpressionValue, 3> rhs_;
			std::array<utils::ExpressionValue, 3> exact_;
			std::array<utils::ExpressionValue, 9> exact_grad_;

			std::map<int, TensorBCValue> nodal_dirichlet_;
			std::map<int, TensorBCValue> nodal_neumann_;
			std::vector<Eigen::MatrixXd> nodal_dirichlet_mat_;

			bool is_all_;
		};

		class GenericScalarProblem : public Problem
		{
		public:
			GenericScalarProblem(const std::string &name);
			void set_units(const assembler::Assembler &assembler, const Units &units) override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return rhs_.is_zero(); }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const override;
			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const override;

			void dirichlet_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const double t, Eigen::MatrixXd &val) const override;
			void neumann_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const Eigen::MatrixXd &normal, const double t, Eigen::MatrixXd &val) const override;
			bool is_nodal_dirichlet_boundary(const int n_id, const int tag) override;
			bool is_nodal_neumann_boundary(const int n_id, const int tag) override;
			bool has_nodal_dirichlet() override;
			bool has_nodal_neumann() override;
			void update_nodes(const Eigen::VectorXi &in_node_to_node) override;

			bool has_exact_sol() const override { return has_exact_; }
			bool is_scalar() const override { return true; }
			bool is_time_dependent() const override { return is_time_dept_; }
			void set_time_dependent(const bool val) { is_time_dept_ = val; }
			bool is_constant_in_time() const override { return !is_time_dept_; }
			bool might_have_no_dirichlet() override { return !is_all_; }

			void set_parameters(const json &params) override;

			void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void add_dirichlet_boundary(const int id, const double val, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void add_neumann_boundary(const int id, const double val, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());

			void update_dirichlet_boundary(const int id, const double val, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void update_neumann_boundary(const int id, const double val, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());

			void add_dirichlet_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void add_neumann_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());

			void update_dirichlet_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());
			void update_neumann_boundary(const int id, const std::function<double(double x, double y, double z, double t)> &func, const std::shared_ptr<utils::Interpolation> &interp = std::make_shared<utils::NoInterpolation>());

			void add_dirichlet_boundary(const int id, const json &val, const std::string &interp = "");
			void add_neumann_boundary(const int id, const json &val, const std::string &interp = "");

			void update_dirichlet_boundary(const int id, const json &val, const std::string &interp = "");
			void update_neumann_boundary(const int id, const json &val, const std::string &interp = "");

			void clear() override;

		private:
			std::vector<ScalarBCValue> neumann_;
			std::vector<ScalarBCValue> dirichlet_;
			std::vector<std::pair<int, utils::ExpressionValue>> initial_solution_;

			std::map<int, ScalarBCValue> nodal_dirichlet_;
			std::map<int, ScalarBCValue> nodal_neumann_;
			std::vector<Eigen::MatrixXd> nodal_dirichlet_mat_;

			utils::ExpressionValue rhs_;
			utils::ExpressionValue exact_;
			std::array<utils::ExpressionValue, 3> exact_grad_;
			bool is_all_;
			bool has_exact_ = false;
			bool has_exact_grad_ = false;
			bool is_time_dept_ = false;
		};
	} // namespace assembler
} // namespace polyfem
