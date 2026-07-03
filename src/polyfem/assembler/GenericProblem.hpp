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
			int fe_space_id = -1;
			int size = 0;
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
			int fe_space_id = -1;
			utils::ExpressionValue value;
			std::shared_ptr<utils::Interpolation> interpolation;

			void set_unit_type(const std::string &unit_type)
			{
				value.set_unit_type(unit_type);
			}

			double eval(const RowVectorNd &pts, const double t) const;
		};

		struct TensorInitialValue
		{
			int body_id = -1;
			int fe_space_id = -1;
			int size = 0;
			std::array<utils::ExpressionValue, 3> value;
		};

		struct ScalarInitialValue
		{
			int body_id = -1;
			int fe_space_id = -1;
			utils::ExpressionValue value;
		};

		class GenericTensorProblem : public Problem
		{
		public:
			GenericTensorProblem(const std::string &name);
			void set_units(const assembler::Assembler &assembler, const Units &units) override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			bool is_rhs_zero(const int fe_space_id = -1) const override;

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			void neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			void pressure_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const override;
			double pressure_cavity_bc(const int boundary_id, const double t) const override;

			void dirichlet_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const double t, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			void neumann_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const Eigen::MatrixXd &normal, const double t, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			bool is_nodal_dirichlet_boundary(const int n_id, const int tag, const int fe_space_id = -1) override;
			bool is_nodal_neumann_boundary(const int n_id, const int tag, const int fe_space_id = -1) override;
			bool has_nodal_dirichlet(const int fe_space_id = -1) override;
			bool has_nodal_neumann(const int fe_space_id = -1) override;
			bool is_nodal_dimension_dirichlet(const int n_id, const int tag, const int dim, const int fe_space_id = -1) const override;
			void update_nodes(const Eigen::VectorXi &in_node_to_node) override;

			bool has_exact_sol() const override { return has_exact_; }
			bool is_scalar() const override { return false; }
			bool is_time_dependent() const override { return is_time_dept_; }
			void set_time_dependent(const bool val) { is_time_dept_ = val; }
			bool is_constant_in_time() const override { return !is_time_dept_; }
			bool might_have_no_dirichlet() override { return !is_all_; }

			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			void initial_velocity(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			void initial_acceleration(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;

			void set_parameters(const json &params, const std::string &root_path) override;

			bool is_dimension_dirichet(const int tag, const int dim, const int fe_space_id = -1) const override;
			bool all_dimensions_dirichlet(const int fe_space_id) const override;

			void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void clear() override;

			void update_dirichlet_boundary(const int id, const int time_step, const Eigen::VectorXd &val);
			void update_dirichlet_nodes(const Eigen::VectorXi &in_node_to_node, const Eigen::VectorXi &node_ids, const Eigen::MatrixXd &nodal_dirichlet);
			void update_pressure_boundary(const int id, const int time_step, const double val);

		private:
			bool has_exact_ = false;
			bool has_exact_grad_ = false;
			bool is_time_dept_ = false;
			// bool is_mixed_ = false;

			std::vector<TensorBCValue> forces_;
			std::vector<TensorBCValue> displacements_;
			std::vector<ScalarBCValue> normal_aligned_forces_;
			std::vector<ScalarBCValue> pressures_;
			std::unordered_map<int, ScalarBCValue> cavity_pressures_;

			std::vector<TensorInitialValue> initial_position_;
			std::vector<TensorInitialValue> initial_velocity_;
			std::vector<TensorInitialValue> initial_acceleration_;

			std::map<int, std::array<utils::ExpressionValue, 3>> rhs_;
			std::map<int, int> rhs_size_;
			std::array<utils::ExpressionValue, 3> exact_;
			std::array<utils::ExpressionValue, 9> exact_grad_;

			std::map<int, TensorBCValue> nodal_dirichlet_;
			std::map<int, TensorBCValue> nodal_neumann_;
			std::vector<Eigen::MatrixXd> nodal_dirichlet_mat_;

			bool is_all_;

		protected:
			bool has_boundary(const BoundaryKind kind, const int tag, const int fe_space_id) override;
		};

		class GenericScalarProblem : public Problem
		{
		public:
			GenericScalarProblem(const std::string &name);
			void set_units(const assembler::Assembler &assembler, const Units &units) override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			bool is_rhs_zero(const int fe_space_id = -1) const override;

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			void neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			void initial_solution(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;

			void dirichlet_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const double t, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			void neumann_nodal_value(const mesh::Mesh &mesh, const int node_id, const RowVectorNd &pt, const Eigen::MatrixXd &normal, const double t, Eigen::MatrixXd &val, const int fe_space_id = -1) const override;
			bool is_nodal_dirichlet_boundary(const int n_id, const int tag, const int fe_space_id = -1) override;
			bool is_nodal_neumann_boundary(const int n_id, const int tag, const int fe_space_id = -1) override;
			bool has_nodal_dirichlet(const int fe_space_id = -1) override;
			bool has_nodal_neumann(const int fe_space_id = -1) override;
			void update_nodes(const Eigen::VectorXi &in_node_to_node) override;

			bool has_exact_sol() const override { return has_exact_; }
			bool is_scalar() const override { return true; }
			bool is_time_dependent() const override { return is_time_dept_; }
			void set_time_dependent(const bool val) { is_time_dept_ = val; }
			bool is_constant_in_time() const override { return !is_time_dept_; }
			bool might_have_no_dirichlet() override { return !is_all_; }

			void set_parameters(const json &params, const std::string &root_path) override;

			void exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

			void clear() override;

		private:
			std::vector<ScalarBCValue> neumann_;
			std::vector<ScalarBCValue> dirichlet_;
			std::vector<ScalarInitialValue> initial_solution_;

			std::map<int, ScalarBCValue> nodal_dirichlet_;
			std::map<int, ScalarBCValue> nodal_neumann_;
			std::vector<Eigen::MatrixXd> nodal_dirichlet_mat_;

			std::map<int, utils::ExpressionValue> rhs_;
			utils::ExpressionValue exact_;
			std::array<utils::ExpressionValue, 3> exact_grad_;
			bool is_all_;
			bool has_exact_ = false;
			bool has_exact_grad_ = false;
			bool is_time_dept_ = false;

		protected:
			bool has_boundary(const BoundaryKind kind, const int tag, const int fe_space_id) override;
		};
	} // namespace assembler
} // namespace polyfem
