#pragma once

#include <polyfem/assembler/Problem.hpp>
#include <polyfem/utils/InterpolatedFunction.hpp>
#include <polyfem/utils/RBFInterpolation.hpp>

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	namespace problem
	{
		class NodeValues
		{
		public:
			NodeValues();

			void load(const std::string &path);
			void init(const mesh::Mesh &mesh);

			double dirichlet_interpolate(const int p_id, const Eigen::MatrixXd &uv) const
			{
				return interpolate(p_id, uv, true);
			}
			double neumann_interpolate(const int p_id, const Eigen::MatrixXd &uv) const
			{
				return interpolate(p_id, uv, false);
			}

		private:
			double interpolate(const int p_id, const Eigen::MatrixXd &uv, bool is_dirichlet) const;

			std::vector<int> raw_ids_;
			std::vector<std::vector<double>> raw_data_;
			std::vector<bool> raw_dirichlet_;

			std::vector<Eigen::VectorXd> data_;
			std::vector<bool> dirichlet_;
		};

		class NodeProblem : public assembler::Problem
		{
		public:
			NodeProblem(const std::string &name);
			void init(const mesh::Mesh &mesh) override;

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return abs(rhs_) < 1e-10; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const override;

			bool has_exact_sol() const override { return false; }
			bool is_scalar() const override { return true; }

			void set_parameters(const json &params) override;

			bool is_dimension_dirichet(const int tag, const int dim) const override;
			bool all_dimensions_dirichlet() const override { return all_dimensions_dirichlet_; }

		private:
			bool all_dimensions_dirichlet_ = true;
			std::vector<Eigen::Matrix<bool, 1, 3, Eigen::RowMajor>> dirichlet_dimensions_;
			double rhs_;
			NodeValues values_;
			bool is_all_;
		};
	} // namespace problem
} // namespace polyfem
