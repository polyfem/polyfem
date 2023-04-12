#pragma once

#include <polyfem/assembler/Problem.hpp>
#include <polyfem/utils/InterpolatedFunction.hpp>
#include <polyfem/utils/RBFInterpolation.hpp>

#include <polyfem/Common.hpp>

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{
	namespace problem
	{
		class PointBasedTensorProblem : public assembler::Problem
		{
		private:
			class BCValue
			{
			public:
				BCValue()
				{
					Eigen::Matrix<bool, 3, 1> dd;
					dd.setConstant(true);
					init(0, 0, 0, dd);
				}

				Eigen::RowVector3d operator()(const Eigen::RowVector3d &pt) const;

				bool init(const json &data);

				void init(const double x, const double y, const double z, const Eigen::Matrix<bool, 3, 1> &dd)
				{
					this->val << x, y, z;
					dirichlet_dims = dd;
					is_val = true;
				}

				void init(const Eigen::Vector3d &v, const Eigen::Matrix<bool, 3, 1> &dd)
				{
					this->val = v;
					dirichlet_dims = dd;
					is_val = true;
				}

				void init(const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tri, const Eigen::MatrixXd &fun, const int coord, const Eigen::Matrix<bool, 3, 1> &dd)
				{
					tri_func.init(fun, pts, tri);
					is_tri = true;

					coordiante_0 = (coord + 1) % 3;
					coordiante_1 = (coord + 2) % 3;

					dirichlet_dims = dd;

					is_val = false;
				}

				void init(const Eigen::MatrixXd &pts, const Eigen::MatrixXd &fun, const std::string &rbf, const double eps, const int coord, const Eigen::Matrix<bool, 3, 1> &dd)
				{
					rbf_func.init(fun, pts, rbf, eps);
					is_tri = false;

					if (coord >= 0)
					{
						coordiante_0 = (coord + 1) % 3;
						coordiante_1 = (coord + 2) % 3;
					}
					else
					{
						coordiante_0 = -1;
						coordiante_1 = -1;
					}

					dirichlet_dims = dd;

					is_val = false;
				}

				bool is_dirichet_dim(const int d) const { return dirichlet_dims(d); }

			private:
				Eigen::Vector3d val;
				utils::InterpolatedFunction2d tri_func;
				utils::RBFInterpolation rbf_func;
				bool is_val;
				bool is_tri;
				int coordiante_0 = 0;
				int coordiante_1 = 1;
				Eigen::Matrix<bool, 3, 1> dirichlet_dims;
			};

		public:
			PointBasedTensorProblem(const std::string &name);

			void rhs(const assembler::Assembler &assembler, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			bool is_rhs_zero() const override { return abs(rhs_) < 1e-10; }

			void dirichlet_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
			void neumann_bc(const mesh::Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &normals, const double t, Eigen::MatrixXd &val) const override;

			bool has_exact_sol() const override { return false; }
			bool is_scalar() const override { return false; }

			void set_parameters(const json &params) override;

			void add_constant(const int bc_tag, const Eigen::Vector3d &value)
			{
				Eigen::Matrix<bool, 3, 1> dd;
				dd.setConstant(true);
				add_constant(bc_tag, value, dd);
			}

			void add_function(const int bc_tag, const Eigen::MatrixXd &func, const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tri, const int coord)
			{
				Eigen::Matrix<bool, 3, 1> dd;
				dd.setConstant(true);
				add_function(bc_tag, func, pts, tri, coord, dd);
			}

			void add_function(const int bc_tag, const Eigen::MatrixXd &func, const Eigen::MatrixXd &pts, const std::string &rbf, const double eps, const int coord)
			{
				Eigen::Matrix<bool, 3, 1> dd;
				dd.setConstant(true);
				add_function(bc_tag, func, pts, rbf, eps, coord, dd);
			}

			void add_constant(const int bc_tag, const Eigen::Vector3d &value, const Eigen::Matrix<bool, 3, 1> &dd, const bool is_neumann = false);
			void add_function(const int bc_tag, const Eigen::MatrixXd &func, const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tri, const int coord, const Eigen::Matrix<bool, 3, 1> &dd, const bool is_neumann = false);
			void add_function(const int bc_tag, const Eigen::MatrixXd &func, const Eigen::MatrixXd &pts, const std::string &rbf, const double eps, const int coord, const Eigen::Matrix<bool, 3, 1> &dd, const bool is_neumann = false);

			bool is_dimension_dirichet(const int tag, const int dim) const override;
			bool all_dimensions_dirichlet() const override { return all_dimensions_dirichlet_; }

		private:
			bool initialized_ = false;
			bool all_dimensions_dirichlet_ = true;
			double rhs_;
			double scaling_;
			Eigen::Vector3d translation_;
			std::vector<BCValue> bc_;
			std::vector<BCValue> neumann_bc_;
		};
	} // namespace problem
} // namespace polyfem
