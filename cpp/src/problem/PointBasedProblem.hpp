#pragma once

#include "Problem.hpp"
#include "InterpolatedFunction.hpp"


#include <vector>
#include <Eigen/Dense>

namespace poly_fem
{
	class PointBasedTensorProblem: public Problem
	{
	private:
		class BCValue
		{
		public:
			BCValue()
			{
				init(0, 0, 0);
			}

			Eigen::RowVector3d operator()(const Eigen::RowVector3d &pt) const;

			void init(const json &data);

			void init(const double x, const double y, const double z)
			{
				this->val << x, y, z;
				is_val = true;
			}

			void init(const Eigen::Vector3d &v)
			{
				this->val = v;
				is_val = true;
			}

			void init(const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tri, const Eigen::MatrixXd &fun, const int coord)
			{
				func.init(fun, pts, tri);

				coordiante_0 = (coord + 1) % 3;
				coordiante_1 = (coord + 2) % 3;

				is_val = false;
			}
		private:
			Eigen::Vector3d val;
			InterpolatedFunction2d func;
			bool is_val;
			int coordiante_0 = 0;
			int coordiante_1 = 1;
		};
	public:
		PointBasedTensorProblem(const std::string &name);

		void rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;
		void bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const override;

		bool has_exact_sol() const override { return false; }
		bool is_scalar() const override { return false; }

		void set_parameters(const json &params) override;
		
		void add_constant(const int bc_tag, const Eigen::Vector3d &value);
		void add_function(const int bc_tag, const Eigen::MatrixXd &func, const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tri, const int coord);
	private:
		bool initialized_ = false;
		double rhs_;
		double scaling_;
		Eigen::Vector3d translation_;
		std::vector<BCValue> bc_;
	};
}

