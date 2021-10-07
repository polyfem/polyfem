#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/GenericProblem.hpp>
#include <polyfem/ExpressionValue.hpp>

#include <Eigen/Dense>

namespace polyfem
{
	class Obstacle
	{
	public:
		void init(const json &json, const std::string &root_path);

		inline int n_vertices() const { return v_.rows(); }
		inline const Eigen::MatrixXd &v() const { return v_; }
		inline const Eigen::MatrixXi &f() const { return f_; }
		inline const Eigen::MatrixXi &e() const { return e_; }
		inline const Eigen::MatrixXi &f_2_e() const { return f_2_e_; }

		inline const Eigen::MatrixXi get_face_connectivity() const { return in_f_; }
		inline const Eigen::MatrixXi get_edge_connectivity() const { return in_e_; }
		inline const Eigen::MatrixXi get_vertex_connectivity() const { return in_v_; }

		void update_displacement(const double t, Eigen::MatrixXd &sol) const;
		void set_zero(Eigen::MatrixXd &sol) const;

		void clear();

	private:
		int dim_;
		Eigen::MatrixXd v_;
		Eigen::MatrixXi f_;
		Eigen::MatrixXi e_;
		Eigen::MatrixXi f_2_e_;

		Eigen::MatrixXi in_v_;
		Eigen::MatrixXi in_f_;
		Eigen::MatrixXi in_e_;

		std::vector<std::array<ExpressionValue, 3>> displacements_;
		std::vector<std::shared_ptr<Interpolation>> displacements_interpolation_;

		std::vector<int> endings_;
	};
} // namespace polyfem