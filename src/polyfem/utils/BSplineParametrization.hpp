#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

#include <nanospline/BSpline.h>
#include <nanospline/BSplinePatch.h>

namespace polyfem
{
	class BSplineParametrization
	{
	public:
		BSplineParametrization(const Eigen::MatrixXd &V) { num_vertices = V.rows(); }
		virtual ~BSplineParametrization() = default;

		virtual int vertex_size() = 0;
		virtual void reparametrize(const Eigen::MatrixXd &control_points, Eigen::MatrixXd &newV) = 0;
		// virtual void get_parameters(const Eigen::MatrixXd &V, Eigen::MatrixXd &control_points) final;
		virtual void get_parameters(const Eigen::MatrixXd &V, Eigen::MatrixXd &control_points, const bool mesh_changed) = 0;
		virtual void derivative_wrt_params(const Eigen::VectorXd &grad_boundary, Eigen::VectorXd &grad_control_points) = 0;

	protected:
		int num_vertices;
	};

	class BSplineParametrization2D : public BSplineParametrization
	{
	public:
		BSplineParametrization2D(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &knots, const Eigen::MatrixXd &V);

		int vertex_size() override { return node_id_to_t_.size(); }

		void reparametrize(const Eigen::MatrixXd &control_points, Eigen::MatrixXd &newV) override;

		// Assume the connectivity has not changed. Does not work with remeshing
		void get_parameters(const Eigen::MatrixXd &V, Eigen::MatrixXd &control_points, const bool mesh_changed) override;

		void derivative_wrt_params(const Eigen::VectorXd &grad_boundary, Eigen::VectorXd &grad_control_points) override;

		static void gradient(const Eigen::MatrixXd &point, const Eigen::MatrixXd &control_points, const double t_parameter, const double distance, Eigen::MatrixXd &grad);
		static void eval(const Eigen::MatrixXd &control_points, const double t, Eigen::MatrixXd &val);
		static void deriv(const Eigen::MatrixXd &control_points, const double t, Eigen::MatrixXd &val);

	private:
		std::vector<int> node_ids_;
		std::map<int, double> node_id_to_t_;
		const int dim;
		nanospline::BSpline<double, 2, 3> curve;
	};

	class BSplineParametrization3D : public BSplineParametrization
	{
	public:
		BSplineParametrization3D(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &knots_u, const Eigen::MatrixXd &knots_v, const Eigen::MatrixXd &V);

		int vertex_size() override { return node_id_to_param_.size(); }

		// Assume the connectivity has not changed. Does not work with remeshing
		void get_parameters(const Eigen::MatrixXd &V, Eigen::MatrixXd &control_points, const bool mesh_changed) override;

		void reparametrize(const Eigen::MatrixXd &control_points, Eigen::MatrixXd &newV) override;

		void derivative_wrt_params(const Eigen::VectorXd &grad_boundary, Eigen::VectorXd &grad_control_points) override;

		static void gradient(const Eigen::MatrixXd &point, const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &uv_parameter, const double distance, Eigen::MatrixXd &grad);
		static void eval(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &uv_parameter, Eigen::MatrixXd &val);
		static void deriv(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &uv_parameter, Eigen::MatrixXd &deriv_u, Eigen::MatrixXd &deriv_v);

	private:
		std::vector<int> node_ids_;
		std::map<int, Eigen::MatrixXd> node_id_to_param_;
		const int dim;
		nanospline::BSplinePatch<double, 3, 3, 3> patch;
	};

} // namespace polyfem