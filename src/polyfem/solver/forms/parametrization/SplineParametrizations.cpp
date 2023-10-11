#include "SplineParametrizations.hpp"
#include <polyfem/utils/BSplineParametrization.hpp>
#include <polyfem/State.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <igl/bbw.h>
#include <igl/boundary_conditions.h>
#include <igl/normalize_row_sums.h>
#include <igl/boundary_loop.h>
#include <igl/exact_geodesic.h>
#include <igl/bounding_box.h>
#include <igl/writeOBJ.h>

#include <polyfem/utils/AutodiffTypes.hpp>

#include <unsupported/Eigen/SparseExtra>

#include <unordered_map>

namespace polyfem::solver
{
	namespace
	{
		typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> Diff;

		template <typename T>
		Eigen::Matrix<T, 3, 3> rotation_matrix(const T x_angle, const T y_angle, const T z_angle)
		{
			Eigen::Matrix<T, 3, 3> x_rot;
			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					x_rot(i, j) = T(0);
			x_rot(0, 0) = 1.;
			x_rot(1, 1) = cos(x_angle);
			x_rot(1, 2) = -sin(x_angle);
			x_rot(2, 1) = sin(x_angle);
			x_rot(2, 2) = cos(x_angle);
			Eigen::Matrix<T, 3, 3> y_rot;
			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					y_rot(i, j) = T(0);
			y_rot(1, 1) = 1.;
			y_rot(0, 0) = cos(y_angle);
			y_rot(0, 2) = sin(y_angle);
			y_rot(2, 0) = -sin(y_angle);
			y_rot(2, 2) = cos(y_angle);
			Eigen::Matrix<T, 3, 3> z_rot;
			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					z_rot(i, j) = T(0);
			z_rot(2, 2) = 1.;
			z_rot(0, 0) = cos(z_angle);
			z_rot(0, 1) = -sin(z_angle);
			z_rot(1, 0) = sin(z_angle);
			z_rot(1, 1) = cos(z_angle);

			return z_rot * y_rot * x_rot;
		}

		template <typename T>
		Eigen::Matrix<T, 3, 1> affine_transformation(const Eigen::VectorXd &point, const Eigen::Matrix<T, 6, 1> &param)
		{
			Eigen::Matrix<T, 3, 3> rot = rotation_matrix(param(0), param(1), param(2));
			Eigen::Matrix<T, 3, 1> transformed_point(3);
			for (int i = 0; i < 3; ++i)
			{
				T val(0);
				for (int j = 0; j < 3; ++j)
					val = val + rot(j, i) * point(j);
				transformed_point(i) = val + param(3 + i);
			}
			return transformed_point;
		}

		Eigen::MatrixXd grad_affine_transformation(const Eigen::VectorXd &point, const Eigen::VectorXd &param)
		{
			DiffScalarBase::setVariableCount(6);
			Eigen::Matrix<Diff, 6, 1> full_diff(6, 1);
			for (int i = 0; i < 6; i++)
				full_diff(i) = Diff(i, param(i));
			auto reduced_diff = affine_transformation(point, full_diff);

			Eigen::MatrixXd grad(3, 6);
			for (int i = 0; i < 3; ++i)
				grad.row(i) = reduced_diff[i].getGradient();

			return grad;
		}
	} // namespace
	Eigen::VectorXd BSplineParametrization1DTo2D::inverse_eval(const Eigen::VectorXd &y)
	{
		spline_ = std::make_shared<BSplineParametrization2D>(initial_control_points_, knots_, utils::unflatten(y, 2));
		invoked_inverse_eval_ = true;
		assert(size_ == spline_->vertex_size());
		if (exclude_ends_)
			return utils::flatten(initial_control_points_).segment(2, (initial_control_points_.rows() - 2) * 2);
		else
			return utils::flatten(initial_control_points_);
	}

	Eigen::VectorXd BSplineParametrization1DTo2D::eval(const Eigen::VectorXd &x) const
	{
		if (!invoked_inverse_eval_)
			log_and_throw_error("Must call inverse eval on this parametrization first!");
		Eigen::MatrixXd new_control_points;
		if (exclude_ends_)
		{
			new_control_points = initial_control_points_;
			for (int i = 1; i < new_control_points.rows() - 1; ++i)
				new_control_points.row(i) = x.segment(2 * i - 2, 2);
		}
		else
		{
			new_control_points = utils::unflatten(x, 2);
		}
		Eigen::MatrixXd new_vertices;
		spline_->reparametrize(new_control_points, new_vertices);
		Eigen::VectorXd y = utils::flatten(new_vertices);
		return y;
	}

	Eigen::VectorXd BSplineParametrization1DTo2D::apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd grad;
		spline_->derivative_wrt_params(grad_full, grad);
		if (exclude_ends_)
			return grad.segment(2, (initial_control_points_.rows() - 2) * 2);
		else
			return grad;
	}

	Eigen::VectorXd BSplineParametrization2DTo3D::inverse_eval(const Eigen::VectorXd &y)
	{
		spline_ = std::make_shared<BSplineParametrization3D>(initial_control_point_grid_, knots_u_, knots_v_, y);
		invoked_inverse_eval_ = true;
		return Eigen::VectorXd();
	}

	Eigen::VectorXd BSplineParametrization2DTo3D::eval(const Eigen::VectorXd &x) const
	{
		if (!invoked_inverse_eval_)
			log_and_throw_error("Must call inverse eval on this parametrization first!");
		return Eigen::VectorXd();
	}

	Eigen::VectorXd BSplineParametrization2DTo3D::apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const
	{
		return Eigen::VectorXd();
	}

	BoundedBiharmonicWeights2Dto3D::BoundedBiharmonicWeights2Dto3D(const int num_control_vertices, const int num_vertices, const State &state, const bool allow_rotations)
		: num_control_vertices_(num_control_vertices), num_vertices_(num_vertices), allow_rotations_(allow_rotations)
	{
		Eigen::MatrixXd V;
		state.get_vertices(V);

		auto map = state.node_to_primitive();

		int f_size = 0;
		const auto &mesh = state.mesh;
		const auto &bases = state.bases;
		const auto &gbases = state.geom_bases();
		for (const auto &lb : state.total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const int boundary_id = mesh->get_boundary_id(primitive_global_id);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);
				F_surface_.conservativeResize(++f_size, 3);
				for (int f = 0; f < nodes.size(); ++f)
				{
					F_surface_(f_size - 1, f) = map[gbases[e].bases[nodes(f)].global()[0].index];
				}
			}
		}
		V_surface_.resizeLike(V);
		for (int e = 0; e < gbases.size(); e++)
		{
			for (const auto &gbs : gbases[e].bases)
				V_surface_.row(map[gbs.global()[0].index]) = gbs.global()[0].node;
		}
	}

	int BoundedBiharmonicWeights2Dto3D::optimal_new_control_point_idx(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::VectorXi &boundary_loop, const std::vector<int> &existing_points) const
	{
		std::set<int> fixed_vertices;
		{
			for (int i = 0; i < boundary_loop.size(); ++i)
				fixed_vertices.insert(boundary_loop(i));
			for (const auto &i : existing_points)
				fixed_vertices.insert(i);
		}

		Eigen::VectorXi free_vertices(V.rows() - fixed_vertices.size());
		int s = 0;
		for (int i = 0; i < V.rows(); ++i)
			if (fixed_vertices.find(i) == fixed_vertices.end())
				free_vertices(s++) = i;

		Eigen::VectorXi VS, FS, FT;
		VS.resize(fixed_vertices.size());
		s = 0;
		for (const auto &j : fixed_vertices)
			VS(s++) = j;

		Eigen::VectorXd d;
		igl::exact_geodesic(V, F, VS, FS, free_vertices, FT, d);
		int opt_idx = -1;
		double max_min_dist = -1;
		for (int i = 0; i < d.size(); ++i)
		{
			if (opt_idx == -1)
			{
				max_min_dist = d(i);
				opt_idx = free_vertices(i);
				continue;
			}
			else if (d(i) > max_min_dist)
			{
				max_min_dist = d(i);
				opt_idx = free_vertices(i);
			}
		}

		return opt_idx;
	}

	Eigen::VectorXd BoundedBiharmonicWeights2Dto3D::inverse_eval(const Eigen::VectorXd &y)
	{
		y_start = y;

		Eigen::MatrixXd V = utils::unflatten(y, 3);
		Eigen::MatrixXi F;
		compute_faces_for_partial_vertices(V, F);

		Eigen::VectorXi outer_loop;
		std::vector<std::vector<int>> loops;
		igl::boundary_loop(F, loops);
		if (loops.size() == 1)
		{
			outer_loop.resize(loops[0].size());
			for (int i = 0; i < loops[0].size(); ++i)
				outer_loop(i) = loops[0][i];
		}
		else
		{
			logger().error("More than 1 boundary loop! Concatenating and continuing as normal.");
			for (const auto &l : loops)
			{
				int size = outer_loop.size();
				outer_loop.conservativeResize(outer_loop.size() + l.size());
				for (int i = 0; i < l.size(); ++i)
					outer_loop(size + i) = l[i];
			}
		}
		Eigen::MatrixXd V_outer_loop = V(outer_loop, Eigen::all);

		Eigen::MatrixXd point_handles(num_control_vertices_ + outer_loop.size(), 3);
		control_points_.resize(num_control_vertices_, 3);
		std::vector<int> control_indices;
		{
			std::set<int> possible_control_vertices;
			for (int i = 0; i < F.rows(); ++i)
				for (int j = 0; j < F.cols(); ++j)
					possible_control_vertices.insert(F(i, j));
			for (int i = 0; i < outer_loop.size(); ++i)
				possible_control_vertices.erase(outer_loop(i));
			for (int i = 0; i < num_control_vertices_; ++i)
				control_indices.push_back(optimal_new_control_point_idx(V, F, outer_loop, control_indices));

			const int recompute_loops = 5;
			for (int r = 0; r < recompute_loops; ++r)
			{
				for (int i = 0; i < num_control_vertices_; ++i)
				{
					std::vector<int> indices = control_indices;
					indices.erase(indices.begin() + i);
					int new_idx = optimal_new_control_point_idx(V, F, outer_loop, indices);
					control_indices[i] = new_idx;
				}
			}
		}
		for (int i = 0; i < num_control_vertices_; ++i)
			control_points_.row(i) = V.row(control_indices[i]);
		point_handles.block(0, 0, num_control_vertices_, 3) = control_points_;
		point_handles.block(num_control_vertices_, 0, outer_loop.size(), 3) = V_outer_loop;

		igl::writeOBJ("bbw_control_points_" + std::to_string(V.rows()) + ".obj", control_points_, Eigen::MatrixXi::Zero(0, 3));
		igl::writeOBJ("bbw_outer_loop_" + std::to_string(V.rows()) + ".obj", V_outer_loop, Eigen::MatrixXi::Zero(0, 3));

		Eigen::VectorXi b;
		Eigen::MatrixXd bc;
		Eigen::VectorXi point_handles_idx(point_handles.rows());
		for (int i = 0; i < point_handles_idx.size(); ++i)
			point_handles_idx(i) = i;
		igl::boundary_conditions(V, F, point_handles, point_handles_idx, Eigen::VectorXi(), Eigen::VectorXi(), b, bc);

		igl::BBWData bbw_data;
		bbw_data.active_set_params.max_iter = 500;
		bbw_data.verbosity = 1;
		bbw_data.partition_unity = false; // Not implemented in libigl
		Eigen::MatrixXd complete_bbw_weights;
		bool computation = igl::bbw(V, F, b, bc, bbw_data, complete_bbw_weights);
		if (!computation)
			log_and_throw_error("Bounded Bihamonic Weight computation failed!");
		igl::normalize_row_sums(complete_bbw_weights, complete_bbw_weights);
		bbw_weights_ = complete_bbw_weights.block(0, 0, V.rows(), num_control_vertices_).matrix();
		boundary_bbw_weights_ = complete_bbw_weights.block(0, num_control_vertices_, V.rows(), V_outer_loop.rows()).matrix();

		igl::writeOBJ("surface_mesh.obj", V, F);
		Eigen::saveMarket(bbw_weights_.sparseView(0, 1e-8).eval(), "bbw_control_weights.mat");
		Eigen::saveMarket(boundary_bbw_weights_.sparseView(0, 1e-8).eval(), "bbw_boundary_weights.mat");

		invoked_inverse_eval_ = true;

		return Eigen::VectorXd::Zero(num_control_vertices_ * (allow_rotations_ ? 6 : 3));
	}

	Eigen::VectorXd BoundedBiharmonicWeights2Dto3D::eval(const Eigen::VectorXd &x) const
	{
		if (!invoked_inverse_eval_)
			log_and_throw_error("Must call inverse eval on this parametrization first!");
		Eigen::VectorXd y = Eigen::VectorXd::Zero(y_start.size());
		if (allow_rotations_)
		{
			for (int j = 0; j < bbw_weights_.cols(); ++j)
			{
				Eigen::Matrix<double, 6, 1> affine_params = x.segment(j * 6, 6);
				for (int i = 0; i < bbw_weights_.rows(); ++i)
					y.segment(i * 3, 3) += bbw_weights_(i, j) * affine_transformation(y_start.segment(i * 3, 3), affine_params);
			}
		}
		else
		{
			for (int j = 0; j < bbw_weights_.cols(); ++j)
				for (int i = 0; i < bbw_weights_.rows(); ++i)
					y.segment(i * 3, 3) += bbw_weights_(i, j) * (y_start.segment(i * 3, 3) + x.segment(j * 3, 3));
		}

		for (int j = 0; j < boundary_bbw_weights_.cols(); ++j)
			for (int i = 0; i < boundary_bbw_weights_.rows(); ++i)
				y.segment(i * 3, 3) += boundary_bbw_weights_(i, j) * y_start.segment(i * 3, 3);

		return y;
	}

	Eigen::VectorXd BoundedBiharmonicWeights2Dto3D::apply_jacobian(const Eigen::VectorXd &grad_full, const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
		if (allow_rotations_)
		{
			for (int j = 0; j < bbw_weights_.cols(); ++j)
				for (int i = 0; i < bbw_weights_.rows(); ++i)
					grad.segment(j * 6, 6) += bbw_weights_(i, j) * grad_affine_transformation(y_start.segment(i * 3, 3), x.segment(j * 6, 6)).transpose() * grad_full.segment(i * 3, 3);
		}
		else
		{
			for (int j = 0; j < bbw_weights_.cols(); ++j)
				for (int i = 0; i < bbw_weights_.rows(); ++i)
					grad.segment(j * 3, 3) += bbw_weights_(i, j) * grad_full.segment(i * 3, 3);
		}
		return grad;
	}

	void BoundedBiharmonicWeights2Dto3D::compute_faces_for_partial_vertices(const Eigen::MatrixXd &V, Eigen::MatrixXi &F) const
	{
		// The following implementation is maybe a bit wasteful, but is independent of state or surface selections
		std::unordered_map<int, int> full_to_reduced_indices;

		Eigen::MatrixXd BV;
		BV.setZero(3, 2);
		for (int i = 0; i < V.rows(); ++i)
		{
			if (i == 0)
			{
				BV.col(0) = V.row(i);
				BV.col(1) = V.row(i);
			}
			else
			{
				for (int j = 0; j < 3; ++j)
				{
					BV(j, 0) = std::min(BV(j, 0), V(i, j));
					BV(j, 1) = std::max(BV(j, 1), V(i, j));
				}
			}
		}

		Eigen::VectorXd bbox_width = (BV.col(1) - BV.col(0));
		for (int j = 0; j < 3; ++j)
			if (bbox_width(j) < 1e-12)
				bbox_width(j) = 1e-3;

		// Pad the bbox to make it conservative
		BV.col(0) -= 0.05 * (BV.col(1) - BV.col(0));
		BV.col(1) += 0.05 * (BV.col(1) - BV.col(0));

		auto in_bbox = [&BV](const Eigen::VectorXd &x) {
			bool in = true;
			in &= (x(0) >= BV(0, 0)) && (x(0) <= BV(0, 1));
			in &= (x(1) >= BV(1, 0)) && (x(1) <= BV(1, 1));
			in &= (x(2) >= BV(2, 0)) && (x(0) <= BV(2, 1));
			return in;
		};

		for (int i = 0; i < V_surface_.rows(); ++i)
			for (int j = 0; j < V.rows(); ++j)
			{
				// if (in_bbox(V_surface_.row(i)))
				if ((V_surface_.row(i) - V.row(j)).norm() < 1e-12)
					full_to_reduced_indices[i] = j;
			}

		F.resize(0, 3);
		for (int i = 0; i < F_surface_.rows(); ++i)
		{
			bool contains_face = true;
			for (int j = 0; j < 3; ++j)
				contains_face &= (full_to_reduced_indices.count(F_surface_(i, j)) == 1);

			if (contains_face)
			{
				F.conservativeResize(F.rows() + 1, 3);
				for (int j = 0; j < 3; ++j)
					F(F.rows() - 1, j) = full_to_reduced_indices.at(F_surface_(i, j));
			}
		}
	}
} // namespace polyfem::solver