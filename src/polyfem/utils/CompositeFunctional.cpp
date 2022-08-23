#pragma once
#include <polyfem/utils/CompositeFunctional.hpp>

#include <polyfem/utils/SplineParam.hpp>

namespace polyfem
{
	namespace
	{
		void vector2matrix(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat)
		{
			int size = sqrt(vec.size());
			assert(size * size == vec.size());

			mat.resize(size, size);
			for (int i = 0; i < size; i++)
				for (int j = 0; j < size; j++)
					mat(i, j) = vec(i * size + j);
		}

		bool state_has_solution(State &state)
		{
			if ((state.problem->is_time_dependent() && state.diff_cached.size() > 0) || (!state.problem->is_time_dependent() && state.sol.size() > 0))
				return true;
			else
				return false;
		}

		bool is_interested(const std::set<int> &interested_body_ids, const std::set<int> &interested_boundary_ids, const int body_id, const std::vector<int> &boundary_id, Eigen::MatrixXd &boundary_points)
		{
			boundary_points = Eigen::MatrixXd::Identity(boundary_id.size(), boundary_id.size());
			if (interested_body_ids.size() != 0 && interested_boundary_ids.size() != 0)
			{
				logger().error("Cannot specify both interested body and boundaries!");
				return false;
			}
			else if (interested_body_ids.size() != 0)
			{
				return interested_body_ids.count(body_id) != 0;
			}
			else if (interested_boundary_ids.size() != 0)
			{
				bool is_interested_boundary = false;
				for (int i = 0; i < boundary_id.size(); ++i)
				{
					if (interested_boundary_ids.count(boundary_id[i]) == 0)
						boundary_points(i, i) = 0.;
					else
						is_interested_boundary |= true;
				}

				return is_interested_boundary;
			}
			else
				// If nothing is specified as interesting, optimize everything.
				return true;
		}

		bool is_not_interested(const std::set<int> &interested_body_ids, const std::set<int> &interested_boundary_ids, const json &params, Eigen::MatrixXd &boundary_points)
		{
			return !is_interested(interested_body_ids, interested_boundary_ids, params["body_id"].get<int>(), params.contains("boundary_ids") ? params["boundary_ids"].get<std::vector<int>>() : std::vector<int>(), boundary_points);
		}
	} // namespace

	std::shared_ptr<CompositeFunctional> CompositeFunctional::create(const std::string &functional_name_)
	{
		std::shared_ptr<CompositeFunctional> func;
		if (functional_name_ == "TargetY")
			func = std::make_shared<TargetYFunctional>();
		else if (functional_name_ == "Trajectory")
			func = std::make_shared<TrajectoryFunctional>();
		else if (functional_name_ == "SDFTrajectory")
			func = std::make_shared<SDFTrajectoryFunctional>();
		else if (functional_name_ == "Volume")
			func = std::make_shared<VolumeFunctional>();
		else if (functional_name_ == "Mass")
			func = std::make_shared<MassFunctional>();
		else if (functional_name_ == "Height")
			func = std::make_shared<HeightFunctional>();
		else if (functional_name_ == "Stress")
			func = std::make_shared<StressFunctional>();
		else if (functional_name_ == "Compliance")
			func = std::make_shared<ComplianceFunctional>();
		else if (functional_name_ == "HomogenizedStiffness")
			func = std::make_shared<HomogenizedStiffnessFunctional>();
		else if (functional_name_ == "HomogenizedPermeability")
			func = std::make_shared<HomogenizedPermeabilityFunctional>();
		else if (functional_name_ == "CenterTrajectory")
			func = std::make_shared<CenterTrajectoryFunctional>();
		else if (functional_name_ == "CenterXYTrajectory")
			func = std::make_shared<CenterXYTrajectoryFunctional>();
		else if (functional_name_ == "CenterXZTrajectory")
			func = std::make_shared<CenterXZTrajectoryFunctional>();
		else if (functional_name_ == "NodeTrajectory")
			func = std::make_shared<NodeTrajectoryFunctional>();
		else
			logger().error("Unknown CompositeFunctional type!");

		return func;
	}

	double TargetYFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_target_y_functional();

		return sqrt(state.J(j));
	}

	Eigen::VectorXd TargetYFunctional::gradient(State &state, const std::string &type)
	{
		IntegrableFunctional j = get_target_y_functional();

		Eigen::VectorXd grad = state.integral_gradient(j, type);

		return grad / 2 / energy(state);
	}

	IntegrableFunctional TargetYFunctional::get_target_y_functional()
	{
		IntegrableFunctional j(surface_integral);
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), 1);
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			for (int q = 0; q < u.rows(); q++)
			{
				val(q) = pow(u(q, 1) - target_y(pts(q, 0) + u(q, 0)), p);
			}
		});

		j.set_dj_du([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			for (int q = 0; q < u.rows(); q++)
			{
				val(q, 1) = p * pow(u(q, 1) - target_y(pts(q, 0) + u(q, 0)), p - 1.);

				double grad = target_y_derivative(pts(q, 0) + u(q, 0));
				val(q, 0) += p * pow(target_y(pts(q, 0) + u(q, 0)) - u(q, 1), p - 1.) * grad;
			}
		});

		j.set_dj_dx([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			for (int q = 0; q < u.rows(); q++)
			{
				val(q, 1) = p * pow(u(q, 1) - target_y(pts(q, 0) + u(q, 0)), p - 1.);

				double grad = target_y_derivative(pts(q, 0) + u(q, 0));
				val(q, 0) += p * pow(target_y(pts(q, 0) + u(q, 0)) - u(q, 1), p - 1.) * grad;
			}
		});

		return j;
	}

	double TrajectoryFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_trajectory_functional(/*Doesn't matter for value*/ "");

		return sqrt(state.J(j));
	}

	Eigen::VectorXd TrajectoryFunctional::gradient(State &state, const std::string &type)
	{
		IntegrableFunctional j = get_trajectory_functional(type);

		Eigen::VectorXd grad = state.integral_gradient(j, type);

		return grad / 2 / energy(state);
	}

	void TrajectoryFunctional::set_reference(State *state_ref, const State &state, const std::set<int> &reference_cached_body_ids)
	{
		state_ref_ = state_ref;

		const int ref_n_bases = state_ref_->iso_parametric() ? state_ref_->bases.size() : state_ref_->geom_bases.size();
		const int n_bases = state.iso_parametric() ? state.bases.size() : state.geom_bases.size();

		std::map<int, std::vector<int>> ref_interested_body_id_to_e;
		int ref_count = 0;
		for (int e = 0; e < ref_n_bases; ++e)
		{
			int body_id = state_ref_->mesh->get_body_id(e);
			if (reference_cached_body_ids.size() > 0 && reference_cached_body_ids.count(body_id) == 0)
				continue;
			if (ref_interested_body_id_to_e.find(body_id) != ref_interested_body_id_to_e.end())
				ref_interested_body_id_to_e[body_id].push_back(e);
			else
				ref_interested_body_id_to_e[body_id] = {e};
			ref_count++;
		}

		std::map<int, std::vector<int>> interested_body_id_to_e;
		int count = 0;
		for (int e = 0; e < n_bases; ++e)
		{
			int body_id = state.mesh->get_body_id(e);
			if (reference_cached_body_ids.size() > 0 && reference_cached_body_ids.count(body_id) == 0)
				continue;
			if (interested_body_id_to_e.find(body_id) != interested_body_id_to_e.end())
				interested_body_id_to_e[body_id].push_back(e);
			else
				interested_body_id_to_e[body_id] = {e};
			count++;
		}

		if (count != ref_count)
			logger().error("Number of interested elements in the reference and optimization examples do not match!");
		else
			logger().trace("Found {} matching elements.", count);

		for (const auto &kv : interested_body_id_to_e)
		{
			for (int i = 0; i < kv.second.size(); ++i)
			{
				e_to_ref_e_[kv.second[i]] = ref_interested_body_id_to_e[kv.first][i];
			}
		}
	}

	IntegrableFunctional TrajectoryFunctional::get_trajectory_functional(const std::string &derivative_type)
	{
		assert(state_ref_);
		IntegrableFunctional j(surface_integral);
		j.set_transient_integral_type(transient_integral_type);
		{
			auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), 1);
				Eigen::MatrixXd boundary_points;
				if (is_not_interested(interested_body_ids_, interested_boundary_ids_, params, boundary_points))
					return;
				const int e = params["elem"];
				const int e_ref = e_to_ref_e_.find(e) != e_to_ref_e_.end() ? e_to_ref_e_[e] : e;
				const auto &gbase_ref = state_ref_->iso_parametric() ? state_ref_->bases[e_ref] : state_ref_->geom_bases[e_ref];

				Eigen::MatrixXd pts_ref;
				gbase_ref.eval_geom_mapping(local_pts, pts_ref);

				Eigen::MatrixXd u_ref, grad_u_ref;
				const Eigen::MatrixXd &sol_ref = state_ref_->problem->is_time_dependent() ? state_ref_->diff_cached[params["step"].get<int>()].u : state_ref_->sol;
				state_ref_->interpolate_at_local_vals(e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

				for (int q = 0; q < u.rows(); q++)
				{
					val(q) = pow(((u_ref.row(q) + pts_ref.row(q)) - (u.row(q) + pts.row(q))).squaredNorm(), p / 2.);
				}
				if (boundary_points.cols() == val.size())
					val = boundary_points * val;
			};

			auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), u.cols());
				Eigen::MatrixXd boundary_points;
				if (is_not_interested(interested_body_ids_, interested_boundary_ids_, params, boundary_points))
					return;
				const int e = params["elem"];
				const int e_ref = e_to_ref_e_.find(e) != e_to_ref_e_.end() ? e_to_ref_e_[e] : e;
				const auto &gbase_ref = state_ref_->iso_parametric() ? state_ref_->bases[e_ref] : state_ref_->geom_bases[e_ref];

				Eigen::MatrixXd pts_ref;
				gbase_ref.eval_geom_mapping(local_pts, pts_ref);

				Eigen::MatrixXd u_ref, grad_u_ref;
				const Eigen::MatrixXd &sol_ref = state_ref_->problem->is_time_dependent() ? state_ref_->diff_cached[params["step"].get<int>()].u : state_ref_->sol;
				state_ref_->interpolate_at_local_vals(e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

				for (int q = 0; q < u.rows(); q++)
				{
					auto x = (u.row(q) + pts.row(q)) - (u_ref.row(q) + pts_ref.row(q));
					val.row(q) = (p * pow(x.squaredNorm(), p / 2. - 1)) * x;
				}
				if (boundary_points.cols() == val.size())
					val = boundary_points * val;
			};

			auto djdx_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), u.cols());
				Eigen::MatrixXd boundary_points;
				if (is_not_interested(interested_body_ids_, interested_boundary_ids_, params, boundary_points))
					return;
				const int e = params["elem"];
				const int e_ref = e_to_ref_e_.find(e) != e_to_ref_e_.end() ? e_to_ref_e_[e] : e;
				const auto &gbase_ref = state_ref_->iso_parametric() ? state_ref_->bases[e_ref] : state_ref_->geom_bases[e_ref];

				Eigen::MatrixXd pts_ref;
				gbase_ref.eval_geom_mapping(local_pts, pts_ref);

				Eigen::MatrixXd u_ref, grad_u_ref;
				const Eigen::MatrixXd &sol_ref = state_ref_->problem->is_time_dependent() ? state_ref_->diff_cached[params["step"].get<int>()].u : state_ref_->sol;
				state_ref_->interpolate_at_local_vals(e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

				for (int q = 0; q < u.rows(); q++)
				{
					auto x = u_ref.row(q) - u.row(q);
					Eigen::MatrixXd grad_u_ref_q;
					vector2matrix(grad_u_ref.row(q), grad_u_ref_q);
					val.row(q) = (p * pow(x.squaredNorm(), p / 2. - 1)) * x.transpose() * grad_u_ref_q;
				}
				if (boundary_points.cols() == val.size())
					val = boundary_points * val;
			};

			j.set_j(j_func);
			if (derivative_type == "shape")
			{
				j.set_dj_du(djdu_func);
				j.set_dj_dx(djdu_func);
			}
			else
			{
				j.set_dj_du(djdu_func);
			}

			return j;
		}
	}

	double SDFTrajectoryFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_trajectory_functional(/*Doesn't matter for value*/ "shape");

		return state.J(j);
	}

	Eigen::VectorXd SDFTrajectoryFunctional::gradient(State &state, const std::string &type)
	{
		IntegrableFunctional j = get_trajectory_functional(type);

		Eigen::VectorXd grad = state.integral_gradient(j, type);

		return grad;
	}

	void SDFTrajectoryFunctional::compute_distance(const Eigen::MatrixXd &point, double &distance)
	{
		auto g = [&](const Eigen::VectorXd &t) {
			Eigen::MatrixXd fun(dim * (control_points_.rows() - 1), 1);
			for (int i = 0; i < control_points_.rows() - 1; ++i)
			{
				Eigen::MatrixXd val;
				SplineParam::eval(control_points_.block(i, 0, 2, dim), tangents_.block(i, 0, 2, dim), t(i, 0), val);
				fun.block(dim * i, 0, dim, 1) = val.transpose() - point;
			}
			return fun;
		};
		auto J = [&](const Eigen::VectorXd &t) {
			Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(dim * (control_points_.rows() - 1), control_points_.rows() - 1);
			for (int i = 0; i < t.rows(); ++i)
			{
				Eigen::MatrixXd val;
				SplineParam::deriv(control_points_.block(i, 0, 2, dim), tangents_.block(i, 0, 2, dim), t(i, 0), val);
				jac.block(dim * i, i, dim, 1) = val.transpose();
			}
			return jac;
		};

		Eigen::MatrixXd t = Eigen::MatrixXd::Ones(control_points_.rows() - 1, 1) / 2.;
		for (int i = 0; i < 100; ++i)
		{
			Eigen::MatrixXd jac_inv = J(t).completeOrthogonalDecomposition().pseudoInverse();
			Eigen::MatrixXd func = g(t);
			t -= jac_inv * func;
		}

		Eigen::MatrixXd distances = g(t);
		double min_distance = DBL_MAX;
		bool found = false;
		for (int i = 0; i < t.rows(); ++i)
		{
			if ((t(i, 0) < 0) || (t(i, 0) > 1))
				continue;
			double curr_distance = distances.block(dim * i, 0, dim, 1).norm();
			if (curr_distance < min_distance)
			{
				min_distance = curr_distance;
				found = true;
			}
		}

		if (!found)
			min_distance = std::min(distances(0, 0), distances(t.rows() - 1, 0));

		distance = min_distance;
	}

	void SDFTrajectoryFunctional::evaluate(const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad)
	{
		grad.setZero(dim, 1);
		int num_points = dim == 2 ? 4 : 8;
		Eigen::MatrixXd A(num_points, num_points);
		Eigen::VectorXd b(num_points);
		if (dim == 2)
		{
			Eigen::MatrixXi bin(dim, 1);
			for (int k = 0; k < dim; ++k)
				bin(k, 0) = (int)std::floor(point(k) / delta_(k));
			Eigen::MatrixXd keys(4, dim);
			keys << bin(0), bin(1),
				bin(0) + 1, bin(1),
				bin(0), bin(1) + 1,
				bin(0) + 1, bin(1) + 1;
			std::vector<std::string> keys_string;
			keys_string.push_back(std::to_string(bin(0)) + "," + std::to_string(bin(1)));
			keys_string.push_back(std::to_string(bin(0) + 1) + "," + std::to_string(bin(1)));
			keys_string.push_back(std::to_string(bin(0)) + "," + std::to_string(bin(1) + 1));
			keys_string.push_back(std::to_string(bin(0) + 1) + "," + std::to_string(bin(1) + 1));
			for (int i = 0; i < 4; ++i)
			{
				Eigen::MatrixXd clamped_point = keys.row(i).cwiseProduct(delta_).transpose();
				if (implicit_function.count(keys_string[i]) == 0)
					compute_distance(clamped_point, implicit_function[keys_string[i]]);
				A.row(i) << 1., clamped_point(0), clamped_point(1), clamped_point(0) * clamped_point(1);
				b(i) = implicit_function[keys_string[i]];
			}
		}
		else
		{
			logger().error("Don't yet support 3D SDF.");
		}

		Eigen::VectorXd weights = A.householderQr().solve(b);
		if (dim == 2)
		{
			val = weights(0) + weights(1) * point(0) + weights(2) * point(1) + weights(3) * point(0) * point(1);
			grad << weights(1) + weights(3) * point(1), weights(2) + weights(3) * point(0);
		}
		else
		{
			val = weights(0) + weights(1) * point(0) + weights(2) * point(1) + weights(3) * point(2) + weights(4) * point(0) * point(1) + weights(5) * point(1) * point(2) + weights(6) * point(0) * point(2) + weights(7) * point(0) * point(1) * point(2);
			logger().error("Don't yet support trilinear interpolation.");
		}

		for (int i = 0; i < dim; ++i)
			if (std::isnan(grad(i)))
			{
				logger().error("Nan found in gradient computation.");
				break;
			}
	}

	IntegrableFunctional SDFTrajectoryFunctional::get_trajectory_functional(const std::string &derivative_type)
	{
		assert(transient_integral_type == "final");
		IntegrableFunctional j(surface_integral);
		j.set_transient_integral_type(transient_integral_type);
		{
			auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), 1);
				Eigen::MatrixXd boundary_points;
				if (is_not_interested(interested_body_ids_, interested_boundary_ids_, params, boundary_points))
					return;

				for (int q = 0; q < u.rows(); q++)
				{
					double distance;
					Eigen::MatrixXd unused_grad;
					evaluate(u.row(q) + pts.row(q), distance, unused_grad);
					val(q) = pow(distance, p);
				}
				val = boundary_points * val;
			};

			auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), u.cols());
				Eigen::MatrixXd boundary_points;
				if (is_not_interested(interested_body_ids_, interested_boundary_ids_, params, boundary_points))
					return;

				for (int q = 0; q < u.rows(); q++)
				{
					double distance;
					Eigen::MatrixXd grad;
					evaluate(u.row(q) + pts.row(q), distance, grad);
					val.row(q) = p * pow(distance, p - 1) * grad.transpose();
				}
				val = boundary_points * val;
			};

			j.set_j(j_func);
			if (derivative_type == "shape")
			{
				j.set_dj_du(djdu_func);
				j.set_dj_dx(djdu_func);
			}
			else
			{
				logger().error("Don't yet support optimization of this Functional on not shape.");
			}

			return j;
		}
	}

	double NodeTrajectoryFunctional::energy(State &state)
	{
		SummableFunctional j = get_trajectory_functional();
		if (active_vertex_mask.size() > 0 && state.n_bases != active_vertex_mask.size())
			logger().error("vertex mask size doesn't match number of nodes!");
		return sqrt(state.J_static(j));
	}

	Eigen::VectorXd NodeTrajectoryFunctional::gradient(State &state, const std::string &type)
	{
		SummableFunctional j = get_trajectory_functional();
		if (active_vertex_mask.size() > 0 && state.n_bases != active_vertex_mask.size())
			logger().error("vertex mask size doesn't match number of nodes!");
		Eigen::VectorXd grad;
		state.dJ_material_static(j, grad);

		return grad / 2 / energy(state);
	}

	SummableFunctional NodeTrajectoryFunctional::get_trajectory_functional()
	{
		SummableFunctional j;
		{
			auto j_func = [this](const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const json &params, Eigen::MatrixXd &val) {
				val.setZero(1, 1);
				const int vid = params["node"];
				if (active_vertex_mask.size() > 0 && !active_vertex_mask[vid])
					return;
				assert(pts.rows() == 1);
				val(0) = (pts + u - target_vertex_positions.row(vid)).squaredNorm();
			};

			auto djdu_func = [this](const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const json &params, Eigen::MatrixXd &val) {
				val.setZero(pts.rows(), pts.cols());
				const int vid = params["node"];
				if (active_vertex_mask.size() > 0 && !active_vertex_mask[vid])
					return;
				assert(pts.rows() == 1);
				val = (pts + u - target_vertex_positions.row(vid)) * 2;
			};

			j.set_j(j_func);
			j.set_dj_du(djdu_func);
			j.set_dj_dx(djdu_func);

			return j;
		}
	}

	double VolumeFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_volume_functional();

		double current_volume = state.J(j);
		logger().trace("Current volume: {}", current_volume);

		if (current_volume > max_volume)
			return pow(current_volume - max_volume, 2);
		else if (current_volume < min_volume)
			return pow(current_volume - min_volume, 2);
		else
			return 0.;
	}

	Eigen::VectorXd VolumeFunctional::gradient(State &state, const std::string &type)
	{
		assert(type == "shape");
		IntegrableFunctional j = get_volume_functional();

		Eigen::VectorXd grad = state.integral_gradient(j, type);
		double current_volume = state.J(j);

		double derivative = 0;
		if (current_volume > max_volume)
			derivative = 2 * (current_volume - max_volume);
		else if (current_volume < min_volume)
			derivative = 2 * (current_volume - min_volume);

		return derivative * grad;
	}

	IntegrableFunctional VolumeFunctional::get_volume_functional()
	{
		assert(max_volume >= min_volume);
		assert(!surface_integral);
		assert(transient_integral_type == "final");
		IntegrableFunctional j(surface_integral);
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				val.setZero(u.rows(), 1);
			else
				val.setOnes(u.rows(), 1);
		});
		return j;
	}

	double MassFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_mass_functional();

		double current_mass = state.J(j);
		logger().trace("Current mass: {}", current_mass);

		if (current_mass > max_mass)
			return pow(current_mass - max_mass, 2);
		else if (current_mass < min_mass)
			return pow(current_mass - min_mass, 2);
		else
			return 0.;
	}

	Eigen::VectorXd MassFunctional::gradient(State &state, const std::string &type)
	{
		IntegrableFunctional j = get_mass_functional();

		Eigen::VectorXd grad = state.integral_gradient(j, type);
		double current_mass = state.J(j);

		double derivative = 0;
		if (current_mass > max_mass)
			derivative = 2 * (current_mass - max_mass);
		else if (current_mass < min_mass)
			derivative = 2 * (current_mass - min_mass);

		return derivative * grad;
	}

	IntegrableFunctional MassFunctional::get_mass_functional()
	{
		assert(max_mass >= min_mass);
		assert(!surface_integral);
		assert(transient_integral_type == "final");
		IntegrableFunctional j(surface_integral);
		j.set_name("Mass");
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				val.setZero(u.rows(), 1);
			else
			{
				val.setOnes(u.rows(), 1);
				val *= params["density"].get<double>();
			}
		});
		return j;
	}

	double HeightFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_height_functional();

		return state.J(j);
	}

	Eigen::VectorXd HeightFunctional::gradient(State &state, const std::string &type)
	{
		IntegrableFunctional j = get_height_functional();

		return state.integral_gradient(j, type);
	}

	IntegrableFunctional HeightFunctional::get_height_functional()
	{
		assert(!surface_integral);
		IntegrableFunctional j(surface_integral);
		j.set_name("Center");
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				val.setZero(u.rows(), 1);
			else
				val = -(u.col(1) + pts.col(1));
		});

		j.set_dj_du([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			val.col(1).array() = -1;
		});

		j.set_dj_dx([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(pts.rows(), pts.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			val.col(1).array() = -1;
		});

		return j;
	}

	double StressFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_stress_functional(state.formulation(), p);

		return pow(state.J(j), 1. / p);
	}

	Eigen::VectorXd StressFunctional::gradient(State &state, const std::string &type)
	{
		IntegrableFunctional j = get_stress_functional(state.formulation(), p);

		Eigen::VectorXd grad = state.integral_gradient(j, type);
		double val = state.J(j);

		return (pow(val, 1. / p - 1) / p) * grad;
	}

	IntegrableFunctional StressFunctional::get_stress_functional(const std::string &formulation, const int power)
	{
		assert(!surface_integral);
		IntegrableFunctional j(surface_integral);
		j.set_name("Stress");
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([formulation, power, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				Eigen::MatrixXd grad_u_q, stress;
				vector2matrix(grad_u.row(q), grad_u_q);
				if (formulation == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else if (formulation == "NeoHookean")
				{
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
				}
				else
					logger().error("Unknown formulation!");
				val(q) = pow(stress.squaredNorm(), power / 2.);
			}
		});

		j.set_dj_dgradu([formulation, power, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			const int dim = sqrt(grad_u.cols());
			for (int q = 0; q < grad_u.rows(); q++)
			{
				Eigen::MatrixXd grad_u_q, stress, stress_dstress;
				vector2matrix(grad_u.row(q), grad_u_q);
				if (formulation == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
					stress_dstress = mu(q) * (stress + stress.transpose()) + lambda(q) * stress.trace() * Eigen::MatrixXd::Identity(stress.rows(), stress.cols());
				}
				else if (formulation == "NeoHookean")
				{
					Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
					Eigen::MatrixXd FmT = def_grad.inverse().transpose();
					stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
					stress_dstress = mu(q) * stress + FmT * stress.transpose() * FmT * (mu(q) - lambda(q) * std::log(def_grad.determinant())) + (lambda(q) * (FmT.array() * stress.array()).sum()) * FmT;
				}
				else
					logger().error("Unknown formulation!");

				const double coef = power * pow(stress.squaredNorm(), power / 2. - 1.);
				for (int i = 0; i < dim; i++)
					for (int l = 0; l < dim; l++)
						val(q, i * dim + l) = coef * stress_dstress(i, l);
			}
		});

		return j;
	}

	double HomogenizedStiffnessFunctional::energy(State &state)
	{
		Eigen::MatrixXd C_H;
		state.homogenize_weighted_linear_elasticity(C_H);

		return -C_H(1, 1);
	}

	Eigen::VectorXd HomogenizedStiffnessFunctional::gradient(State &state, const std::string &type)
	{
		Eigen::MatrixXd C_H;
		Eigen::MatrixXd grad;
		state.homogenize_weighted_linear_elasticity_grad(C_H, grad);

		return -grad.col(1);
	}

	double HomogenizedPermeabilityFunctional::energy(State &state)
	{
		Eigen::MatrixXd C_H;
		state.homogenize_weighted_stokes(C_H);

		std::cout << "Permeability tensor:\n" << C_H << "\n";
		if (state.mesh->is_volume())
			return -C_H(0, 1)-C_H(1, 2);
		else
			return -C_H(0, 1);
	}

	Eigen::VectorXd HomogenizedPermeabilityFunctional::gradient(State &state, const std::string &type)
	{
		Eigen::MatrixXd C_H;
		Eigen::MatrixXd grad;
		state.homogenize_weighted_stokes_grad(C_H, grad);

		if (state.mesh->is_volume())
			return -grad.col(1)-grad.col(5);
		else
			return -grad.col(1);
	}

	double ComplianceFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_compliance_functional(state.formulation());

		return state.J(j);
	}

	Eigen::VectorXd ComplianceFunctional::gradient(State &state, const std::string &type)
	{
		IntegrableFunctional j = get_compliance_functional(state.formulation());

		Eigen::VectorXd grad = state.integral_gradient(j, type);
		double val = state.J(j);

		return grad;
	}

	IntegrableFunctional ComplianceFunctional::get_compliance_functional(const std::string &formulation)
	{
		assert(!surface_integral);
		IntegrableFunctional j(surface_integral);
		j.set_name("Compliance");
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([formulation, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			for (int q = 0; q < grad_u.rows(); q++)
			{
				Eigen::MatrixXd grad_u_q, stress;
				vector2matrix(grad_u.row(q), grad_u_q);
				if (formulation == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else
					logger().error("Unknown formulation!");
				val(q) = (stress.array() * grad_u_q.array()).sum();
			}
		});

		j.set_dj_dgradu([formulation, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			const int dim = sqrt(grad_u.cols());
			for (int q = 0; q < grad_u.rows(); q++)
			{
				Eigen::MatrixXd grad_u_q, stress;
				vector2matrix(grad_u.row(q), grad_u_q);
				if (formulation == "LinearElasticity")
				{
					stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
				}
				else
					logger().error("Unknown formulation!");

				for (int i = 0; i < dim; i++)
					for (int l = 0; l < dim; l++)
						val(q, i * dim + l) = 2 * stress(i, l);
			}
		});

		return j;
	}

	double CenterTrajectoryFunctional::energy(State &state)
	{
		const int dim = state.mesh->dimension();
		std::vector<IntegrableFunctional> js(dim + 1);
		for (int d = 0; d < dim; d++)
			js[d] = get_center_trajectory_functional(d);
		js[dim] = get_volume_functional();

		if (transient_integral_type != "final" && target_series.size() != state.args["time"]["time_steps"].get<int>() + 1)
			logger().error("Number of center series {} doesn't match with number of time steps + 1: {}!", target_series.size(), state.args["time"]["time_steps"].get<int>() + 1);

		auto func = [&](const Eigen::VectorXd &x, const json &param) {
			double t = param["t"];
			int step = param["step"];

			Eigen::VectorXd target;
			if (transient_integral_type != "final")
				target = target_series[step];
			else
				target = target_series.back();

			if (step == state.args["time"]["time_steps"].get<int>())
				logger().trace("center: {}", x.head(dim).transpose() / x(dim));
			return (x.head(dim) / x(dim) - target).squaredNorm();
		};

		return sqrt(state.J_transient(js, func));
	}

	Eigen::VectorXd CenterTrajectoryFunctional::gradient(State &state, const std::string &type)
	{
		const int dim = state.mesh->dimension();
		std::vector<IntegrableFunctional> js(dim + 1);
		for (int d = 0; d < dim; d++)
			js[d] = get_center_trajectory_functional(d);
		js[dim] = get_volume_functional();

		if (transient_integral_type != "final" && target_series.size() != state.args["time"]["time_steps"].get<int>() + 1)
			logger().error("Number of center series {} doesn't match with number of time steps + 1: {}!", target_series.size(), state.args["time"]["time_steps"].get<int>() + 1);

		auto func = [dim, this](const Eigen::VectorXd &x, const json &param) {
			double t = param["t"];
			int step = param["step"];

			Eigen::VectorXd target;
			if (transient_integral_type != "final")
				target = target_series[step];
			else
				target = target_series.back();

			double val = (x.head(dim) / x(dim) - target).squaredNorm();

			Eigen::VectorXd grad;
			grad.setZero(dim + 1);
			for (int d = 0; d < dim; d++)
			{
				grad(d) = 2 / x(dim) * (x(d) / x(dim) - target(d));

				grad(dim) += -x(d) * grad(d) / x(dim);
			}
			return grad;
		};

		return state.integral_gradient(js, func, type) / 2 / energy(state);
	}

	IntegrableFunctional CenterTrajectoryFunctional::get_center_trajectory_functional(const int d)
	{
		IntegrableFunctional j;
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				val.setZero(u.rows(), 1);
			else
				val = u.col(d) + pts.col(d);
		});

		j.set_dj_du([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			val.col(d).setOnes();
		});

		j.set_dj_dx([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(pts.rows(), pts.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			val.col(d).setOnes();
		});
		return j;
	}

	IntegrableFunctional CenterTrajectoryFunctional::get_volume_functional()
	{
		IntegrableFunctional j;
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				val.setZero(u.rows(), 1);
			else
				val.setOnes(u.rows(), 1);
		});
		return j;
	}

	void CenterTrajectoryFunctional::get_barycenter_series(State &state, std::vector<Eigen::VectorXd> &barycenters)
	{
		assert(state.problem->is_time_dependent());
		const int dim = state.mesh->dimension();
		const int n_steps = state.args["time"]["time_steps"].get<int>();

		std::vector<IntegrableFunctional> js(dim);
		for (int d = 0; d < dim; d++)
		{
			js[d].set_transient_integral_type("uniform");
			js[d].set_j([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
					val.setZero(u.rows(), 1);
				else
					val = u.col(d) + pts.col(d);
			});

			js[d].set_dj_du([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), u.cols());
				if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
					return;
				val.col(d).setOnes();
			});

			js[d].set_dj_dx([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(pts.rows(), pts.cols());
				if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
					return;
				val.col(d).setOnes();
			});
		}

		barycenters.clear();
		Eigen::VectorXd barycenter(dim);
		for (int step = 0; step <= n_steps; step++)
		{
			for (int d = 0; d < dim; d++)
				barycenter(d) = state.J_transient_step(js[d], step);
			barycenters.push_back(barycenter);
		}

		IntegrableFunctional j_vol;
		j_vol.set_transient_integral_type("final");
		j_vol.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				val.setZero(u.rows(), 1);
			else
				val.setOnes(u.rows(), 1);
		});
		double volume = state.J(j_vol);

		for (auto &point : barycenters)
			point /= volume;
	}

	double CenterXYTrajectoryFunctional::energy(State &state)
	{
		const int n_steps = state.args["time"]["time_steps"];
		const int n_targets = target_series.size();
		const int sample_rate = n_targets > 1 ? n_steps / (n_targets - 1) : 1;
		if (transient_integral_type != "final" && sample_rate * (n_targets - 1) != n_steps)
			logger().error("Number of center series {} doesn't match with number of time steps: {}!", n_targets, n_steps);


		std::vector<IntegrableFunctional> js(3);
		js[0] = get_center_trajectory_functional(0);
		js[1] = get_center_trajectory_functional(1);
		js[2] = get_volume_functional();

		auto func = [this, sample_rate](const Eigen::VectorXd &x, const json &param) {
			double t = param["t"];
			int step = param["step"];

			if (step % sample_rate != 0)
				return 0.0;

			Eigen::VectorXd target;
			if (transient_integral_type != "final")
				target = target_series[step / sample_rate];
			else
				target = target_series.back();

			return (x.head(2) / x(2) - target.head(2)).squaredNorm();
		};

		return sqrt(state.J_transient(js, func));
	}

	Eigen::VectorXd CenterXYTrajectoryFunctional::gradient(State &state, const std::string &type)
	{
		const int n_steps = state.args["time"]["time_steps"];
		const int n_targets = target_series.size();
		const int sample_rate = n_targets > 1 ? n_steps / (n_targets - 1) : 1;
		if (transient_integral_type != "final" && sample_rate * (n_targets - 1) != n_steps)
			logger().error("Number of center series {} doesn't match with number of time steps: {}!", n_targets, n_steps);


		std::vector<IntegrableFunctional> js(3);
		js[0] = get_center_trajectory_functional(0);
		js[1] = get_center_trajectory_functional(1);
		js[2] = get_volume_functional();

		auto func = [sample_rate, this](const Eigen::VectorXd &x, const json &param) {
			double t = param["t"];
			int step = param["step"];

			Eigen::VectorXd target;
			if (transient_integral_type != "final")
				target = target_series[step / sample_rate];
			else
				target = target_series.back();

			double val = (x.head(2) / x(2) - target.head(2)).squaredNorm();

			Eigen::VectorXd grad;
			grad.setZero(3);

			if (step % sample_rate != 0)
				return grad;

			for (int d = 0; d < 2; d++)
			{
				grad(d) = 2 / x(2) * (x(d) / x(2) - target(d));

				grad(2) += -x(d) * grad(d) / x(2);
			}
			return grad;
		};

		return state.integral_gradient(js, func, type) / 2 / energy(state);
	}

	IntegrableFunctional CenterXYTrajectoryFunctional::get_center_trajectory_functional(const int d)
	{
		IntegrableFunctional j;
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), 1);
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			else
				val = u.col(d) + pts.col(d);
		});

		j.set_dj_du([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			val.col(d).setOnes();
		});

		j.set_dj_dx([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(pts.rows(), pts.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			val.col(d).setOnes();
		});
		return j;
	}

	IntegrableFunctional CenterXYTrajectoryFunctional::get_volume_functional()
	{
		IntegrableFunctional j;
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				val.setZero(u.rows(), 1);
			else
				val.setOnes(u.rows(), 1);
		});
		return j;
	}

	double CenterXZTrajectoryFunctional::energy(State &state)
	{
		const int n_steps = state.args["time"]["time_steps"];
		const int n_targets = target_series.size();
		const int sample_rate = n_targets > 1 ? n_steps / (n_targets - 1) : 1;
		if (transient_integral_type != "final" && sample_rate * (n_targets - 1) != n_steps)
			logger().error("Number of center series {} doesn't match with number of time steps: {}!", n_targets, n_steps);


		std::vector<IntegrableFunctional> js(3);
		js[0] = get_center_trajectory_functional(0);
		js[1] = get_center_trajectory_functional(2);
		js[2] = get_volume_functional();

		auto func = [this, sample_rate](const Eigen::VectorXd &x, const json &param) {
			double t = param["t"];
			int step = param["step"];

			if (step % sample_rate != 0)
				return 0.0;

			Eigen::VectorXd target;
			if (transient_integral_type != "final")
				target = target_series[step / sample_rate];
			else
				target = target_series.back();

			// return (x.head(2) / x(2) - target_series[step / sample_rate].head(2)).squaredNorm();
			return pow(x(0) / x(2) - target(0), 2) + pow(x(1) / x(2) - target(2), 2);
		};

		return sqrt(state.J_transient(js, func));
	}

	Eigen::VectorXd CenterXZTrajectoryFunctional::gradient(State &state, const std::string &type)
	{
		const int n_steps = state.args["time"]["time_steps"];
		const int n_targets = target_series.size();
		const int sample_rate = n_targets > 1 ? n_steps / (n_targets - 1) : 1;
		if (transient_integral_type != "final" && sample_rate * (n_targets - 1) != n_steps)
			logger().error("Number of center series {} doesn't match with number of time steps: {}!", n_targets, n_steps);


		std::vector<IntegrableFunctional> js(3);
		js[0] = get_center_trajectory_functional(0);
		js[1] = get_center_trajectory_functional(2);
		js[2] = get_volume_functional();

		auto func = [sample_rate, this](const Eigen::VectorXd &x, const json &param) {
			double t = param["t"];
			int step = param["step"];

			Eigen::VectorXd target;
			if (transient_integral_type != "final")
				target = target_series[step / sample_rate];
			else
				target = target_series.back();
			
			// double val = (x.head(2) / x(2) - target.head(2)).squaredNorm();
			double val = pow(x(0) / x(2) - target(0), 2) + pow(x(1) / x(2) - target(2), 2);

			Eigen::VectorXd grad;
			grad.setZero(3);

			if (step % sample_rate != 0)
				return grad;

			for (int d = 0; d < 2; d++)
			{
				grad(d) = 2 / x(2) * (x(d) / x(2) - target(2*d));

				grad(2) += -x(d) * grad(d) / x(2);
			}
			return grad;
		};

		return state.integral_gradient(js, func, type) / 2 / energy(state);
	}

	IntegrableFunctional CenterXZTrajectoryFunctional::get_center_trajectory_functional(const int d)
	{
		IntegrableFunctional j;
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), 1);
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			else
				val = u.col(d) + pts.col(d);
		});

		j.set_dj_du([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			val.col(d).setOnes();
		});

		j.set_dj_dx([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(pts.rows(), pts.cols());
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				return;
			val.col(d).setOnes();
		});
		return j;
	}

	IntegrableFunctional CenterXZTrajectoryFunctional::get_volume_functional()
	{
		IntegrableFunctional j;
		j.set_transient_integral_type(transient_integral_type);
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			if (interested_body_ids_.size() > 0 && interested_body_ids_.count(params["body_id"].get<int>()) == 0)
				val.setZero(u.rows(), 1);
			else
				val.setOnes(u.rows(), 1);
		});
		return j;
	}
} // namespace polyfem