#include <polyfem/utils/CompositeFunctional.hpp>
#include <polyfem/utils/CompositeSplineParam.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/solver/AdjointForm.hpp>

using namespace polyfem::solver;
using namespace polyfem::utils;

namespace polyfem
{
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
		else if (functional_name_ == "Height")
			func = std::make_shared<HeightFunctional>();
		else if (functional_name_ == "Stress")
			func = std::make_shared<StressFunctional>();
		else if (functional_name_ == "Compliance")
			func = std::make_shared<ComplianceFunctional>();
		// else if (functional_name_ == "CenterTrajectory")
		// 	func = std::make_shared<CenterTrajectoryFunctional>();
		else if (functional_name_ == "NodeTrajectory")
			func = std::make_shared<NodeTrajectoryFunctional>();
		else
			logger().error("Unknown CompositeFunctional type!");

		return func;
	}

	double CompositeFunctional::energy(State &state)
	{
		return AdjointForm::value(state, get_target_functional(), surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);
	}

	Eigen::VectorXd CompositeFunctional::gradient(State &state, const std::string &type)
	{
		Eigen::VectorXd grad;
		AdjointForm::gradient(state, get_target_functional(type), type, grad, surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);

		return grad;
	}

	IntegrableFunctional TargetYFunctional::get_target_functional(const std::string &type)
	{
		IntegrableFunctional j;
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), 1);
			for (int q = 0; q < u.rows(); q++)
			{
				val(q) = pow(u(q, 1) - target_y(pts(q, 0) + u(q, 0)), p);
			}
		});

		j.set_dj_du([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			for (int q = 0; q < u.rows(); q++)
			{
				val(q, 1) = p * pow(u(q, 1) - target_y(pts(q, 0) + u(q, 0)), p - 1.);

				double grad = target_y_derivative(pts(q, 0) + u(q, 0));
				val(q, 0) += p * pow(target_y(pts(q, 0) + u(q, 0)) - u(q, 1), p - 1.) * grad;
			}
		});

		j.set_dj_dx([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			for (int q = 0; q < u.rows(); q++)
			{
				val(q, 1) = p * pow(u(q, 1) - target_y(pts(q, 0) + u(q, 0)), p - 1.);

				double grad = target_y_derivative(pts(q, 0) + u(q, 0));
				val(q, 0) += p * pow(target_y(pts(q, 0) + u(q, 0)) - u(q, 1), p - 1.) * grad;
			}
		});

		return j;
	}

	void TrajectoryFunctional::set_reference(State *state_ref, const State &state, const std::set<int> &reference_cached_body_ids)
	{
		state_ref_ = state_ref;

		std::map<int, std::vector<int>> ref_interested_body_id_to_e;
		int ref_count = 0;
		for (int e = 0; e < state_ref_->bases.size(); ++e)
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
		for (int e = 0; e < state.bases.size(); ++e)
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

	IntegrableFunctional TrajectoryFunctional::get_target_functional(const std::string &type)
	{
		assert(state_ref_);
		IntegrableFunctional j;
		{
			auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), 1);
				const int e = params["elem"];
				const int e_ref = e_to_ref_e_.find(e) != e_to_ref_e_.end() ? e_to_ref_e_[e] : e;
				const auto &gbase_ref = state_ref_->geom_bases()[e_ref];

				Eigen::MatrixXd pts_ref;
				gbase_ref.eval_geom_mapping(local_pts, pts_ref);

				Eigen::MatrixXd u_ref, grad_u_ref;
				const Eigen::MatrixXd &sol_ref = state_ref_->problem->is_time_dependent() ? state_ref_->diff_cached[params["step"].get<int>()].u : state_ref_->diff_cached[0].u;
				io::Evaluator::interpolate_at_local_vals(*(state_ref_->mesh), state_ref_->problem->is_scalar(), state_ref_->bases, state_ref_->geom_bases(), e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

				for (int q = 0; q < u.rows(); q++)
				{
					val(q) = pow(((u_ref.row(q) + pts_ref.row(q)) - (u.row(q) + pts.row(q))).squaredNorm(), p / 2.);
				}
			};

			auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), u.cols());
				const int e = params["elem"];
				const int e_ref = e_to_ref_e_.find(e) != e_to_ref_e_.end() ? e_to_ref_e_[e] : e;
				const auto &gbase_ref = state_ref_->geom_bases()[e_ref];

				Eigen::MatrixXd pts_ref;
				gbase_ref.eval_geom_mapping(local_pts, pts_ref);

				Eigen::MatrixXd u_ref, grad_u_ref;
				const Eigen::MatrixXd &sol_ref = state_ref_->problem->is_time_dependent() ? state_ref_->diff_cached[params["step"].get<int>()].u : state_ref_->diff_cached[0].u;
				io::Evaluator::interpolate_at_local_vals(*(state_ref_->mesh), state_ref_->problem->is_scalar(), state_ref_->bases, state_ref_->geom_bases(), e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

				for (int q = 0; q < u.rows(); q++)
				{
					auto x = (u.row(q) + pts.row(q)) - (u_ref.row(q) + pts_ref.row(q));
					val.row(q) = (p * pow(x.squaredNorm(), p / 2. - 1)) * x;
				}
			};

			auto djdx_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), u.cols());
				const int e = params["elem"];
				const int e_ref = e_to_ref_e_.find(e) != e_to_ref_e_.end() ? e_to_ref_e_[e] : e;
				const auto &gbase_ref = state_ref_->geom_bases()[e_ref];

				Eigen::MatrixXd pts_ref;
				gbase_ref.eval_geom_mapping(local_pts, pts_ref);

				Eigen::MatrixXd u_ref, grad_u_ref;
				const Eigen::MatrixXd &sol_ref = state_ref_->problem->is_time_dependent() ? state_ref_->diff_cached[params["step"].get<int>()].u : state_ref_->diff_cached[0].u;
				io::Evaluator::interpolate_at_local_vals(*(state_ref_->mesh), state_ref_->problem->is_scalar(), state_ref_->bases, state_ref_->geom_bases(), e_ref, local_pts, sol_ref, u_ref, grad_u_ref);

				for (int q = 0; q < u.rows(); q++)
				{
					auto x = u_ref.row(q) - u.row(q);
					Eigen::MatrixXd grad_u_ref_q;
					vector2matrix(grad_u_ref.row(q), grad_u_ref_q);
					val.row(q) = (p * pow(x.squaredNorm(), p / 2. - 1)) * x.transpose() * grad_u_ref_q;
				}
			};

			j.set_j(j_func);
			if (type == "shape")
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

	void SDFTrajectoryFunctional::compute_distance(const Eigen::MatrixXd &point, double &distance, Eigen::MatrixXd &grad)
	{
		int nearest;
		double t_optimal, distance_to_start, distance_to_end;
		CompositeSplineParam::find_nearest_spline(point, control_points_, tangents_, nearest, t_optimal, distance, distance_to_start, distance_to_end);

		// If no nearest with t \in [0, 1] found, check the endpoints and assign one
		if (nearest == -1)
		{
			if (distance_to_start < distance_to_end)
			{
				nearest = 0;
				t_optimal = 0;
				distance = distance_to_start;
			}
			else
			{

				nearest = control_points_.rows() - 2;
				t_optimal = 1;
				distance = distance_to_end;
			}
		}
		distance = pow(distance, 1. / 2.);

		grad.setZero(3, 1);
		if (distance < 1e-8)
			return;

		CompositeSplineParam::gradient(point, control_points_, tangents_, nearest, t_optimal, distance, grad);
		assert(abs(1 - grad.col(0).segment(0, 2).norm()) < 1e-6);
	}

	void SDFTrajectoryFunctional::bicubic_interpolation(const Eigen::MatrixXd &corner_point, const std::vector<std::string> &keys, const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad)
	{
		Eigen::MatrixXd corner_val(4, 1);
		Eigen::MatrixXd corner_grad(4, 3);
		for (int i = 0; i < 4; ++i)
		{
			corner_val(i) = implicit_function_distance.at(keys[i]);
			corner_grad.row(i) = implicit_function_grad.at(keys[i]).transpose();
		}
		Eigen::MatrixXd x(16, 1);
		x << corner_val(0), corner_val(1), corner_val(2), corner_val(3),
			delta_(0) * corner_grad(0, 0), delta_(0) * corner_grad(1, 0), delta_(0) * corner_grad(2, 0), delta_(0) * corner_grad(3, 0),
			delta_(1) * corner_grad(0, 1), delta_(1) * corner_grad(1, 1), delta_(1) * corner_grad(2, 1), delta_(1) * corner_grad(3, 1),
			delta_(0) * delta_(1) * corner_grad(0, 2), delta_(0) * delta_(1) * corner_grad(1, 2), delta_(0) * delta_(1) * corner_grad(2, 2), delta_(0) * delta_(1) * corner_grad(3, 2);

		Eigen::MatrixXd coeffs = bicubic_mat * x;

		auto bar_x = [&corner_point](double x_) { return (x_ - corner_point(0, 0)) / (corner_point(1, 0) - corner_point(0, 0)); };
		auto bar_y = [&corner_point](double y_) { return (y_ - corner_point(0, 1)) / (corner_point(2, 1) - corner_point(0, 1)); };

		val = 0;
		grad.setZero(2, 1);
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
			{
				val += coeffs(i + j * 4) * pow(bar_x(point(0)), i) * pow(bar_y(point(1)), j);
				grad(0) += i == 0 ? 0 : (coeffs(i + j * 4) * i * pow(bar_x(point(0)), i - 1) * pow(bar_y(point(1)), j));
				grad(1) += j == 0 ? 0 : coeffs(i + j * 4) * pow(bar_x(point(0)), i) * j * pow(bar_y(point(1)), j - 1);
			}

		grad(0) /= (corner_point(1, 0) - corner_point(0, 0));
		grad(1) /= (corner_point(2, 1) - corner_point(0, 1));

		assert(!std::isnan(grad(0)) && !std::isnan(grad(0)));
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
			Eigen::MatrixXd corner_point(4, 2);
			for (int i = 0; i < 4; ++i)
			{
				Eigen::MatrixXd clamped_point = keys.row(i).cwiseProduct(delta_).transpose();
				corner_point.row(i) = clamped_point.transpose();
				if (implicit_function_distance.count(keys_string[i]) == 0)
				{
					std::unique_lock lock(mutex_);
					compute_distance(clamped_point, implicit_function_distance[keys_string[i]], implicit_function_grad[keys_string[i]]);
				}
			}
			{
				std::shared_lock lock(mutex_);
				bicubic_interpolation(corner_point, keys_string, point, val, grad);
			}
		}
		else
		{
			logger().error("Don't yet support 3D SDF.");
		}

		for (int i = 0; i < dim; ++i)
			if (std::isnan(grad(i)))
			{
				logger().error("Nan found in gradient computation.");
				break;
			}
	}

	IntegrableFunctional SDFTrajectoryFunctional::get_target_functional(const std::string &type)
	{
		IntegrableFunctional j;
		{
			auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), 1);

				for (int q = 0; q < u.rows(); q++)
				{
					double distance;
					Eigen::MatrixXd unused_grad;
					evaluate(u.row(q) + pts.row(q), distance, unused_grad);
					val(q) = pow(distance, p);
				}
			};

			auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(u.rows(), u.cols());

				for (int q = 0; q < u.rows(); q++)
				{
					double distance;
					Eigen::MatrixXd grad;
					evaluate(u.row(q) + pts.row(q), distance, grad);
					val.row(q) = p * pow(distance, p - 1) * grad.transpose();
				}
			};

			j.set_j(j_func);
			if (type == "shape")
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

	double NodeTrajectoryFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_trajectory_functional();
		if (active_vertex_mask.size() > 0 && state.n_bases != active_vertex_mask.size())
			logger().error("vertex mask size doesn't match number of nodes!");
		return sqrt(AdjointForm::value(state, j, {}, AdjointForm::SpatialIntegralType::VERTEX_SUM, transient_integral_type));
	}

	Eigen::VectorXd NodeTrajectoryFunctional::gradient(State &state, const std::string &type)
	{
		IntegrableFunctional j = get_trajectory_functional();
		if (active_vertex_mask.size() > 0 && state.n_bases != active_vertex_mask.size())
			logger().error("vertex mask size doesn't match number of nodes!");
		Eigen::VectorXd grad;
		AdjointForm::gradient(state, j, type, grad, {}, AdjointForm::SpatialIntegralType::VERTEX_SUM, transient_integral_type);

		return grad / 2 / energy(state);
	}

	IntegrableFunctional NodeTrajectoryFunctional::get_trajectory_functional()
	{
		IntegrableFunctional j;
		{
			auto j_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
				val.setZero(1, 1);
				const int vid = params["node"];
				if (active_vertex_mask.size() > 0 && !active_vertex_mask[vid])
					return;
				assert(pts.rows() == 1);
				val(0) = (pts + u - target_vertex_positions.row(vid)).squaredNorm();
			};

			auto djdu_func = [this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
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

		double current_volume = AdjointForm::value(state, j, surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);
		logger().trace("Current volume: {}", current_volume);

		if (current_volume > max_volume)
			return pow(current_volume / max_volume - 1, 2);
		else if (current_volume < min_volume)
			return pow(current_volume / min_volume - 1, 2);
		else
			return 0.;
	}

	Eigen::VectorXd VolumeFunctional::gradient(State &state, const std::string &type)
	{
		assert(type == "shape");
		IntegrableFunctional j = get_volume_functional();

		Eigen::VectorXd grad;
		AdjointForm::gradient(state, j, type, grad, surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);
		
		double current_volume = AdjointForm::value(state, j, surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);

		double derivative = 0;
		if (current_volume > max_volume)
			derivative = 2 * (current_volume / max_volume - 1) / max_volume;
		else if (current_volume < min_volume)
			derivative = 2 * (current_volume / min_volume - 1) / min_volume;

		return derivative * grad;
	}

	IntegrableFunctional VolumeFunctional::get_volume_functional()
	{
		assert(max_volume >= min_volume);
		assert(!surface_integral);
		assert(transient_integral_type == "final");
		IntegrableFunctional j;
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setOnes(u.rows(), 1);
		});
		return j;
	}

	IntegrableFunctional HeightFunctional::get_target_functional(const std::string &type)
	{
		assert(!surface_integral);
		IntegrableFunctional j;
		j.set_name("Center");
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val = -(u.col(1) + pts.col(1));
		});

		j.set_dj_du([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			val.col(1).array() = -1;
		});

		j.set_dj_dx([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(pts.rows(), pts.cols());
			val.col(1).array() = -1;
		});

		return j;
	}

	double StressFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_stress_functional(state.formulation(), p);

		double val = AdjointForm::value(state, j, surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);

		return pow(val, 1. / p);
	}

	Eigen::VectorXd StressFunctional::gradient(State &state, const std::string &type)
	{
		IntegrableFunctional j = get_stress_functional(state.formulation(), p);

		Eigen::VectorXd grad;
		AdjointForm::gradient(state, j, type, grad, surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);

		double val = AdjointForm::value(state, j, surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);

		return (pow(val, 1. / p - 1) / p) * grad;
	}

	IntegrableFunctional StressFunctional::get_stress_functional(const std::string &formulation, const int power)
	{
		assert(!surface_integral);
		IntegrableFunctional j;
		j.set_j([formulation, power](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
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

		j.set_dj_dgradu([formulation, power](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), grad_u.cols());
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

	double ComplianceFunctional::energy(State &state)
	{
		IntegrableFunctional j = get_compliance_functional(state.formulation());

		double val = AdjointForm::value(state, j, surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);

		return pow(val, 1. / p);
	}

	Eigen::VectorXd ComplianceFunctional::gradient(State &state, const std::string &type)
	{
		IntegrableFunctional j = get_compliance_functional(state.formulation());

		Eigen::VectorXd grad;
		AdjointForm::gradient(state, j, type, grad, surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);

		double val = AdjointForm::value(state, j, surface_integral ? interested_boundary_ids_ : interested_body_ids_, surface_integral ? AdjointForm::SpatialIntegralType::SURFACE : AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);

		return (pow(val, 1. / p - 1) / p) * grad;
	}

	IntegrableFunctional ComplianceFunctional::get_compliance_functional(const std::string &formulation)
	{
		assert(!surface_integral);
		IntegrableFunctional j;
		j.set_name("Compliance");
		j.set_j([formulation, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(grad_u.rows(), 1);
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

	// double CenterTrajectoryFunctional::energy(State &state)
	// {
	// 	const int dim = state.mesh->dimension();
	// 	std::vector<IntegrableFunctional> js(dim + 1);
	// 	for (int d = 0; d < dim; d++)
	// 		js[d] = get_center_trajectory_functional(d);
	// 	js[dim] = get_volume_functional();

	// 	if (transient_integral_type != "final" && target_series.size() != state.args["time"]["time_steps"].get<int>() + 1)
	// 		logger().error("Number of center series {} doesn't match with number of time steps + 1: {}!", target_series.size(), state.args["time"]["time_steps"].get<int>() + 1);

	// 	auto func = [&](const Eigen::VectorXd &x, const json &param) {
	// 		double t = param["t"];
	// 		int step = param["step"];

	// 		Eigen::VectorXd target;
	// 		if (transient_integral_type != "final")
	// 			target = target_series[step];
	// 		else
	// 			target = target_series.back();
			
	// 		Eigen::VectorXd err = x.head(dim) / x(dim) - target;
	// 		for (int d = 0; d < dim; d++)
	// 			if (!flags_[d])
	// 				err(d) = 0;

	// 		return err.squaredNorm();
	// 	};

	// 	// return sqrt(state.J_transient(js, func));
	// 	return AdjointForm::value(state, js, func, interested_body_ids_, AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);
	// }

	// Eigen::VectorXd CenterTrajectoryFunctional::gradient(State &state, const std::string &type)
	// {
	// 	const int dim = state.mesh->dimension();
	// 	std::vector<IntegrableFunctional> js(dim + 1);
	// 	for (int d = 0; d < dim; d++)
	// 		js[d] = get_center_trajectory_functional(d);
	// 	js[dim] = get_volume_functional();

	// 	if (transient_integral_type != "final" && target_series.size() != state.args["time"]["time_steps"].get<int>() + 1)
	// 		logger().error("Number of center series {} doesn't match with number of time steps + 1: {}!", target_series.size(), state.args["time"]["time_steps"].get<int>() + 1);

	// 	auto func = [dim, this](const Eigen::VectorXd &x, const json &param) {
	// 		double t = param["t"];
	// 		int step = param["step"];

	// 		Eigen::VectorXd target;
	// 		if (transient_integral_type != "final")
	// 			target = target_series[step];
	// 		else
	// 			target = target_series.back();

	// 		Eigen::VectorXd err = x.head(dim) / x(dim) - target;
	// 		for (int d = 0; d < dim; d++)
	// 			if (!flags_[d])
	// 				err(d) = 0;
	// 		double val = err.squaredNorm();

	// 		Eigen::VectorXd grad;
	// 		grad.setZero(dim + 1);
	// 		for (int d = 0; d < dim; d++)
	// 		{
	// 			if (!flags_[d])
	// 				continue;
	// 			grad(d) = 2 / x(dim) * (x(d) / x(dim) - target(d));
	// 			grad(dim) += -x(d) * grad(d) / x(dim);
	// 		}
	// 		return grad;
	// 	};

	// 	// return state.integral_gradient(js, func, type) / 2 / energy(state);

	// 	Eigen::VectorXd grad;
	// 	AdjointForm::gradient(state, js, func, type, grad, interested_body_ids_, AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type);

	// 	return grad;
	// }

	// IntegrableFunctional CenterTrajectoryFunctional::get_center_trajectory_functional(const int d)
	// {
	// 	IntegrableFunctional j;
	// 	j.set_j([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
	// 		val = u.col(d) + pts.col(d);
	// 	});

	// 	j.set_dj_du([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
	// 		val.setZero(u.rows(), u.cols());
	// 		val.col(d).setOnes();
	// 	});

	// 	j.set_dj_dx([d, this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
	// 		val.setZero(pts.rows(), pts.cols());
	// 		val.col(d).setOnes();
	// 	});
	// 	return j;
	// }

	// IntegrableFunctional CenterTrajectoryFunctional::get_volume_functional()
	// {
	// 	IntegrableFunctional j;
	// 	j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
	// 		val.setOnes(u.rows(), 1);
	// 	});
	// 	return j;
	// }

	// void CenterTrajectoryFunctional::get_barycenter_series(State &state, std::vector<Eigen::VectorXd> &barycenters)
	// {
	// 	assert(state.problem->is_time_dependent());
	// 	const int dim = state.mesh->dimension();
	// 	const int n_steps = state.args["time"]["time_steps"].get<int>();

	// 	barycenters.clear();
	// 	Eigen::VectorXd barycenter(dim);
	// 	for (int step = 0; step <= n_steps; step++)
	// 	{
	// 		for (int d = 0; d < dim; d++)
	// 			barycenter(d) = AdjointForm::integrate_objective(state, get_center_trajectory_functional(d), state.diff_cached[step].u, interested_body_ids_, AdjointForm::SpatialIntegralType::VOLUME, step);
	// 		barycenters.push_back(barycenter);
	// 	}

	// 	double volume = AdjointForm::value(state, get_volume_functional(), interested_body_ids_, AdjointForm::SpatialIntegralType::VOLUME, "final");

	// 	for (auto &point : barycenters)
	// 		point /= volume;
	// }

} // namespace polyfem