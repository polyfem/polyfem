#pragma once
#include <polyfem/State.hpp>
#include <shared_mutex>

namespace polyfem
{
	class CompositeFunctional
	{
	public:
		CompositeFunctional() = default;
		virtual ~CompositeFunctional() = default;

		static std::shared_ptr<CompositeFunctional> create(const std::string &functional_name_);

		const std::string &get_functional_name() { return functional_name; }

		void set_power(const int power) { p = power; }
		void set_surface_integral() { surface_integral = true; }
		void set_volume_integral() { surface_integral = false; }
		void set_transient_integral_type(const std::string &transient_integral_type_) { transient_integral_type = transient_integral_type_; }
		void set_interested_ids(const std::set<int> &interested_body_ids, const std::set<int> &interested_boundary_ids)
		{
			interested_body_ids_ = interested_body_ids;
			interested_boundary_ids_ = interested_boundary_ids;
		}
		const std::set<int> &get_interested_body_ids() { return interested_body_ids_; }
		const std::set<int> &get_interested_boundary_ids() { return interested_boundary_ids_; }

		virtual double energy(State &state);
		virtual Eigen::VectorXd gradient(State &state, const std::string &type);

	protected:
		std::string functional_name;
		std::string subtype = "";
		int p = 2;             // only used in stress functional
		bool surface_integral; // only can be true for trajectory functional
		std::string transient_integral_type = "uniform";
		std::set<int> interested_body_ids_;
		std::set<int> interested_boundary_ids_;

		virtual IntegrableFunctional get_target_functional(const std::string &type = "") { return IntegrableFunctional(); };
	};

	class TrajectoryFunctional : public CompositeFunctional
	{
	public:
		TrajectoryFunctional()
		{
			functional_name = "Trajectory";
			surface_integral = true;
			state_ref_ = NULL;
		}
		~TrajectoryFunctional() = default;

		void set_reference(State *state_ref, const State &state, const std::set<int> &reference_cached_body_ids);

	private:
		State *state_ref_;
		std::map<int, int> e_to_ref_e_;

		IntegrableFunctional get_target_functional(const std::string &type) override;
	};

	class SDFTrajectoryFunctional : public CompositeFunctional
	{
	public:
		SDFTrajectoryFunctional()
		{
			functional_name = "SDFTrajectory";
			surface_integral = true;

			bicubic_mat.resize(16, 16);
			bicubic_mat << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0,
				-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0,
				0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0,
				9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1,
				-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1,
				2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0,
				-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1,
				4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1;
		}
		~SDFTrajectoryFunctional() = default;

		void set_spline_target(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &tangents, const Eigen::MatrixXd &delta)
		{
			control_points_ = control_points;
			tangents_ = tangents;
			dim = control_points.cols();
			if (dim != 2)
				logger().error("Only support 2d sdf at the moment.");
			delta_ = delta;
		}

		void evaluate(const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad);
		void compute_distance(const Eigen::MatrixXd &point, double &distance, Eigen::MatrixXd &grad);
		void bicubic_interpolation(const Eigen::MatrixXd &corner_point, const std::vector<std::string> &keys, const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad);

	private:
		IntegrableFunctional get_target_functional(const std::string &type) override;

		int dim;
		double t_cached;
		Eigen::MatrixXd delta_;
		std::unordered_map<std::string, double> implicit_function_distance;
		std::unordered_map<std::string, Eigen::MatrixXd> implicit_function_grad;

		Eigen::MatrixXd bicubic_mat;

		Eigen::MatrixXd control_points_;
		Eigen::MatrixXd tangents_;

		mutable std::shared_mutex mutex_;
	};
} // namespace polyfem
