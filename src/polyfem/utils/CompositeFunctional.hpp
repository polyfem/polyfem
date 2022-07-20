#pragma once
#include <polyfem/State.hpp>

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
		void set_interested_ids(const std::set<int> &interested_ids_) { interested_ids = interested_ids_; }
		const std::set<int> &get_interested_ids() { return interested_ids; }

		virtual double energy(State &state) = 0;
		virtual Eigen::VectorXd gradient(State &state, const std::string &type) = 0;

	protected:
		std::string functional_name;
		int p = 2;             // only used in stress functional
		bool surface_integral; // only can be true for trajectory functional
		std::string transient_integral_type = "simpson";
		std::set<int> interested_ids;
	};

	class TargetYFunctional : public CompositeFunctional
	{
	public:
		TargetYFunctional()
		{
			functional_name = "TargetY";
			surface_integral = true;
			target_y = NULL;
			target_y_derivative = NULL;
		}
		~TargetYFunctional() = default;

		void set_target_function(std::function<double(const double x)> target_y_) { target_y = target_y_; }
		void set_target_function_derivative(std::function<double(const double x)> target_y_derivative_) { target_y_derivative = target_y_derivative_; }

		double energy(State &state) override;
		Eigen::VectorXd gradient(State &state, const std::string &type) override;

	private:
		std::function<double(const double x)> target_y;
		std::function<double(const double x)> target_y_derivative;

		IntegrableFunctional get_target_y_functional();
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

		double energy(State &state) override;
		Eigen::VectorXd gradient(State &state, const std::string &type) override;

		void set_reference(State *state_ref, const State &state);

	private:
		State *state_ref_;
		std::map<int, int> e_to_ref_e_;

		IntegrableFunctional get_trajectory_functional(const std::string &derivative_type);
	};

	class SDFTrajectoryFunctional : public CompositeFunctional
	{
	public:
		SDFTrajectoryFunctional()
		{
			functional_name = "SDFTrajectory";
			surface_integral = true;
			delta.setZero(2, 1);
			delta << 0.01, 0.01;
			dim = 2;
		}
		~SDFTrajectoryFunctional() = default;

		double energy(State &state) override;
		Eigen::VectorXd gradient(State &state, const std::string &type) override;

		void set_spline_target(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &tangents)
		{
			control_points_ = control_points;
			tangents_ = tangents;
		}

	private:
		IntegrableFunctional get_trajectory_functional(const std::string &derivative_type);

		void evaluate(const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad);
		void compute_distance(const Eigen::MatrixXd &point, double &distance);

		int dim;
		double t;
		Eigen::MatrixXd delta;
		std::unordered_map<std::string, double> implicit_function;

		Eigen::MatrixXd control_points_;
		Eigen::MatrixXd tangents_;
	};

	class NodeTrajectoryFunctional : public CompositeFunctional
	{
	public:
		NodeTrajectoryFunctional() { functional_name = "NodeTrajectory"; }
		~NodeTrajectoryFunctional() = default;

		double energy(State &state) override;
		Eigen::VectorXd gradient(State &state, const std::string &type) override;

		void set_target_vertex_positions(const Eigen::MatrixXd &target) { target_vertex_positions = target; }
		void set_active_vertex_mask(const std::vector<bool> &mask) { active_vertex_mask = mask; }
		const std::vector<bool> &get_active_vertex_mask() const { return active_vertex_mask; }

	private:
		Eigen::MatrixXd target_vertex_positions;
		std::vector<bool> active_vertex_mask;

		SummableFunctional get_trajectory_functional();
	};

	class VolumeFunctional : public CompositeFunctional
	{
	public:
		VolumeFunctional()
		{
			functional_name = "Volume";
			surface_integral = false;
			transient_integral_type = "final";
			min_volume = 0;
			max_volume = std::numeric_limits<double>::max();
		}
		~VolumeFunctional() = default;

		double energy(State &state) override;
		Eigen::VectorXd gradient(State &state, const std::string &type) override;

		void set_min_volume(double min_volume_) { min_volume = min_volume_; }
		void set_max_volume(double max_volume_) { max_volume = max_volume_; }

	private:
		double min_volume;
		double max_volume;

		IntegrableFunctional get_volume_functional();
	};

	class HeightFunctional : public CompositeFunctional
	{
	public:
		HeightFunctional()
		{
			functional_name = "Height";
			surface_integral = false;
			transient_integral_type = "final";
		}
		~HeightFunctional() = default;

		double energy(State &state) override;
		Eigen::VectorXd gradient(State &state, const std::string &type) override;

	private:
		IntegrableFunctional get_height_functional();
	};

	class StressFunctional : public CompositeFunctional
	{
	public:
		StressFunctional()
		{
			functional_name = "Stress";
			surface_integral = false;
			transient_integral_type = "simpson";
		}
		~StressFunctional() = default;

		double energy(State &state) override;
		Eigen::VectorXd gradient(State &state, const std::string &type) override;

	private:
		IntegrableFunctional get_stress_functional(const std::string &formulation, const int power);
	};

	class CenterTrajectoryFunctional : public CompositeFunctional
	{
	public:
		CenterTrajectoryFunctional()
		{
			functional_name = "CenterTrajectory";
			surface_integral = false;
			transient_integral_type = "simpson";
		}
		~CenterTrajectoryFunctional() = default;

		double energy(State &state) override;
		Eigen::VectorXd gradient(State &state, const std::string &type) override;

		void set_center_series(const std::vector<Eigen::VectorXd> &target_series_) { target_series = target_series_; }

		void get_barycenter_series(State &state, std::vector<Eigen::VectorXd> &barycenters);

	private:
		std::vector<Eigen::VectorXd> target_series;
		IntegrableFunctional get_volume_functional();
		IntegrableFunctional get_center_trajectory_functional(const int d);
	};

	class CenterXYTrajectoryFunctional : public CompositeFunctional
	{
	public:
		CenterXYTrajectoryFunctional()
		{
			functional_name = "CenterTrajectory";
			surface_integral = false;
			transient_integral_type = "simpson";
		}
		~CenterXYTrajectoryFunctional() = default;

		double energy(State &state) override;
		Eigen::VectorXd gradient(State &state, const std::string &type) override;

		void set_center_series(const std::vector<Eigen::VectorXd> &target_series_) { target_series = target_series_; }

	private:
		std::vector<Eigen::VectorXd> target_series;
		IntegrableFunctional get_volume_functional();
		IntegrableFunctional get_center_trajectory_functional(const int d);
	};
} // namespace polyfem
