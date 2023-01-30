#pragma once

#include "AdjointForm.hpp"
#include "Objective.hpp"

#include <polyfem/solver/forms/ContactForm.hpp>

#include <polyfem/utils/UnsignedDistanceFunction.hpp>
#include <nanospline/BSpline.h>
#include <nanospline/BSplinePatch.h>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

#include <shared_mutex>
#include <array>

namespace polyfem::solver
{
	class SpatialIntegralObjective : public StaticObjective
	{
	public:
		SpatialIntegralObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args);
		SpatialIntegralObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> macro_strain_param, const json &args);
		virtual ~SpatialIntegralObjective() = default;

		double value() override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;

		virtual IntegrableFunctional get_integral_functional() = 0;

		const State &get_state() { return state_; }

	protected:
		const State &state_;
		std::shared_ptr<const Parameter> shape_param_;
		std::shared_ptr<const Parameter> macro_strain_param_;
		AdjointForm::SpatialIntegralType spatial_integral_type_;
		std::set<int> interested_ids_;
	};

	class StressObjective : public SpatialIntegralObjective
	{
	public:
		StressObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args, bool has_integral_sqrt = true);
		~StressObjective() = default;

		double value() override;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;
		IntegrableFunctional get_integral_functional() override;

	protected:
		int in_power_;
		bool out_sqrt_;
		std::string formulation_;

		std::shared_ptr<const Parameter> elastic_param_; // stress depends on elastic param
	};

	class PositionObjective : public SpatialIntegralObjective
	{
	public:
		PositionObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args);
		~PositionObjective() = default;

		void set_dim(const int dim) { dim_ = dim; }
		void set_integral_type(const AdjointForm::SpatialIntegralType type) { spatial_integral_type_ = type; }

		IntegrableFunctional get_integral_functional() override;

	protected:
		int dim_ = 0; // integrate the "dim" dimension
	};

	class ComplianceObjective : public SpatialIntegralObjective
	{
	public:
		ComplianceObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const std::shared_ptr<const Parameter> &elastic_param, const json &args);
		~ComplianceObjective() = default;

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		IntegrableFunctional get_integral_functional() override;

	protected:
		std::string formulation_;

		std::shared_ptr<const Parameter> elastic_param_; // stress depends on elastic param
	};

	class TargetObjective : public SpatialIntegralObjective
	{
	public:
		TargetObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args) : SpatialIntegralObjective(state, shape_param, args)
		{
			spatial_integral_type_ = AdjointForm::SpatialIntegralType::SURFACE;
			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
			active_dimension_mask.assign(true, state_.mesh->dimension());
		}
		~TargetObjective() = default;

		IntegrableFunctional get_integral_functional() override;
		void set_reference(const std::shared_ptr<const State> &target_state, const std::set<int> &reference_cached_body_ids); // target is another simulation solution
		void set_reference(const Eigen::VectorXd &disp) { target_disp = disp; }                                               // target is a constant displacement
		void set_reference(const json &func, const json &grad_func);                                                          // target is a lambda function depending on deformed position
		void set_active_dimension(const std::vector<bool> &mask) { active_dimension_mask = mask; }

	protected:
		std::shared_ptr<const State> target_state_;
		std::map<int, int> e_to_ref_e_;

		std::vector<bool> active_dimension_mask;
		Eigen::VectorXd target_disp;

		bool have_target_func = false;
		utils::ExpressionValue target_func;
		std::array<utils::ExpressionValue, 3> target_func_grad;
	};

	class SDFTargetObjective : public SpatialIntegralObjective
	{
	public:
		SDFTargetObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args) : SpatialIntegralObjective(state, shape_param, args)
		{
			spatial_integral_type_ = AdjointForm::SpatialIntegralType::SURFACE;
			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		~SDFTargetObjective() = default;

		IntegrableFunctional get_integral_functional() override;

		void set_spline_target(const Eigen::MatrixXd &control_points, const Eigen::MatrixXd &tangents, const Eigen::MatrixXd &delta)
		{
			assert(false);
		}

		void set_bspline_target(const Eigen::MatrixXd &control_points, const Eigen::VectorXd &knots, const double delta)
		{
			dim = control_points.cols();
			delta_ = delta;
			assert(dim == 2);

			samples = 100;

			nanospline::BSpline<double, 2, 3> curve;
			curve.set_control_points(control_points);
			curve.set_knots(knots);

			t_or_uv_sampling = Eigen::VectorXd::LinSpaced(samples, 0, 1);
			point_sampling.setZero(samples, 2);
			for (int i = 0; i < t_or_uv_sampling.size(); ++i)
				point_sampling.row(i) = curve.evaluate(t_or_uv_sampling(i));

			Eigen::MatrixXi edges(samples - 1, 2);
			edges.col(0) = Eigen::VectorXi::LinSpaced(samples - 1, 0, samples - 2);
			edges.col(1) = Eigen::VectorXi::LinSpaced(samples - 1, 1, samples - 1);
			io::OBJWriter::write(state_.resolve_output_path(fmt::format("spline_target_{:d}.obj", rand() % 100)), point_sampling, edges);

			distance_fn = std::make_unique<UnsignedDistanceFunction>(dim, delta_);
		}

		void set_bspline_target(const Eigen::MatrixXd &control_points, const Eigen::VectorXd &knots_u, const Eigen::VectorXd &knots_v, const double delta)
		{

			dim = control_points.cols();
			delta_ = delta;
			assert(dim == 3);

			samples = 100;

			nanospline::BSplinePatch<double, 3, 3, 3> patch;
			patch.set_control_grid(control_points);
			patch.set_knots_u(knots_u);
			patch.set_knots_v(knots_v);
			patch.initialize();

			t_or_uv_sampling.resize(samples * samples, 2);
			for (int i = 0; i < samples; ++i)
			{
				t_or_uv_sampling.block(i * samples, 0, samples, 1) = Eigen::VectorXd::LinSpaced(samples, 0, 1);
				t_or_uv_sampling.block(i * samples, 1, samples, 1) = (double)i / (samples - 1) * Eigen::VectorXd::Ones(samples);
			}
			point_sampling.setZero(samples * samples, 3);
			for (int i = 0; i < t_or_uv_sampling.rows(); ++i)
			{
				point_sampling.row(i) = patch.evaluate(t_or_uv_sampling(i, 0), t_or_uv_sampling(i, 1));
			}

			distance_fn = std::make_unique<UnsignedDistanceFunction>(dim, delta_);
		}

		inline void evaluate(const Eigen::MatrixXd &point, double &val, Eigen::MatrixXd &grad)
		{
			distance_fn->evaluate([this](const Eigen::MatrixXd &point, double &distance) { compute_distance(point, distance); }, point, val, grad);
		}
		void compute_distance(const Eigen::MatrixXd &point, double &distance);

	protected:
		int dim;
		double delta_;

		Eigen::MatrixXd t_or_uv_sampling;
		Eigen::MatrixXd point_sampling;
		int samples;

		std::unique_ptr<UnsignedDistanceFunction> distance_fn;
	};

	class BarycenterTargetObjective : public StaticObjective
	{
	public:
		BarycenterTargetObjective(const State &state, const std::shared_ptr<const Parameter> shape_param, const json &args, const Eigen::MatrixXd &target);
		~BarycenterTargetObjective() = default;

		double value() override;
		Eigen::VectorXd get_target() const;
		void set_time_step(int time_step) override;

		int dim() const { return dim_; }

		Eigen::VectorXd compute_partial_gradient(const Parameter &param, const Eigen::VectorXd &param_value) override;
		Eigen::VectorXd compute_adjoint_rhs_step(const State &state) override;

		Eigen::VectorXd get_barycenter() const;

	protected:
		int dim_ = -1;
		std::vector<std::shared_ptr<PositionObjective>> objp;
		std::shared_ptr<VolumeObjective> objv;
		Eigen::MatrixXd target_; // N/1 by 3/2
	};
} // namespace polyfem::solver