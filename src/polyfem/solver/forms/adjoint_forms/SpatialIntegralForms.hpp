#pragma once

#include "AdjointForm.hpp"

#include <polyfem/utils/LazyCubicInterpolator.hpp>

namespace polyfem::solver
{
	class SpatialIntegralForm : public StaticForm
	{
	public:
		SpatialIntegralForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : StaticForm(variable_to_simulations), state_(state)
		{
		}

		const State &get_state() { return state_; }

		void set_integral_type(const SpatialIntegralType type) { spatial_integral_type_ = type; }

		double value_unweighted(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd compute_adjoint_rhs_unweighted_step(const Eigen::VectorXd &x, const State &state) override;
		virtual void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		virtual IntegrableFunctional get_integral_functional() const = 0;

		const State &state_;
		SpatialIntegralType spatial_integral_type_;
		std::set<int> ids_;
	};

	class StressNormForm : public SpatialIntegralForm
	{
	public:
		StressNormForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::VOLUME);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		int in_power_ = 2;
	};

	class ComplianceForm : public SpatialIntegralForm
	{
	public:
		ComplianceForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::VOLUME);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};

	class PositionForm : public SpatialIntegralForm
	{
	public:
		PositionForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::VOLUME);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		void set_dim(const int dim) { dim_ = dim; }

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		int dim_ = 0;
	};

	class TargetForm : public SpatialIntegralForm
	{
	public:
		TargetForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::SURFACE);

			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}
		~TargetForm() = default;

		void set_reference(const std::shared_ptr<const State> &target_state, const std::set<int> &reference_cached_body_ids); // target is another simulation solution
		void set_reference(const Eigen::VectorXd &disp) { target_disp = disp; }                                               // target is a constant displacement
		void set_reference(const json &func, const json &grad_func);                                                          // target is a lambda function depending on deformed position
		void set_active_dimension(const std::vector<bool> &mask) { active_dimension_mask = mask; }

	protected:
		IntegrableFunctional get_integral_functional() const override;

		std::shared_ptr<const State> target_state_;
		std::map<int, int> e_to_ref_e_;

		std::vector<bool> active_dimension_mask;
		Eigen::VectorXd target_disp;

		bool have_target_func = false;
		utils::ExpressionValue target_func;
		std::array<utils::ExpressionValue, 3> target_func_grad;
	};

	class StressForm : public SpatialIntegralForm
	{
	public:
		StressForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::VOLUME);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			dimensions_ = args["dimensions"].get<std::vector<int>>();
		}

		void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		IntegrableFunctional get_integral_functional() const override;

		std::vector<int> dimensions_;
	};

	class VolumeForm : public SpatialIntegralForm
	{
	public:
		VolumeForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::VOLUME);

			auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

	protected:
		IntegrableFunctional get_integral_functional() const override;
	};

	class SDFTargetForm : public SpatialIntegralForm
	{
	public:
		SDFTargetForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::SURFACE);

			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		void set_bspline_target(const Eigen::MatrixXd &control_points, const Eigen::VectorXd &knots, const double delta);
		void set_bspline_target(const Eigen::MatrixXd &control_points, const Eigen::VectorXd &knots_u, const Eigen::VectorXd &knots_v, const double delta);

	protected:
		IntegrableFunctional get_integral_functional() const override;

		void solution_changed(const Eigen::VectorXd &x) override;

	private:
		void compute_distance(const Eigen::MatrixXd &point, double &distance) const;

		int dim;
		double delta_;

		Eigen::MatrixXd t_or_uv_sampling;
		Eigen::MatrixXd point_sampling;
		int samples;

		std::unique_ptr<LazyCubicInterpolator> interpolation_fn;
	};
} // namespace polyfem::solver