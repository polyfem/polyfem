#pragma once

#include "SpatialIntegralForms.hpp"

#include <igl/AABB.h>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/LazyCubicInterpolator.hpp>

namespace polyfem::solver
{
	class TargetForm : public SpatialIntegralForm
	{
	public:
		TargetForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Surface);

			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}
		~TargetForm() = default;

		virtual std::string name() const override { return "target"; }

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

	class SDFTargetForm : public SpatialIntegralForm
	{
	public:
		SDFTargetForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Surface);

			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		virtual std::string name() const override { return "sdf-target"; }

		void solution_changed_step(const int time_step, const Eigen::VectorXd &new_x) override;
		void set_bspline_target(const Eigen::MatrixXd &control_points, const Eigen::VectorXd &knots, const double delta);
		void set_bspline_target(const Eigen::MatrixXd &control_points, const Eigen::VectorXd &knots_u, const Eigen::VectorXd &knots_v, const double delta);

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		void compute_distance(const Eigen::MatrixXd &point, double &distance) const;

		int dim;
		double delta_;

		Eigen::MatrixXd t_or_uv_sampling;
		Eigen::MatrixXd point_sampling;
		int samples;

		std::unique_ptr<LazyCubicInterpolator> interpolation_fn;
	};

	class MeshTargetForm : public SpatialIntegralForm
	{
	public:
		MeshTargetForm(const VariableToSimulationGroup &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::Surface);

			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
		}

		virtual std::string name() const override { return "mesh-target"; }

		void solution_changed_step(const int time_step, const Eigen::VectorXd &new_x) override;
		void set_surface_mesh_target(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const double delta);

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		int dim;
		double delta_;

		Eigen::MatrixXd V_;
		Eigen::MatrixXi F_;
		igl::AABB<Eigen::MatrixXd, 3> tree_;

		std::unique_ptr<LazyCubicInterpolator> interpolation_fn;
	};

	class NodeTargetForm : public StaticForm
	{
	public:
		NodeTargetForm(const State &state, const VariableToSimulationGroup &variable_to_simulations, const json &args);
		NodeTargetForm(const State &state, const VariableToSimulationGroup &variable_to_simulations, const std::vector<int> &active_nodes_, const Eigen::MatrixXd &target_vertex_positions_);
		~NodeTargetForm() = default;

		std::string name() const override { return "node-target"; }

		Eigen::VectorXd compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const override;
		double value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const override;
		void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		const State &state_;

		Eigen::MatrixXd target_vertex_positions;
		std::vector<int> active_nodes;
	};

	class BarycenterTargetForm : public StaticForm
	{
	public:
		BarycenterTargetForm(const VariableToSimulationGroup &variable_to_simulations, const json &args, const std::shared_ptr<State> &state1, const std::shared_ptr<State> &state2);

		Eigen::VectorXd compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const override;
		void compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
		double value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const override;

	private:
		std::vector<std::unique_ptr<PositionForm>> center1, center2;
		int dim;
	};

	class MinTargetDistForm : public AdjointForm
	{
	public:
		MinTargetDistForm(const VariableToSimulationGroup &variable_to_simulations, const std::vector<int> &steps, const Eigen::VectorXd &target, const json &args, const std::shared_ptr<State> &state);
		virtual ~MinTargetDistForm() = default;

		Eigen::MatrixXd compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const override;
		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;

	private:
		static double eval1(Eigen::VectorXd &x)
		{
			return 1. / x.array().inverse().sum();
		}
		static Eigen::VectorXd eval1_grad(Eigen::VectorXd &x)
		{
			return x.array().pow(-2.0) * pow(eval1(x), 2);
		}

		double eval2(Eigen::VectorXd &x) const
		{
			assert(x.size() == dim + 1);
			double out = 0;
			for (int d = 0; d < dim; d++)
				out += pow(x(d) / x(dim) - target_(d), 2);
			return out;
		}
		Eigen::VectorXd eval2_grad(Eigen::VectorXd &x) const
		{
			Eigen::VectorXd g = Eigen::VectorXd::Zero(dim + 1);
			for (int d = 0; d < dim; d++)
			{
				const double tmp = 2 * (x(d) / x(dim) - target_(d));
				g(d) += tmp / x(dim);
				g(dim) += tmp * (-x(d) / x(dim) / x(dim));
			}
			return g;
		}

		const std::vector<int> steps_;
		const Eigen::VectorXd target_;

		int dim;
		std::vector<std::unique_ptr<StaticForm>> objs;
	};
} // namespace polyfem::solver