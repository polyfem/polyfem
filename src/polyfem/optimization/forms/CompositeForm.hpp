#pragma once

#include "AdjointForm.hpp"
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	class CompositeForm : public AdjointForm
	{
		friend AdjointForm;

	public:
		CompositeForm(const VariableToSimulationGroup &variable_to_simulations, const std::vector<std::shared_ptr<AdjointForm>> &forms) : AdjointForm(variable_to_simulations), forms_(forms) {}
		CompositeForm(const std::vector<std::shared_ptr<AdjointForm>> &forms) : AdjointForm(forms[0]->get_variable_to_simulations()), forms_(forms) {}
		virtual ~CompositeForm() {}

		virtual int n_objs() const final { return forms_.size(); }

		virtual Eigen::MatrixXd compute_reduced_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const override final;
		virtual void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override final;

		virtual double compose(const Eigen::VectorXd &inputs) const = 0;
		virtual Eigen::VectorXd compose_grad(const Eigen::VectorXd &inputs) const = 0;

		Eigen::VectorXd get_inputs(const Eigen::VectorXd &x) const;

		virtual double value_unweighted(const Eigen::VectorXd &x) const final override;
		virtual void init(const Eigen::VectorXd &x) final override;
		virtual bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final override;
		virtual double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final override;
		virtual void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) final override;
		virtual void line_search_end() final override;
		virtual void post_step(const polysolve::nonlinear::PostStepData &data) final override;
		virtual void solution_changed(const Eigen::VectorXd &new_x) final override;
		virtual bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final override;

	private:
		std::vector<std::shared_ptr<AdjointForm>> forms_;
	};
} // namespace polyfem::solver
