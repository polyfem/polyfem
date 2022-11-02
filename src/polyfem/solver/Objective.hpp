#pragma once

#include "AdjointForm.hpp"
#include "Parameter.hpp"

namespace polyfem::solver
{
	class Objective
	{
	public:
		Objective(const State &state, const json &obj_args);

		inline double value() const
		{
			return weight_ * value_unweighted();
		}

		inline void first_derivative(const Parameter &param, Eigen::VectorXd &gradv) const
		{
			first_derivative_unweighted(param, gradv);
			gradv *= weight_;
		}

		void enable() { enabled_ = true; }
		void disable() { enabled_ = false; }
		void set_enabled(const bool enabled) { enabled_ = enabled; }
		bool enabled() const { return enabled_; }

		double weight() const { return weight_; }
		void set_weight(const double weight) { weight_ = weight; }

	private:
		const State &state_;

		std::shared_ptr<AdjointForm> form_;
		IntegrableFunctional j_;

		bool is_volume_integral;
		std::string transient_integral_type;
		std::set<int> interested_ids;

		double weight_ = 1;
		bool enabled_ = true;

		double value_unweighted() const
		{
			return form_->value(state_, j_, interested_ids, is_volume_integral ? AdjointForm::SpatialIntegralType::VOLUME : AdjointForm::SpatialIntegralType::SURFACE, transient_integral_type);
		}

		void first_derivative_unweighted(const Parameter &param, Eigen::VectorXd &gradv) const
		{
			form_->gradient(state_, j_, param, gradv, interested_ids, is_volume_integral ? AdjointForm::SpatialIntegralType::VOLUME : AdjointForm::SpatialIntegralType::SURFACE, transient_integral_type);
		}
	};
} // namespace polyfem::solver