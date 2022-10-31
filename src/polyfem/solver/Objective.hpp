#pragma once

#include "AdjointForm.hpp"

namespace polyfem::solver
{
	class Objective
	{
	public:
		Objective(const State &state, const json &obj_args): state_(state)
        {
            std::vector<int> volume_selection = obj_args["volume_selection"].get<std::vector<int>>();
            std::vector<int> surface_selection = obj_args["surface_selection"].get<std::vector<int>>();

            if (volume_selection.size() > 0 && surface_selection.size() > 0)
                log_and_throw_error("Can't specify both volume and surface in one functional!");

            if (volume_selection.size() > 0)
                interested_ids = std::set(volume_selection.begin(), volume_selection.end());
            else if (surface_selection.size() > 0)
                interested_ids = std::set(surface_selection.begin(), surface_selection.end());
            else
                log_and_throw_error("No domain is selected for functional!");

            is_volume_integral = volume_selection.size() > 0;
            transient_integral_type = obj_args["transient_integral_type"];

            // TODO: build form based on obj_args["type"]
        }

		inline double value() const
		{
            return 0.0;
		}

		inline void first_derivative(const std::string &param, Eigen::VectorXd &gradv) const
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
        bool is_volume_integral;
        std::string transient_integral_type;
        std::set<int> interested_ids;

		double weight_ = 1;
		bool enabled_ = true;

		double value_unweighted() const
        {
            return form_->value(state_, interested_ids, is_volume_integral, transient_integral_type);
        }
        
		void first_derivative_unweighted(const std::string &param, Eigen::VectorXd &gradv) const
        {
            form_->gradient(state_, interested_ids, is_volume_integral, param, gradv, transient_integral_type);
        }
    };
}