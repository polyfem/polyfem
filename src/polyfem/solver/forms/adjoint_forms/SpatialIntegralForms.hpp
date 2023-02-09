#pragma once

#include "AdjointForm.hpp"

namespace polyfem::solver
{
    class SpatialIntegralForm : public AdjointForm
    {
    public:
        SpatialIntegralForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args): AdjointForm(variable_to_simulations), state_(state) {}

        const State &get_state() { return state_; }

        Eigen::MatrixXd compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) override;
        
    protected:

        virtual IntegrableFunctional get_integral_functional() const = 0;
        double value_unweighted(const Eigen::VectorXd &x) const override;
        virtual void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
        
        const State &state_;
        SpatialIntegralType spatial_integral_type_;
        std::set<int> ids_;
        
        int time_step_ = 0;
    };

    class StressForm : public SpatialIntegralForm
    {
    public:
        StressForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args): SpatialIntegralForm(variable_to_simulations, state, args) 
        {
            spatial_integral_type_ = SpatialIntegralType::VOLUME;

            auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
            ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
        }

    protected:
        IntegrableFunctional get_integral_functional() const override;
        void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

    private:
        int in_power_ = 2;
    };
}