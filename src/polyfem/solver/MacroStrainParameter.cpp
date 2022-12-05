#include "MacroStrainParameter.hpp"

namespace polyfem
{
    MacroStrainParameter::MacroStrainParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args) : Parameter(states_ptr, args)
    {
        parameter_name_ = "macro_strain";

        dim = get_state().mesh->dimension();
        full_dim_ = dim*dim;
        optimization_dim_ = dim*dim;

        max_change_ = args["max_change"];

        initial_disp_grad.setZero(dim * dim);
        if (args["initial"].is_array())
        {
            if (args["initial"].size() != dim*dim)
                logger().error("Invalid size for disp grad initial guess!");
            else
                initial_disp_grad = args["initial"];
        }
        pre_solve(initial_disp_grad);

        for (auto state : states_ptr_)
            if (state->problem->is_time_dependent())
                log_and_throw_error("Macro Strain parameter optimization is only supported in static simulations!");
    }

    Eigen::VectorXd MacroStrainParameter::initial_guess() const
    {
        return initial_disp_grad;
    }
    
    bool MacroStrainParameter::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
    {
        if ((x1 - x0).cwiseAbs().maxCoeff() > max_change_)
            return false;
        
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(dim, dim) + utils::unflatten(x1, dim);
        if (F.determinant() <= 0)
            return false;
        
        return true;
    }

    bool MacroStrainParameter::pre_solve(const Eigen::VectorXd &newX)
    {
        Eigen::MatrixXd disp_grad = utils::unflatten(newX, dim);
		for (auto state : states_ptr_)
        {
            state->disp_offset.setZero(state->n_bases * dim, 1);
            for (int i = 0; i < state->n_bases; i++)
            {
                state->disp_offset.block(i * dim, 0, dim, 1) = disp_grad * state->mesh_nodes->node_position(i).transpose();
            }
        }
		return true;
    }

    Eigen::VectorXd MacroStrainParameter::get_lower_bound(const Eigen::VectorXd &x) const 
    {
        return (x.array() - max_change_).matrix();
    }

    Eigen::VectorXd MacroStrainParameter::get_upper_bound(const Eigen::VectorXd &x) const 
    {
        return (x.array() + max_change_).matrix();
    }
}