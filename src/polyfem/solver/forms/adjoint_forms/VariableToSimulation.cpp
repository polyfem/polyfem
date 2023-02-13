#include "VariableToSimulation.hpp"

namespace polyfem::solver
{
    void ShapeVariableToSimulation::update_state(const Eigen::VectorXd &state_variable)
    {
        Eigen::MatrixXd V_rest, V;
        Eigen::MatrixXi F;
        state_ptr_->get_vf(V_rest, F);
        assert(state_variable.size() == V_rest.size());
        V = utils::unflatten(state_variable, V_rest.cols());

        state_ptr_->set_mesh_vertices(V);
    }

    void ElasticVariableToSimulation::update_state(const Eigen::VectorXd &state_variable)
    {
        const int n_elem = state_ptr_->bases.size();
        assert(n_elem * 2 == state_variable.size());
        state_ptr_->assembler.update_lame_params(state_variable.segment(0, n_elem), state_variable.segment(n_elem, n_elem));
    }

    void FrictionCoeffientVariableToSimulation::update_state(const Eigen::VectorXd &state_variable)
    {
        assert(state_variable.size() == 1);
        assert(state_variable(0) >= 0);
        state_ptr_->args["contact"]["friction_coefficient"] = state_variable(0);
    }

    void DampingCoeffientVariableToSimulation::update_state(const Eigen::VectorXd &state_variable)
    {
        assert(state_variable.size() == 2);
        json damping_param = {
            {"psi", state_variable(0)},
            {"phi", state_variable(1)},
        };
        state_ptr_->assembler.add_multimaterial(0, damping_param);
        logger().info("Current damping params: {}, {}", state_variable(0), state_variable(1));
    }

    void InitialConditionVariableToSimulation::update_state(const Eigen::VectorXd &state_variable)
    {
        assert(state_variable.size() == state_ptr_->ndof() * 2);
        state_ptr_->initial_sol_update = state_variable.head(state_ptr_->ndof());
        state_ptr_->initial_vel_update = state_variable.tail(state_ptr_->ndof());
    }

    void DirichletVariableToSimulation::update_state(const Eigen::VectorXd &state_variable)
    {
        log_and_throw_error("Dirichlet variable to simulation not implemented!");
    }

    void MacroStrainVariableToSimulation::update_state(const Eigen::VectorXd &state_variable)
    {
        assert(state_variable.size() == state_ptr_->mesh->dimension() * state_ptr_->mesh->dimension());
        state_ptr_->disp_grad = utils::unflatten(state_variable, state_ptr_->mesh->dimension());
    }
}