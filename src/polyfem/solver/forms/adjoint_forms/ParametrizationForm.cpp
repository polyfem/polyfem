#include "ParametrizationForm.hpp"
#include <polyfem/State.hpp>

namespace polyfem::solver
{
    Eigen::MatrixXd ParametrizationForm::compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
    { 
        return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size()); 
    }
}