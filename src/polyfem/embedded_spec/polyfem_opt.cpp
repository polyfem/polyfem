#include "polyfem_opt.hpp"

#include <string>

namespace jse
{
namespace embed
{
namespace polyfem_opt
{

const nlohmann::json &spec()
{
    static const nlohmann::json value = []() {
        std::string text;
        text.reserve(44538);
        text += R"JSE_JSON(
[
    {
        "doc": "Root of the configuration file.",
        "optional": [
            "output",
            "solver",
            "stopping_conditions",
            "functionals",
            "compute_objective",
            "root_path"
        ],
        "pointer": "/",
        "required": [
            "parameters",
            "variable_to_simulation",
            "states"
        ],
        "type": "object"
    },
    {
        "default": "",
        "doc": "Path for all relative paths, set automatically to the folder containing this JSON.",
        "pointer": "/root_path",
        "type": "string"
    },
    {
        "default": [],
        "doc": "A list of functionals. Their sum is the functional being optimized.",
        "pointer": "/functionals",
        "type": "list"
    },
    {
        "default": [],
        "doc": "A list of functionals. The optimization stops if these functionals are all negative, even if the objective gradient norm is not small enough.",
        "pointer": "/stopping_conditions",
        "type": "list"
    },
    {
        "doc": "Specify a numebr of simulations used in the optimization.",
        "pointer": "/states",
        "type": "list"
    },
    {
        "optional": [
            "initial_guess"
        ],
        "pointer": "/states/*",
        "required": [
            "path"
        ],
        "type": "object"
    },
    {
        "doc": "The json file path for this state.",
        "pointer": "/states/*/path",
        "type": "file"
    },
    {
        "default": -1,
        "doc": "Specify the state ID, whose solution is used to initialize the solve in this state. Only relevant for nonlinear problems.",
        "pointer": "/states/*/initial_guess",
        "type": "int"
    },
    {
        "default": -1,
        "doc": "Specify the state ID, whose stiffness matrix factorization is used to solve this state. Only relevant for linear problems.",
        "pointer": "/states/*/reuse_stiffness_factorization",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Optimization output options",
        "optional": [
            "log",
            "save_frequency",
            "directory",
            "solution"
        ],
        "pointer": "/output",
        "type": "object"
    },
    {
        "default": "",
        "doc": "Directory for output files.",
        "pointer": "/output/directory",
        "type": "string"
    },
    {
        "default": 1,
        "doc": "Optimization output frequency",
        "pointer": "/output/save_frequency",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Setting for the output log.",
        "optional": [
            "level",
            "file_level",
            "path",
            "quiet"
        ],
        "pointer": "/output/log",
        "type": "object"
    },
    {
        "doc": "Level of logging, 0 trace, 1 debug, 2 info, 3 warning, 4 error, 5 critical, and 6 off.",
        "max": 6,
        "min": 0,
        "pointer": "/output/log/level",
        "type": "int"
    },
    {
        "default": "debug",
        "doc": "Level of logging.",
        "options": [
            "trace",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "off"
        ],
        "pointer": "/output/log/level",
        "type": "string"
    },
    {
        "doc": "Level of logging to a file, 0 trace, 1 debug, 2 info, 3 warning, 4 error, 5 critical, and 6 off.",
        "max": 6,
        "min": 0,
        "pointer": "/output/log/file_level",
        "type": "int"
    },
    {
        "default": "trace",
        "doc": "Level of logging.",
        "options": [
            "trace",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "off"
        ],
        "pointer": "/output/log/file_level",
        "type": "string"
    },
    {
        "default": "",
        "doc": "File where to save the log; empty string is output to terminal.",
        "pointer": "/output/log/path",
        "type": "string"
    },
    {
        "default": false,
        "doc": "Disable cout for logging.",
        "pointer": "/output/log/quiet",
        "type": "bool"
    },
    {
        "default": "",
        "doc": "Export optimization variables to file at every iteration.",
        "pointer": "/output/solution",
        "type": "file"
    },
    {
        "default": null,
        "doc": "Optimization solver parameters.",
        "optional": [
            "nonlinear",
            "advanced",
            "max_threads"
        ],
        "pointer": "/solver",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Maximum number of threads used; 0 is unlimited.",
        "min": 0,
        "pointer": "/solver/max_threads",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Settings for nonlinear solver. Interior-loop linear solver settings are defined in the solver/linear section.",
        "optional": [
            "solver",
            "x_delta_tol",
            "grad_norm_tol",
            "rel_grad_norm_tol",
            "newton_decrement_tol",
            "rel_x_delta_tol",
            "first_grad_norm_tol",
            "norm_type",
            "max_iterations",
            "iterations_per_strategy",
            "line_search",
            "allow_out_of_iterations",
            "L-BFGS",
            "L-BFGS-B",
            "Newton",
            "ADAM",
            "StochasticADAM",
            "StochasticGradientDescent",
            "box_constraints",
            "advanced"
        ],
        "pointer": "/solver/nonlinear",
        "type": "object"
    },
    {
        "default": "Newton",
        "doc": "Nonlinear solver type",
        "options": [
            "Newton",
            "DenseNewton",
            "GradientDescent",
            "ADAM",
            "StochasticADAM",
            "StochasticGradientDescent",
            "L-BFGS",
            "BFGS",
            "L-BFGS-B",
            "MMA"
        ],
        "pointer": "/solver/nonlinear/solver",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Stopping criterion: minimal change of the variables x for the iterations to continue. Computed as the L2 norm of x divide by the time step.",
        "min": 0,
        "pointer": "/solver/nonlinear/x_delta_tol",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Stopping criterion: minimal change of the variables x for the iterations to continue relative to first step in nonlinear solve.",
        "min": 0,
        "pointer": "/solver/nonlinear/rel_x_delta_tol",
        "type": "float"
    },
    {
        "default": 1e-10,
        "doc": "Stopping criterion: minimal gradient for the iterations to continue relative to first step in nonlinear solve.",
        "min": 0,
        "pointer": "/solver/nonlinear/rel_grad_norm_tol",
        "type": "float"
    },
    {
        "default": 0,
        "doc": "Stopping criterion: minimal change of energy (as estimated by Newton decrement) for the iterations to continue.",
        "min": 0,
        "pointer": "/solver/nonlinear/newton_decrement_tol",
        "type": "float"
    },
    {
        "default": 1e-10,
        "doc": "Stopping criterion: Minimal gradient norm for the iterations to continue.",
        "min": 0,
        "pointer": "/solver/nonlinear/grad_norm_tol",
        "type": "float"
    },
    {
        "default": 1e-12,
        "doc": "Minimal gradient norm for the iterations to not start, assume we already are at a minimum.",
        "pointer": "/solver/nonlinear/first_grad_norm_tol",
        "type": "float"
    },
    {
        "default": "L2",
        "doc": "Norm to use when computing stopping criteria in nonlinear solve.",
        "options": [
            "Euclidean",
            "L2",
            "Linf"
        ],
        "pointer": "/solver/nonlinear/norm_type",
       )JSE_JSON";
        text += R"JSE_JSON( "type": "string"
    },
    {
        "default": 500,
        "doc": "Maximum number of iterations for a nonlinear solve.",
        "pointer": "/solver/nonlinear/max_iterations",
        "type": "int"
    },
    {
        "default": 5,
        "doc": "Number of iterations for every substrategy before reset.",
        "pointer": "/solver/nonlinear/iterations_per_strategy",
        "type": "int"
    },
    {
        "doc": "Number of iterations for every substrategy before reset.",
        "pointer": "/solver/nonlinear/iterations_per_strategy",
        "type": "list"
    },
    {
        "default": 5,
        "doc": "Number of iterations for every substrategy before reset.",
        "pointer": "/solver/nonlinear/iterations_per_strategy/*",
        "type": "int"
    },
    {
        "default": false,
        "doc": "If false (default), an exception will be thrown when the nonlinear solver reaches the maximum number of iterations.",
        "pointer": "/solver/nonlinear/allow_out_of_iterations",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Options for LBFGS.",
        "optional": [
            "history_size"
        ],
        "pointer": "/solver/nonlinear/L-BFGS",
        "type": "object"
    },
    {
        "default": 6,
        "doc": "The number of corrections to approximate the inverse Hessian matrix.",
        "pointer": "/solver/nonlinear/L-BFGS/history_size",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Options for the boxed L-BFGS.",
        "optional": [
            "history_size"
        ],
        "pointer": "/solver/nonlinear/L-BFGS-B",
        "type": "object"
    },
    {
        "default": 6,
        "doc": "The number of corrections to approximate the inverse Hessian matrix.",
        "pointer": "/solver/nonlinear/L-BFGS-B/history_size",
        "type": "int"
    },
    {
        "default": null,
        "doc": "Options for Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc",
            "force_psd_projection",
            "use_psd_projection",
            "use_psd_projection_in_regularized"
        ],
        "pointer": "/solver/nonlinear/Newton",
        "type": "object"
    },
    {
        "default": 1e-05,
        "doc": "Tolerance of the linear system residual. If residual is above, the direction is rejected.",
        "pointer": "/solver/nonlinear/Newton/residual_tolerance",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Minimum regulariztion weight.",
        "pointer": "/solver/nonlinear/Newton/reg_weight_min",
        "type": "float"
    },
    {
        "default": 100000000.0,
        "doc": "Maximum regulariztion weight.",
        "pointer": "/solver/nonlinear/Newton/reg_weight_max",
        "type": "float"
    },
    {
        "default": 10,
        "doc": "Regulariztion weight increment.",
        "pointer": "/solver/nonlinear/Newton/reg_weight_inc",
        "type": "float"
    },
    {
        "default": false,
        "doc": "Force the Hessian to be PSD when using second order solvers (i.e., Newton's method).",
        "pointer": "/solver/nonlinear/Newton/force_psd_projection",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "Use PSD as fallback using second order solvers (i.e., Newton's method).",
        "pointer": "/solver/nonlinear/Newton/use_psd_projection",
        "type": "bool"
    },
    {
        "default": true,
        "doc": "Use PSD in regularized Newton.",
        "pointer": "/solver/nonlinear/Newton/use_psd_projection_in_regularized",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon"
        ],
        "pointer": "/solver/nonlinear/ADAM",
        "type": "object"
    },
    {
        "default": 0.001,
        "doc": "Parameter alpha for ADAM.",
        "pointer": "/solver/nonlinear/ADAM/alpha",
        "type": "float"
    },
    {
        "default": 0.9,
        "doc": "Parameter beta_1 for ADAM.",
        "pointer": "/solver/nonlinear/ADAM/beta_1",
        "type": "float"
    },
    {
        "default": 0.999,
        "doc": "Parameter beta_2 for ADAM.",
        "pointer": "/solver/nonlinear/ADAM/beta_2",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Parameter epsilon for ADAM.",
        "pointer": "/solver/nonlinear/ADAM/epsilon",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon",
            "erase_component_probability"
        ],
        "pointer": "/solver/nonlinear/StochasticADAM",
        "type": "object"
    },
    {
        "default": 0.001,
        "doc": "Parameter alpha for ADAM.",
        "pointer": "/solver/nonlinear/StochasticADAM/alpha",
        "type": "float"
    },
    {
        "default": 0.9,
        "doc": "Parameter beta_1 for ADAM.",
        "pointer": "/solver/nonlinear/StochasticADAM/beta_1",
        "type": "float"
    },
    {
        "default": 0.999,
        "doc": "Parameter beta_2 for ADAM.",
        "pointer": "/solver/nonlinear/StochasticADAM/beta_2",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Parameter epsilon for ADAM.",
        "pointer": "/solver/nonlinear/StochasticADAM/epsilon",
        "type": "float"
    },
    {
        "default": 0.3,
        "doc": "Probability of erasing a component on the gradient for ADAM.",
        "pointer": "/solver/nonlinear/StochasticADAM/erase_component_probability",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for Stochastic Gradient Descent.",
        "optional": [
            "erase_component_probability"
        ],
        "pointer": "/solver/nonlinear/StochasticGradientDescent",
        "type": "object"
    },
    {
        "default": 0.3,
        "doc": "Probability of erasing a component on the gradient for StochasticGradientDescent.",
        "pointer": "/solver/nonlinear/StochasticGradientDescent/erase_component_probability",
        "type": "float"
    },
    {
        "doc": "List of solvers for ballback. Eg, [{'type':'Newton'}, {'type':'L-BFGS'}, {'type':'GradientDescent'}] will solve using Newton, in case of failure will fallback to L-BFGS and eventually to GradientDescent",
        "pointer": "/solver/nonlinear/solver",
        "type": "list"
    },
    {
        "doc": "Options for Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Newton"
    },
    {
        "doc": "Options for projected Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ProjectedNewton"
    },
    {
        "doc": "Options for regularized Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "RegularizedNewton"
    },
    {
        "doc": "Options for regularized projected Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "RegularizedProjectedNewton"
    })JSE_JSON";
        text += R"JSE_JSON(,
    {
        "doc": "Options for Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseNewton"
    },
    {
        "doc": "Options for projected Newton.",
        "optional": [
            "residual_tolerance"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseProjectedNewton"
    },
    {
        "doc": "Options for regularized Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseRegularizedNewton"
    },
    {
        "doc": "Options for projected regularized Newton.",
        "optional": [
            "residual_tolerance",
            "reg_weight_min",
            "reg_weight_max",
            "reg_weight_inc"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "DenseRegularizedProjectedNewton"
    },
    {
        "doc": "Options for Gradient Descent.",
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "GradientDescent"
    },
    {
        "doc": "Options for Stochastic Gradient Descent.",
        "optional": [
            "erase_component_probability"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "StochasticGradientDescent"
    },
    {
        "doc": "Options for L-BFGS.",
        "optional": [
            "history_size"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "L-BFGS"
    },
    {
        "doc": "Options for BFGS.",
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "BFGS"
    },
    {
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "ADAM"
    },
    {
        "doc": "Options for ADAM.",
        "optional": [
            "alpha",
            "beta_1",
            "beta_2",
            "epsilon",
            "erase_component_probability"
        ],
        "pointer": "/solver/nonlinear/solver/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "StochasticADAM"
    },
    {
        "doc": "Nonlinear solver type",
        "options": [
            "Newton",
            "DenseNewton",
            "ProjectedNewton",
            "DenseProjectedNewton",
            "RegularizedNewton",
            "DenseRegularizedNewton",
            "RegularizedProjectedNewton",
            "DenseRegularizedProjectedNewton",
            "GradientDescent",
            "StochasticGradientDescent",
            "ADAM",
            "StochasticADAM",
            "L-BFGS",
            "BFGS"
        ],
        "pointer": "/solver/nonlinear/solver/*/type",
        "type": "string"
    },
    {
        "default": 1e-05,
        "doc": "Tolerance of the linear system residual. If residual is above, the direction is rejected.",
        "pointer": "/solver/nonlinear/solver/*/residual_tolerance",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Minimum regulariztion weight.",
        "pointer": "/solver/nonlinear/solver/*/reg_weight_min",
        "type": "float"
    },
    {
        "default": 100000000.0,
        "doc": "Maximum regulariztion weight.",
        "pointer": "/solver/nonlinear/solver/*/reg_weight_max",
        "type": "float"
    },
    {
        "default": 10,
        "doc": "Regulariztion weight increment.",
        "pointer": "/solver/nonlinear/solver/*/reg_weight_inc",
        "type": "float"
    },
    {
        "default": 0.3,
        "doc": "Probability of erasing a component on the gradient for stochastic solvers.",
        "pointer": "/solver/nonlinear/solver/*/erase_component_probability",
        "type": "float"
    },
    {
        "default": 6,
        "doc": "The number of corrections to approximate the inverse Hessian matrix.",
        "pointer": "/solver/nonlinear/solver/*/history_size",
        "type": "int"
    },
    {
        "default": 0.001,
        "doc": "Parameter alpha for ADAM.",
        "pointer": "/solver/nonlinear/solver/*/alpha",
        "type": "float"
    },
    {
        "default": 0.9,
        "doc": "Parameter beta_1 for ADAM.",
        "pointer": "/solver/nonlinear/solver/*/beta_1",
        "type": "float"
    },
    {
        "default": 0.999,
        "doc": "Parameter beta_2 for ADAM.",
        "pointer": "/solver/nonlinear/solver/*/beta_2",
        "type": "float"
    },
    {
        "default": 1e-08,
        "doc": "Parameter epsilon for ADAM.",
        "pointer": "/solver/nonlinear/solver/*/epsilon",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Settings for line-search in the nonlinear solver",
        "optional": [
            "method",
            "use_grad_norm_tol",
            "min_step_size",
            "max_step_size_iter",
            "min_step_size_final",
            "max_step_size_iter_final",
            "default_init_step_size",
            "step_ratio",
            "Armijo",
            "RobustArmijo"
        ],
        "pointer": "/solver/nonlinear/line_search",
        "type": "object"
    },
    {
        "default": "RobustArmijo",
        "doc": "Line-search type",
        "options": [
            "Armijo",
            "RobustArmijo",
            "Backtracking",
            "None"
        ],
        "pointer": "/solver/nonlinear/line_search/method",
        "type": "string"
    },
    {
        "default": 1e-06,
        "doc": "When the energy is smaller than use_grad_norm_tol, line-search uses norm of gradient instead of energy",
        "pointer": "/solver/nonlinear/line_search/use_grad_norm_tol",
        "type": "float"
    },
    {
        "default": 1e-10,
        "doc": "Mimimum step size",
        "pointer": "/solver/nonlinear/line_search/min_step_size",
        "type": "float"
    },
    {
        "default": 30,
        "doc": "Number of iterations",
        "pointer": "/solver/nonlinear/line_search/max_step_size_iter",
        "type": "int"
    },
    {
        "default": 1e-20,
        "doc": "Mimimum step size for last descent strategy",
        "pointer": "/solver/nonlinear/line_search/min_step_size_final",
        "type": "float"
    },
    {
        "default": 100,
        "doc": "Number of iterations for last descent strategy",
        "pointer": "/solver/nonlinear/line_search/max_step_size_iter_final",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "Initial step size",
        "pointer": "/solver/nonlinear/line_search/default_init_step_size",
        "type": "float"
    },
    {
        "default": 0.5,
        "doc": "Ratio used to decrease the step",
        "pointer": "/solver/nonlinear/line_search/step_ratio",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for Armijo.",
        "optional": [
            "c"
        ],
        "pointer": "/solver/nonlinear/line_search/Armijo",
        "type": "object"
    },
    {
        "default": 0.0001,
        "doc": "Armijo c parameter.",
      )JSE_JSON";
        text += R"JSE_JSON(  "min_value": 0,
        "pointer": "/solver/nonlinear/line_search/Armijo/c",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Options for RobustArmijo.",
        "optional": [
            "delta_relative_tolerance"
        ],
        "pointer": "/solver/nonlinear/line_search/RobustArmijo",
        "type": "object"
    },
    {
        "default": 0.1,
        "doc": "Relative tolerance on E to switch to approximate.",
        "min_value": 0,
        "pointer": "/solver/nonlinear/line_search/RobustArmijo/delta_relative_tolerance",
        "type": "float"
    },
    {
        "default": null,
        "optional": [
            "bounds",
            "max_change"
        ],
        "pointer": "/solver/nonlinear/box_constraints",
        "type": "object"
    },
    {
        "default": [],
        "doc": "Box constraints on optimization variables.",
        "pointer": "/solver/nonlinear/box_constraints/bounds",
        "type": "list"
    },
    {
        "doc": "Box constraint values on optimization variables.",
        "pointer": "/solver/nonlinear/box_constraints/bounds/*",
        "type": "list"
    },
    {
        "doc": "Box constraint values on optimization variables.",
        "pointer": "/solver/nonlinear/box_constraints/bounds/*/*",
        "type": "float"
    },
    {
        "doc": "Box constraint values on optimization variables.",
        "pointer": "/solver/nonlinear/box_constraints/bounds/*",
        "type": "float"
    },
    {
        "default": -1,
        "doc": "Maximum change of optimization variables in one iteration, only for solvers with box constraints. Negative value to disable this constraint.",
        "pointer": "/solver/nonlinear/box_constraints/max_change",
        "type": "float"
    },
    {
        "doc": "Maximum change of optimization variables in one iteration, only for solvers with box constraints.",
        "pointer": "/solver/nonlinear/box_constraints/max_change",
        "type": "list"
    },
    {
        "doc": "Maximum change of every optimization variable in one iteration, only for solvers with box constraints.",
        "pointer": "/solver/nonlinear/box_constraints/max_change/*",
        "type": "float"
    },
    {
        "default": null,
        "doc": "Nonlinear solver advanced options",
        "optional": [
            "f_delta_tol",
            "f_delta_step_tol",
            "derivative_along_delta_x_tol",
            "apply_gradient_fd",
            "gradient_fd_eps"
        ],
        "pointer": "/solver/nonlinear/advanced",
        "type": "object"
    },
    {
        "default": 0,
        "doc": "Dangerous Option: Quit the optimization if the solver reduces the energy by less than f_delta for consecutive f_delta_step_tol steps.",
        "min": 0,
        "pointer": "/solver/nonlinear/advanced/f_delta_tol",
        "type": "float"
    },
    {
        "default": 100,
        "doc": "Dangerous Option: Quit the optimization if the solver reduces the energy by less than f_delta for consecutive f_delta_step_tol steps.",
        "pointer": "/solver/nonlinear/advanced/f_delta_step_tol",
        "type": "int"
    },
    {
        "default": 0,
        "doc": "Quit the optimization if the directional derivative along the descent direction is smaller than this tolerance.",
        "min": 0,
        "pointer": "/solver/nonlinear/advanced/derivative_along_delta_x_tol",
        "type": "float"
    },
    {
        "default": "None",
        "doc": "Expensive Option: For every iteration of the nonlinear solver, run finite difference to verify gradient of energy.",
        "options": [
            "None",
            "DirectionalDerivative",
            "FullFiniteDiff"
        ],
        "pointer": "/solver/nonlinear/advanced/apply_gradient_fd",
        "type": "string"
    },
    {
        "default": 1e-07,
        "doc": "Expensive Option: Eps for finite difference to verify gradient of energy.",
        "pointer": "/solver/nonlinear/advanced/gradient_fd_eps",
        "type": "float"
    },
    {
        "doc": "A list of mappings from the raw optimization variable to parameters in states.",
        "pointer": "/variable_to_simulation",
        "type": "list"
    },
    {
        "options": [
            "shape",
            "periodic-shape",
            "elastic",
            "friction",
            "damping",
            "macro-strain",
            "initial",
            "dirichlet",
            "dirichlet-nodes",
            "pressure"
        ],
        "pointer": "/variable_to_simulation/*/type",
        "type": "string"
    },
    {
        "optional": [
            "composite_map_type",
            "active_dimensions",
            "composite_map_indices"
        ],
        "pointer": "/variable_to_simulation/*",
        "required": [
            "state",
            "type",
            "composition"
        ],
        "type": "object",
        "type_name": "shape"
    },
    {
        "optional": [
            "composite_map_type",
            "composite_map_indices"
        ],
        "pointer": "/variable_to_simulation/*",
        "required": [
            "state",
            "type",
            "composition"
        ],
        "type": "object",
        "type_name": "periodic-shape"
    },
    {
        "optional": [
            "composite_map_type",
            "composite_map_indices"
        ],
        "pointer": "/variable_to_simulation/*",
        "required": [
            "state",
            "type",
            "composition"
        ],
        "type": "object",
        "type_name": "elastic"
    },
    {
        "optional": [
            "composite_map_type",
            "composite_map_indices"
        ],
        "pointer": "/variable_to_simulation/*",
        "required": [
            "state",
            "type",
            "composition"
        ],
        "type": "object",
        "type_name": "friction"
    },
    {
        "optional": [
            "composite_map_type",
            "composite_map_indices"
        ],
        "pointer": "/variable_to_simulation/*",
        "required": [
            "state",
            "type",
            "composition"
        ],
        "type": "object",
        "type_name": "damping"
    },
    {
        "optional": [
            "composite_map_type",
            "composite_map_indices"
        ],
        "pointer": "/variable_to_simulation/*",
        "required": [
            "state",
            "type",
            "composition"
        ],
        "type": "object",
        "type_name": "initial"
    },
    {
        "optional": [
            "composite_map_type",
            "composite_map_indices"
        ],
        "pointer": "/variable_to_simulation/*",
        "required": [
            "state",
            "type",
            "composition",
            "surface_selection"
        ],
        "type": "object",
        "type_name": "dirichlet"
    },
    {
        "optional": [
            "composite_map_type",
            "composite_map_indices"
        ],
        "pointer": "/variable_to_simulation/*",
        "required": [
            "state",
            "type",
            "composition"
        ],
        "type": "object",
        "type_name": "dirichlet-nodes"
    },
    {
        "optional": [
            "composite_map_type",
            "composite_map_indices"
        ],
        "pointer": "/variable_to_simulation/*",
        "required": [
            "state",
            "type",
            "composition",
            "surface_selection"
        ],
        "type": "object",
        "type_name": "pressure"
    },
    {
        "pointer": "/variable_to_simulation/*/mesh",
        "type": "string"
    },
    {
        "pointer": "/variable_to_simulation/*/mesh_id",
        "type": "int"
    },
    {
        "default": [],
        "pointer": "/variable_to_simulation/*/active_dimensions",
        "type": "list"
    },
    {
        "pointer": "/variable_to_simulation/*/state",
        "type": "int"
    },
    {
   )JSE_JSON";
        text += R"JSE_JSON(     "pointer": "/variable_to_simulation/*/state",
        "type": "list"
    },
    {
        "pointer": "/variable_to_simulation/*/state/*",
        "type": "int"
    },
    {
        "doc": "list of parametrizations that maps raw optimization variables to parameters in simulations",
        "pointer": "/variable_to_simulation/*/composition",
        "type": "list"
    },
    {
        "options": [
            "per-body-to-per-elem",
            "per-body-to-per-node",
            "E-nu-to-lambda-mu",
            "slice",
            "exp",
            "scale",
            "power",
            "append-values",
            "append-const",
            "linear-filter",
            "bounded-biharmonic-weights",
            "scalar-velocity-parametrization"
        ],
        "pointer": "/variable_to_simulation/*/composition/*/type",
        "type": "string"
    },
    {
        "doc": "TODO",
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "num_control_vertices",
            "num_vertices",
            "allow_rotations"
        ],
        "type": "object",
        "type_name": "bounded-biharmonic-weights"
    },
    {
        "doc": "TODO",
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "start_val",
            "dt"
        ],
        "type": "object",
        "type_name": "scalar-velocity-parametrization"
    },
    {
        "doc": "Append repeated constant at the end of the input vector",
        "optional": [
            "start"
        ],
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "value",
            "size",
            "type"
        ],
        "type": "object",
        "type_name": "append-const"
    },
    {
        "pointer": "/variable_to_simulation/*/composition/*/size",
        "type": "int"
    },
    {
        "pointer": "/variable_to_simulation/*/composition/*/value",
        "type": "float"
    },
    {
        "doc": "Append a list of constants at the end of the input vector",
        "optional": [
            "start"
        ],
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "values",
            "type"
        ],
        "type": "object",
        "type_name": "append-values"
    },
    {
        "default": -1,
        "pointer": "/variable_to_simulation/*/composition/*/start",
        "type": "int"
    },
    {
        "pointer": "/variable_to_simulation/*/composition/*/values",
        "type": "list"
    },
    {
        "pointer": "/variable_to_simulation/*/composition/*/values/*",
        "type": "float"
    },
    {
        "doc": "Ouputs x[from:to], where x is the input vector",
        "optional": [
            "from",
            "to",
            "parameter_index",
            "last"
        ],
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "slice"
    },
    {
        "default": -1,
        "pointer": "/variable_to_simulation/*/composition/*/from",
        "type": "int"
    },
    {
        "default": -1,
        "pointer": "/variable_to_simulation/*/composition/*/to",
        "type": "int"
    },
    {
        "default": -1,
        "pointer": "/variable_to_simulation/*/composition/*/last",
        "type": "int"
    },
    {
        "default": -1,
        "pointer": "/variable_to_simulation/*/composition/*/parameter_index",
        "type": "int"
    },
    {
        "doc": "Ouputs x ^ power, where x is the input vector",
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "power",
            "type"
        ],
        "type": "object",
        "type_name": "power"
    },
    {
        "pointer": "/variable_to_simulation/*/composition/*/power",
        "type": "float"
    },
    {
        "doc": "Ouputs exp(x), where x is the input vector",
        "optional": [
            "from",
            "to"
        ],
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "exp"
    },
    {
        "doc": "Ouputs x * value, where x is the input vector",
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "type",
            "value",
            "type"
        ],
        "type": "object",
        "type_name": "scale"
    },
    {
        "pointer": "/variable_to_simulation/*/composition/*/dimension",
        "type": "int"
    },
    {
        "default": null,
        "optional": [
            "translation",
            "rotation",
            "rotation_mode",
            "scale",
            "dimensions"
        ],
        "pointer": "/variable_to_simulation/*/composition/*/transformation",
        "type": "object"
    },
    {
        "default": "xyz",
        "doc": "Type of rotation, supported are any permutation of [xyz]+, axis_angle, quaternion, or rotation_vector.",
        "pointer": "/variable_to_simulation/*/composition/*/transformation/rotation_mode",
        "type": "string"
    },
    {
        "default": [],
        "doc": "Translate (two entries for 2D problems or three entries for 3D problems).",
        "pointer": "/variable_to_simulation/*/composition/*/transformation/translation",
        "type": "list"
    },
    {
        "default": [],
        "doc": "Rotate, in 2D, one number, the rotation angle, in 3D, three or four Euler angles, axis+angle, or a unit quaternion. Depends on rotation mode.",
        "pointer": "/variable_to_simulation/*/composition/*/transformation/rotation",
        "type": "list"
    },
    {
        "default": [],
        "doc": "Scale by specified factors along axes (two entries for 2D problems or three entries for 3D problems).",
        "pointer": "/variable_to_simulation/*/composition/*/transformation/scale",
        "type": "list"
    },
    {
        "default": 1,
        "doc": "Scale the object so that bounding box dimensions match specified dimensions, 2 entries for 2D problems, 3 entries for 3D problems.",
        "pointer": "/variable_to_simulation/*/composition/*/transformation/dimensions",
        "type": "float"
    },
    {
        "doc": "Scale the object so that bounding box dimensions match specified dimensions, 2 entries for 2D problems, 3 entries for 3D problems.",
        "pointer": "/variable_to_simulation/*/composition/*/transformation/dimensions",
        "type": "list"
    },
    {
        "default": 0,
        "pointer": "/variable_to_simulation/*/composition/*/transformation/dimensions/*",
        "type": "float"
    },
    {
        "default": 0,
        "pointer": "/variable_to_simulation/*/composition/*/transformation/translation/*",
        "type": "float"
    },
    {
        "default": 0,
        "pointer": "/variable_to_simulation/*/composition/*/transformation/rotation/*",
        "type": "float"
    },
    {
        "default": 0,
        "pointer": "/variable_to_simulation/*/composition/*/transformation/scale/*",
        "type": "float"
    },
    {
        "pointer": "/variable_to_simulation/*/composition/*/dimensions",
        "type": "list"
    },
    {
        "pointer": "/variable_to_simulation/*/composition/*/dimensions/*",
        "type": "int"
    },
    {
        "doc": "From per volume selection to per element.",
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "per-body-to-per-elem"
    },
    {
        "doc": "From per volume selection to per FE node.",
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "per-body-to-per-node"
    },
    {
        "pointer": "/variable_to_simulation/*/comp)JSE_JSON";
        text += R"JSE_JSON(osition/*",
        "required": [
            "is_volume",
            "type"
        ],
        "type": "object",
        "type_name": "E-nu-to-lambda-mu"
    },
    {
        "pointer": "/variable_to_simulation/*/composition/*/is_volume",
        "type": "bool"
    },
    {
        "doc": "Apply linear smoothing filter on a field per element.",
        "pointer": "/variable_to_simulation/*/composition/*",
        "required": [
            "state",
            "radius",
            "type"
        ],
        "type": "object",
        "type_name": "linear-filter"
    },
    {
        "pointer": "/variable_to_simulation/*/composition/*/state",
        "type": "int"
    },
    {
        "default": "none",
        "options": [
            "none",
            "interior",
            "boundary",
            "boundary_excluding_surface",
            "indices",
            "time_step_indexing"
        ],
        "pointer": "/variable_to_simulation/*/composite_map_type",
        "type": "string"
    },
    {
        "pointer": "/variable_to_simulation/*/composite_map_indices",
        "type": "file"
    },
    {
        "default": [],
        "pointer": "/variable_to_simulation/*/composite_map_indices",
        "type": "list"
    },
    {
        "pointer": "/variable_to_simulation/*/composite_map_indices/*",
        "type": "int"
    },
    {
        "doc": "TODO",
        "pointer": "/parameters",
        "type": "list"
    },
    {
        "doc": "TODO",
        "optional": [
            "number",
            "initial"
        ],
        "pointer": "/parameters/*",
        "type": "object"
    },
    {
        "pointer": "/parameters/*/number",
        "type": "int"
    },
    {
        "default": null,
        "doc": "TODO",
        "pointer": "/parameters/*/number",
        "required": [
            "surface_selection",
            "state"
        ],
        "type": "object"
    },
    {
        "doc": "TODO",
        "pointer": "/parameters/*/number",
        "required": [
            "volume_selection",
            "state",
            "exclude_boundary_nodes"
        ],
        "type": "object"
    },
    {
        "default": [],
        "pointer": "/parameters/*/initial",
        "type": "list"
    },
    {
        "pointer": "/parameters/*/initial/*",
        "type": "float"
    },
    {
        "pointer": "/parameters/*/initial",
        "type": "float"
    },
    {
        "pointer": "/parameters/*/state",
        "type": "int"
    },
    {
        "pointer": "/parameters/*/surface_selection",
        "type": "list"
    },
    {
        "pointer": "/parameters/*/surface_selection/*",
        "type": "int"
    },
    {
        "pointer": "/parameters/*/volume_selection",
        "type": "list"
    },
    {
        "pointer": "/parameters/*/volume_selection/*",
        "type": "int"
    },
    {
        "pointer": "/parameters/*/exclude_boundary_nodes",
        "type": "bool"
    },
    {
        "default": null,
        "doc": "Advanced settings for arranging forward simulations",
        "optional": [
            "solve_in_parallel",
            "solve_in_order",
            "characteristic_length",
            "enable_slim",
            "smooth_line_search"
        ],
        "pointer": "/solver/advanced",
        "type": "object"
    },
    {
        "default": 1,
        "doc": "A scaling on the nonlinear problem for better stability.",
        "pointer": "/solver/advanced/characteristic_length",
        "type": "float"
    },
    {
        "default": false,
        "doc": "Run forward simulations in parallel.",
        "pointer": "/solver/advanced/solve_in_parallel",
        "type": "bool"
    },
    {
        "default": [],
        "doc": "Run forward simulations in order.",
        "pointer": "/solver/advanced/solve_in_order",
        "type": "list"
    },
    {
        "doc": "Id of forward simulations.",
        "pointer": "/solver/advanced/solve_in_order/*",
        "type": "int"
    },
    {
        "default": false,
        "doc": "Evaluate the functionals and exit.",
        "pointer": "/compute_objective",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "Whether to apply slim smoothing to the optimization when step is accepted.",
        "pointer": "/solver/advanced/enable_slim",
        "type": "bool"
    },
    {
        "default": false,
        "doc": "Whether to apply slim smoothing to the optimization line search.",
        "pointer": "/solver/advanced/smooth_line_search",
        "type": "bool"
    }
]
)JSE_JSON";
        return nlohmann::json::parse(text);
    }();
    return value;
}

} // namespace polyfem_opt
} // namespace embed
} // namespace jse
