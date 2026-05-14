#include "polyfem_objective.hpp"

#include <string>

namespace jse
{
namespace embed
{
namespace polyfem_objective
{

const nlohmann::json &spec()
{
    static const nlohmann::json value = []() {
        std::string text;
        text.reserve(21833);
        text += R"JSE_JSON(
[
    {
        "doc": "TODO",
        "options": [
            "transient_integral",
            "power",
            "divide",
            "plus-const",
            "target",
            "sdf-target",
            "mesh-target",
            "center-target",
            "function-target",
            "displacement-target",
            "node-target",
            "position",
            "acceleration",
            "kinetic",
            "disp_grad",
            "homo_disp_grad",
            "stress",
            "stress_norm",
            "dirichlet_energy",
            "elastic_energy",
            "quadratic_contact_force_norm",
            "log_contact_force_norm",
            "max_stress",
            "compliance",
            "weighted_solution",
            "strain_norm",
            "boundary_smoothing",
            "collision_barrier",
            "deformed_collision_barrier",
            "control_smoothing",
            "material_smoothing",
            "volume",
            "soft_constraint",
            "layer_thickness",
            "layer_thickness_log",
            "log",
            "AMIPS",
            "parametrized_product",
            "smooth_contact_force_norm",
            "min_jacobian",
            "min-dist-target"
        ],
        "pointer": "/type",
        "type": "string"
    },
    {
        "default": "",
        "doc": "TODO",
        "pointer": "/print_energy",
        "type": "string"
    },
    {
        "default": 2,
        "doc": "TODO",
        "pointer": "/power",
        "type": "float"
    },
    {
        "default": true,
        "doc": "TODO",
        "pointer": "/scale_invariant",
        "type": "bool"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "type",
            "objective"
        ],
        "type": "object",
        "type_name": "divide"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "type",
            "objective",
            "value"
        ],
        "type": "object",
        "type_name": "plus-const"
    },
    {
        "doc": "TODO",
        "optional": [
            "power",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "type",
            "objective"
        ],
        "type": "object",
        "type_name": "power"
    },
    {
        "doc": "TODO",
        "optional": [
            "control_points",
            "control_points_grid",
            "knots",
            "knots_u",
            "knots_v",
            "weight",
            "print_energy",
            "surface_selection"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type",
            "delta"
        ],
        "type": "object",
        "type_name": "sdf-target"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "surface_selection"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type",
            "delta",
            "mesh_path"
        ],
        "type": "object",
        "type_name": "mesh-target"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "tolerance"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type",
            "target_data_path"
        ],
        "type": "object",
        "type_name": "node-target"
    },
    {
        "pointer": "/target_data_path",
        "type": "file"
    },
    {
        "default": 1e-07,
        "pointer": "/tolerance",
        "type": "float"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "target_function",
            "target_function_gradient",
            "surface_selection"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "function-target"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "surface_selection",
            "reference_cached_body_ids",
            "target_state"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "target"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "target_displacement",
            "active_dimension",
            "surface_selection"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "displacement-target"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "volume_selection",
            "target_state"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "center-target"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "volume_selection",
            "target",
            "steps"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "min-dist-target"
    },
    {
        "doc": "TODO",
        "pointer": "/value",
        "type": "float"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/volume_selection",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/volume_selection",
        "type": "file"
    },
    {
        "doc": "TODO",
        "pointer": "/volume_selection/*",
        "type": "int"
    },
    {
        "doc": "TODO",
        "pointer": "/surface_selection",
        "type": "int"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/surface_selection",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/surface_selection/*",
        "type": "int"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/reference_cached_body_ids",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/reference_cached_body_ids/*",
        "type": "int"
    },
    {
        "default": -1,
        "doc": "TODO",
        "pointer": "/target_state",
        "type": "int"
    },
    {
        "default": 1,
        "doc": "TODO",
        "pointer": "/weight",
        "type": "float"
    },
    {
        "default": "uniform",
        "doc": "TODO",
        "options": [
            "simpson",
            "uniform",
            "final",
            "steps"
        ],
        "pointer": "/integral_type",
        "type": "string"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/steps",
        "type": "list"
    },
    {
        "default": "0",
        "doc": "TODO",
        "pointer": "/target_function",
        "type": "string"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/target_function_gradient",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/target_function_gradient/*",
        "type": "string"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/target_displacement",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/target_displacement/*",
        "type": "float"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/target",
        "type": "list"
    },
    {
        "doc": "TOD)JSE_JSON";
        text += R"JSE_JSON(O",
        "pointer": "/target/*",
        "type": "float"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/active_dimension",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/active_dimension/*",
        "type": "bool"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/control_points",
        "type": "list"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/control_points/*",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/control_points/*/*",
        "type": "float"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/knots",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/knots/*",
        "type": "float"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/control_points_grid",
        "type": "list"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/control_points_grid/*",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/control_points_grid/*/*",
        "type": "float"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/knots_u",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/knots_u/*",
        "type": "float"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/knots_v",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/knots_v/*",
        "type": "float"
    },
    {
        "doc": "TODO",
        "pointer": "delta",
        "type": "int"
    },
    {
        "doc": "TODO",
        "pointer": "mesh_path",
        "type": "string"
    },
    {
        "doc": "TODO",
        "pointer": "/state",
        "type": "int"
    },
    {
        "doc": "TODO",
        "options": [
            "exact",
            "marker-data",
            "exact-marker"
        ],
        "pointer": "/matching",
        "type": "string"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "volume_selection"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type",
            "dim"
        ],
        "type": "object",
        "type_name": "position"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "power"
        ],
        "pointer": "/",
        "required": [
            "type",
            "parametrization"
        ],
        "type": "object",
        "type_name": "parametrized_product"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "power"
        ],
        "pointer": "/",
        "required": [
            "type",
            "objective",
            "soft_bound"
        ],
        "type": "object",
        "type_name": "soft_constraint"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "type",
            "objective"
        ],
        "type": "object",
        "type_name": "log"
    },
    {
        "doc": "TODO",
        "pointer": "/objective",
        "type": "object"
    },
    {
        "doc": "TODO",
        "pointer": "/objective",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/objective/*",
        "type": "object"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "integral_type",
            "steps"
        ],
        "pointer": "/",
        "required": [
            "type",
            "static_objective",
            "state"
        ],
        "type": "object",
        "type_name": "transient_integral"
    },
    {
        "doc": "TODO",
        "pointer": "/static_objective",
        "type": "object"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "volume_selection"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type",
            "dim"
        ],
        "type": "object",
        "type_name": "acceleration"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "volume_selection"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "kinetic"
    },
    {
        "doc": "One entry of displacement gradient matrix",
        "optional": [
            "volume_selection",
            "weight",
            "print_energy",
            "dimensions"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "disp_grad"
    },
    {
        "doc": "One entry of macro displacement gradient matrix, only work for homogenization",
        "optional": [
            "weight",
            "print_energy",
            "dimensions"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "homo_disp_grad"
    },
    {
        "doc": "One entry of elastic stress matrix",
        "optional": [
            "volume_selection",
            "weight",
            "print_energy",
            "dimensions"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "stress"
    },
    {
        "default": [],
        "pointer": "/dimensions",
        "type": "list"
    },
    {
        "doc": "Elastic energy over the volume selection",
        "optional": [
            "volume_selection",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "elastic_energy"
    },
    {
        "doc": "Lp Norm of elastic stress over the volume selection",
        "optional": [
            "volume_selection",
            "power",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "stress_norm"
    },
    {
        "doc": "Dirichlet energy for Poisson problem",
        "optional": [
            "volume_selection",
            "power",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "dirichlet_energy"
    },
    {
        "doc": "Lp Norm of elastic stress over the volume selection",
        "optional": [
            "surface_selection",
            "dhat",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "quadratic_contact_force_norm"
    },
    {
        "doc": "Lp Norm of elastic stress over the volume selection",
        "optional": [
            "surface_selection",
            "dhat",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "log_contact_force_norm"
    },
    {
        "pointer": "/force_matching_function",
        "type": "list"
    },
    {
        "pointer": "/force_matching_function/*",
        "type": "string"
    },
    {
        "doc": "Pointwise max stress over the volume selecti)JSE_JSON";
        text += R"JSE_JSON(on",
        "optional": [
            "volume_selection",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "max_stress"
    },
    {
        "doc": "TODO",
        "optional": [
            "volume_selection",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "compliance"
    },
    {
        "doc": "TODO",
        "optional": [
            "volume_selection",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "strain_norm"
    },
    {
        "doc": "TODO",
        "pointer": "/bounds",
        "type": "list"
    },
    {
        "doc": "Barrier function parameters for material bounds.",
        "pointer": "/bounds/*",
        "required": [
            "min",
            "max",
            "type",
            "dhat",
            "kappa"
        ],
        "type": "object"
    },
    {
        "options": [
            "E",
            "nu",
            "lambda",
            "mu"
        ],
        "pointer": "/bounds/*/type",
        "type": "string"
    },
    {
        "doc": "TODO",
        "optional": [
            "scale_invariant",
            "power",
            "weight",
            "print_energy",
            "surface_selection",
            "dimensions"
        ],
        "pointer": "/",
        "required": [
            "type",
            "state"
        ],
        "type": "object",
        "type_name": "boundary_smoothing"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "volume_selection"
        ],
        "pointer": "/",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "material_smoothing"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "control_smoothing"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "dhat"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "collision_barrier"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "dhat"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "deformed_collision_barrier"
    },
    {
        "doc": "TODO",
        "optional": [
            "boundary_ids",
            "dhat",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "type",
            "state"
        ],
        "type": "object",
        "type_name": "layer_thickness"
    },
    {
        "doc": "TODO",
        "optional": [
            "boundary_ids",
            "dhat",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "type",
            "state",
            "dmin"
        ],
        "type": "object",
        "type_name": "layer_thickness_log"
    },
    {
        "default": 0.001,
        "doc": "The support size of barrier function",
        "pointer": "/dhat",
        "type": "float"
    },
    {
        "doc": "The min distance of barrier function",
        "pointer": "/dmin",
        "type": "float"
    },
    {
        "default": [],
        "doc": "TODO",
        "pointer": "/boundary_ids",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/boundary_ids/*",
        "type": "int"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "type",
            "state"
        ],
        "type": "object",
        "type_name": "weighted_solution"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy",
            "volume_selection"
        ],
        "pointer": "/",
        "required": [
            "type",
            "state"
        ],
        "type": "object",
        "type_name": "volume"
    },
    {
        "doc": "TODO",
        "pointer": "/soft_bound",
        "type": "list"
    },
    {
        "doc": "TODO",
        "pointer": "/soft_bound/*",
        "type": "float"
    },
    {
        "doc": "TODO",
        "optional": [
            "volume_selection",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "type",
            "state"
        ],
        "type": "object",
        "type_name": "AMIPS"
    },
    {
        "optional": [
            "surface_selection",
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "state",
            "type"
        ],
        "type": "object",
        "type_name": "smooth_contact_force_norm"
    },
    {
        "doc": "TODO",
        "optional": [
            "weight",
            "print_energy"
        ],
        "pointer": "/",
        "required": [
            "type",
            "state"
        ],
        "type": "object",
        "type_name": "min_jacobian"
    }
]
)JSE_JSON";
        return nlohmann::json::parse(text);
    }();
    return value;
}

} // namespace polyfem_objective
} // namespace embed
} // namespace jse
