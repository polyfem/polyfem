#include "polyfem_material.hpp"

#include <string>

namespace jse
{
namespace embed
{
namespace polyfem_material
{

const nlohmann::json &spec()
{
    static const nlohmann::json value = []() {
        std::string text;
        text.reserve(50357);
        text += R"JSE_JSON(
[
    {
        "doc": "Type of material",
        "options": [
            "LinearElasticity",
            "HookeLinearElasticity",
            "SaintVenant",
            "NeoHookean",
            "MooneyRivlin",
            "MooneyRivlin3Param",
            "MooneyRivlin3ParamSymbolic",
            "UnconstrainedOgden",
            "IncompressibleOgden",
            "Stokes",
            "ActiveFiber",
            "HGOFiber",
            "IsochoricNeoHookean",
            "NavierStokes",
            "OperatorSplitting",
            "Electrostatics",
            "MaterialSum",
            "IncompressibleLinearElasticity",
            "Laplacian",
            "Helmholtz",
            "Bilaplacian",
            "AMIPS",
            "FixedCorotational",
            "VolumePenalty"
        ],
        "pointer": "/type",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Volume selection ID",
        "pointer": "/id",
        "type": "int"
    },
    {
        "doc": "Volume selection IDs",
        "pointer": "/id",
        "type": "list"
    },
    {
        "doc": "Volume selection ID",
        "pointer": "/id/*",
        "type": "int"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "NeoHookean"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "NeoHookean"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "IsochoricNeoHookean"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "IsochoricNeoHookean"
    },
    {
        "doc": "Material Parameters including ID, for Mooney-Rivlin",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "c1",
            "c2",
            "k"
        ],
        "type": "object",
        "type_name": "MooneyRivlin"
    },
    {
        "doc": "Material Parameters including ID, for Mooney-Rivlin",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "c1",
            "c2",
            "c3",
            "d1"
        ],
        "type": "object",
        "type_name": "MooneyRivlin3Param"
    },
    {
        "doc": "Material Parameters including ID, for Mooney-Rivlin",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "c1",
            "c2",
            "c3",
            "d1"
        ],
        "type": "object",
        "type_name": "MooneyRivlin3ParamSymbolic"
    },
    {
        "doc": "Material Parameters including ID, for [Ogden](https://en.wikipedia.org/wiki/Ogden_hyperelastic_model).",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "alphas",
            "mus",
            "Ds"
        ],
        "type": "object",
        "type_name": "UnconstrainedOgden"
    },
    {
        "doc": "Material Parameters including ID, for [Ogden](https://en.wikipedia.org/wiki/Ogden_hyperelastic_model).",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "c",
            "m",
            "k"
        ],
        "type": "object",
        "type_name": "IncompressibleOgden"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "LinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "LinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "fiber_direction"
        ],
        "pointer": "/",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "HookeLinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "fiber_direction"
        ],
        "pointer": "/",
        "required": [
            "type",
            "elasticity_tensor"
        ],
        "type": "object",
        "type_name": "HookeLinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "fiber_direction",
            "psi"
        ],
        "pointer": "/",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "SaintVenant"
    },
    {
        "doc": "Material Parameters including ID, E, nu, density ($\\rho$)",
        "optional": [
            "id",
            "rho",
            "phi",
            "fiber_direction",
            "psi"
        ],
        "pointer": "/",
        "required": [
            "type",
            "elasticity_tensor"
        ],
        "type": "object",
        "type_name": "SaintVenant"
    },
    {
        "doc": "Material Parameters including ID, viscosity, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "viscosity"
        ],
        "type": "object",
        "type_name": "Stokes"
    },
    {
        "doc": "Material Parameters including ID, viscosity, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "viscosity"
        ],
        "type": "object",
        "type_name": "NavierStokes"
    },
    {
        "doc": "Material Parameters including ID, viscosity, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "viscosity"
)JSE_JSON";
        text += R"JSE_JSON(        ],
        "type": "object",
        "type_name": "OperatorSplitting"
    },
    {
        "doc": "Material Parameters including ID, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "epsilon"
        ],
        "type": "object",
        "type_name": "Electrostatics"
    },
    {
        "doc": "Material Parameters including ID, Young's modulus ($E$), Poisson's ratio ($\\nu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "IncompressibleLinearElasticity"
    },
    {
        "doc": "Model that is a sum of other models",
        "optional": [
            "id",
            "models",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "MaterialSum"
    },
    {
        "doc": "Material Parameters including ID, Lam\u00e9 first ($\\lambda$), Lam\u00e9 second ($\\mu$), density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "IncompressibleLinearElasticity"
    },
    {
        "doc": "Material Parameters including ID, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Laplacian"
    },
    {
        "doc": "Material Parameters including ID, k, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "k"
        ],
        "type": "object",
        "type_name": "Helmholtz"
    },
    {
        "doc": "Material Parameters including ID, density ($\\rho$)",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "Bilaplacian"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "use_rest_pose",
            "weight",
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type"
        ],
        "type": "object",
        "type_name": "AMIPS"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/",
        "required": [
            "type",
            "E",
            "nu"
        ],
        "type": "object",
        "type_name": "FixedCorotational"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "id",
            "rho",
            "phi",
            "psi"
        ],
        "pointer": "/",
        "required": [
            "type",
            "lambda",
            "mu"
        ],
        "type": "object",
        "type_name": "FixedCorotational"
    },
    {
        "doc": "Material Parameters including ID",
        "optional": [
            "id",
            "rho"
        ],
        "pointer": "/",
        "required": [
            "type",
            "k"
        ],
        "type": "object",
        "type_name": "VolumePenalty"
    },
    {
        "doc": "Material Parameters",
        "optional": [
            "id",
            "rho",
            "fiber_direction"
        ],
        "pointer": "/",
        "required": [
            "type",
            "k1",
            "k2"
        ],
        "type": "object",
        "type_name": "HGOFiber"
    },
    {
        "doc": "Material Parameters",
        "optional": [
            "id",
            "rho",
            "Tmax",
            "fiber_direction"
        ],
        "pointer": "/",
        "required": [
            "type",
            "activation"
        ],
        "type": "object",
        "type_name": "ActiveFiber"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/E",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/E",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/E",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/E",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/E/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/E/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/E/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/E/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/E/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/nu",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/nu",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/nu",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/nu",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/nu/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/nu/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/nu/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/nu/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/nu/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/viscosity",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/viscosity",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/viscosity",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/viscosity",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/viscosity/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/viscosity/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/viscosity/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/viscosity/file_name",
        "type": "file"
    },
    {
        "doc": "Python function )JSE_JSON";
        text += R"JSE_JSON(name to evaluate the value",
        "pointer": "/viscosity/function_name",
        "type": "string"
    },
    {
        "doc": "Symmetric elasticity tensor",
        "pointer": "/elasticity_tensor",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/elasticity_tensor/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/elasticity_tensor/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/elasticity_tensor/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/elasticity_tensor/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/elasticity_tensor/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/elasticity_tensor/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/elasticity_tensor/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/elasticity_tensor/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/elasticity_tensor/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "Fiber direction",
        "pointer": "/fiber_direction",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/fiber_direction/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/fiber_direction/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/fiber_direction/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/fiber_direction/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/fiber_direction/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/fiber_direction/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/fiber_direction/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/fiber_direction/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/fiber_direction/*/function_name",
        "type": "string"
    },
    {
        "default": [],
        "doc": "Fiber direction row",
        "pointer": "/fiber_direction/*",
        "type": "list"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/fiber_direction/*/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/fiber_direction/*/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/fiber_direction/*/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/fiber_direction/*/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/fiber_direction/*/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/fiber_direction/*/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/fiber_direction/*/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/fiber_direction/*/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/fiber_direction/*/*/function_name",
        "type": "string"
    },
    {
        "default": 1,
        "doc": "Value as a constant float",
        "pointer": "/rho",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/rho",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/rho",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/rho",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/rho/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/rho/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/rho/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/rho/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/rho/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/epsilon",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/epsilon",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/epsilon",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/epsilon",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/epsilon/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/epsilon/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/epsilon/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/epsilon/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/epsilon/function_name",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/phi",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/phi",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/phi",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/phi",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/phi/unit",
        "type": )JSE_JSON";
        text += R"JSE_JSON("string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/phi/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/phi/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/phi/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/phi/function_name",
        "type": "string"
    },
    {
        "default": 0,
        "doc": "Value as a constant float",
        "pointer": "/psi",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/psi",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/psi",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/psi",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/psi/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/psi/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/psi/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/psi/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/psi/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/k",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/k",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/k",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/k",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/k/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/k/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/k/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/k/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/k/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/mu",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/mu",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/mu",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/mu",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/mu/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/mu/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/mu/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/mu/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/mu/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/lambda",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/lambda",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/lambda",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/lambda",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/lambda/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/lambda/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/lambda/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/lambda/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/lambda/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/c1",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/c1",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/c1",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/c1",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/c1/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/c1/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/c1/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/c1/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/c1/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/c2",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/c2",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/c2",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/c2",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/c2/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/c2/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/c2/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/c2/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/c2/functio)JSE_JSON";
        text += R"JSE_JSON(n_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/c3",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/c3",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/c3",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/c3",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/c3/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/c3/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/c3/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/c3/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/c3/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/d1",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/d1",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/d1",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/d1",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/d1/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/d1/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/d1/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/d1/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/d1/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/k1",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/k1",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/k1",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/k1",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/k1/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/k1/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/k1/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/k1/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/k1/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/k2",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/k2",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/k2",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/k2",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/k2/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/k2/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/k2/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/k2/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/k2/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/alphas",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/alphas",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/alphas",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/alphas",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/alphas/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/alphas/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/alphas/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/alphas/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/alphas/function_name",
        "type": "string"
    },
    {
        "doc": "Ogden mu",
        "pointer": "/mus",
        "type": "list"
    },
    {
        "doc": "Ogden D",
        "pointer": "/Ds",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/alphas/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/alphas/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/alphas/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/alphas/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/alphas/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/alphas/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/alphas/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/alphas/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/alphas/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/mus/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/mus/*",
        "type": "string"
    },
    {
        "d)JSE_JSON";
        text += R"JSE_JSON(oc": "Value with unit",
        "pointer": "/mus/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/mus/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/mus/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/mus/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/mus/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/mus/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/mus/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/Ds/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/Ds/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/Ds/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/Ds/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/Ds/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/Ds/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/Ds/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/Ds/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/Ds/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/c",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/c",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/c",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/c",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/c/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/c/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/c/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/c/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/c/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/m",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/m",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/m",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/m",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/m/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/m/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/m/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/m/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/m/function_name",
        "type": "string"
    },
    {
        "doc": "Coefficient(s) of Incompressible Ogden",
        "pointer": "/c",
        "type": "list"
    },
    {
        "doc": "Exponent(s) of Incompressible Ogden",
        "pointer": "/m",
        "type": "list"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/c/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/c/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/c/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/c/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/c/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/c/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/c/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/c/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/c/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/m/*",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/m/*",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/m/*",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/m/*",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/m/*/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/m/*/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/m/*/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/m/*/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/m/*/function_name",
        "type": "string"
    },
    {
        "doc": "Value as a constant float",
        "pointer": "/activation",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/activation",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/activation",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/activation",
        "required": [
        )JSE_JSON";
        text += R"JSE_JSON(    "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/activation/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/activation/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/activation/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/activation/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/activation/function_name",
        "type": "string"
    },
    {
        "default": 1,
        "doc": "Value as a constant float",
        "pointer": "/Tmax",
        "type": "float"
    },
    {
        "doc": "Value as an expression of $x,y,z,t$ or a file",
        "pointer": "/Tmax",
        "type": "string"
    },
    {
        "doc": "Value with unit",
        "pointer": "/Tmax",
        "required": [
            "value",
            "unit"
        ],
        "type": "object"
    },
    {
        "doc": "Value from python function",
        "pointer": "/Tmax",
        "required": [
            "file_name",
            "function_name"
        ],
        "type": "object"
    },
    {
        "doc": "The unit of the Value",
        "pointer": "/Tmax/unit",
        "type": "string"
    },
    {
        "doc": "The value of the constant",
        "pointer": "/Tmax/value",
        "type": "float"
    },
    {
        "doc": "The value as an expression or a file",
        "pointer": "/Tmax/value",
        "type": "string"
    },
    {
        "doc": "Python filename containing the function to evaluate the value",
        "pointer": "/Tmax/file_name",
        "type": "file"
    },
    {
        "doc": "Python function name to evaluate the value",
        "pointer": "/Tmax/function_name",
        "type": "string"
    },
    {
        "default": false,
        "doc": "Use amips wrt to rest pose or the regular element",
        "pointer": "/use_rest_pose",
        "type": "bool"
    },
    {
        "default": 1.0,
        "doc": "Scale factor for the AMIPS energy.",
        "pointer": "/weight",
        "type": "float"
    }
]
)JSE_JSON";
        return nlohmann::json::parse(text);
    }();
    return value;
}

} // namespace polyfem_material
} // namespace embed
} // namespace jse
