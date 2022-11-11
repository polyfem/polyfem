#include "Parameter.hpp"
#include "InitialConditionParameter.hpp"
#include "ElasticParameter.hpp"
#include "DampingParameter.hpp"
#include "FrictionParameter.hpp"
#include "ControlParameter.hpp"
#include "ShapeParameter.hpp"
#include "TopologyOptimizationParameter.hpp"

namespace polyfem
{
    std::shared_ptr<Parameter> Parameter::create(const json &args, std::vector<std::shared_ptr<State>> &states_ptr)
    {
        const std::string type = args["type"];
        if (type == "initial")
            return std::make_shared<InitialConditionParameter>(states_ptr, args);
        else if (type == "material")
            return std::make_shared<ElasticParameter>(states_ptr, args);
        else if (type == "damping")
            return std::make_shared<DampingParameter>(states_ptr, args);
        else if (type == "friction")
            return std::make_shared<FrictionParameter>(states_ptr, args);
        else if (type == "control")
        {
            assert(false);
            // implement class first
            // return std::make_shared<ControlParameter>(states_ptr, args);
        }
        else if (type == "shape")
            return std::make_shared<ShapeParameter>(states_ptr, args);
        else if (type == "topology")
            return std::make_shared<TopologyOptimizationParameter>(states_ptr, args);
        
        log_and_throw_error("Unknown type of parameter!");
        return std::make_shared<ShapeParameter>(states_ptr, args);
    }
}