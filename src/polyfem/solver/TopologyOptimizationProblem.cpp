#include "TopologyOptimizationProblem.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <filesystem>

namespace polyfem
{
    TopologyOptimizationProblem::TopologyOptimizationProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_) : OptimizationProblem(state_, j_)
    {
        optimization_name = "topology";
        state.args["output"]["paraview"]["options"]["material"] = true;

        if (opt_params.contains("min_density"))
            min_density = opt_params["min_density"];
        if (opt_params.contains("max_density"))
            max_density = opt_params["max_density"];
    }

    
}