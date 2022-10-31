#include "Objective.hpp"

namespace polyfem::solver
{
    Objective::Objective(const State &state, const json &obj_args): state_(state)
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
}