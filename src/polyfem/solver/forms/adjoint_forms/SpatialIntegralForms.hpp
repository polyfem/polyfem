#pragma once

#include "AdjointForm.hpp"

namespace polyfem::solver
{
    class SpatialIntegralForm : public AdjointForm
    {
    public:
        SpatialIntegralObjective(const State &state, const json &args);
    };
}