#pragma once

#include <polyfem/Mesh.hpp>
#include <polyfem/Common.hpp>

namespace polyfem
{
    class BoxSetter
    {
    public:
        static void set_sidesets(const json &args, Mesh &mesh);
    };
} // namespace polyfem
