#pragma once

#include <polyfem/State.hpp>

#include <string>

namespace polyfem
{
    class FEBioReader
    {
    public:
        static void load(const std::string &path, State &state, const std::string &export_solution = "");
    };
} // namespace polyfem
