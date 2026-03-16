#pragma once

#include "easy_polyfem/config.hpp"
#include <filesystem>

namespace easy_polyfem
{
    int run_polyfem(const Options &opt, const std::filesystem::path &json_path);
}