#pragma once

#include "easy_polyfem/config.hpp"
#include <string>

namespace easy_polyfem
{
    bool validate_options(const Options &opt, std::string &error_message);
}