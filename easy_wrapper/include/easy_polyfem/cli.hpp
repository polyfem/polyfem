#pragma once

#include "easy_polyfem/config.hpp"

namespace easy_polyfem
{
    void print_usage();
    bool parse_args(int argc, char **argv, Options &opt, std::string &error_message);
}