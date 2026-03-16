#include "easy_polyfem/runner.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>

namespace easy_polyfem
{
    int run_polyfem(const Options &opt, const std::filesystem::path &json_path)
    {
        std::ostringstream cmd;
        cmd << "\"" << opt.polyfem_bin << "\""
            << " --json "
            << "\"" << json_path.string() << "\"";

        std::cout << "Running: " << cmd.str() << "\n";
        return std::system(cmd.str().c_str());
    }
}