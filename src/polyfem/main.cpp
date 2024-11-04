#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <CLI/CLI.hpp>
#include <h5pp/h5pp.h>
#include <polyfem/State.hpp>
#include <polyfem/OptState.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/io/YamlToJson.hpp>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>

using json = nlohmann::json;

using namespace polyfem;
using namespace solver;

/**
 * @brief Checks if a specific argument is present in the command line.
 *
 * @param command_line The CLI::App object containing parsed command-line arguments.
 * @param value The name of the argument to check (e.g., "log_level").
 * @return true If the argument is present.
 * @return false If the argument is not present.
 */
bool has_arg(const CLI::App &command_line, const std::string &value)
{
    const auto *opt = command_line.get_option_no_throw(
        value.size() == 1 ? ("-" + value) : ("--" + value));
    if (!opt)
        return false;

    return opt->count() > 0;
}

/**
 * @brief Loads input parameters from a JSON file.
 *
 * @param json_file Path to the JSON file.
 * @param out JSON object to store the loaded parameters.
 * @return true If loading is successful.
 * @return false If the file cannot be opened or parsed.
 */
bool load_json(const std::string &json_file, json &out)
{
    std::ifstream file(json_file);

    if (!file.is_open())
        return false;

    try
    {
        file >> out;
    }
    catch (const std::exception &e)
    {
        spdlog::error("Error parsing JSON file {}: {}", json_file, e.what());
        return false;
    }

    if (!out.contains("root_path"))
        out["root_path"] = json_file;

    return true;
}

/**
 * @brief Loads input parameters from a YAML file.
 *
 * @param yaml_file Path to the YAML file.
 * @param out JSON object to store the converted parameters.
 * @return true If loading and conversion are successful.
 * @return false If the file cannot be opened or parsed.
 */
bool load_yaml(const std::string &yaml_file, json &out)
{
    try
    {
        out = io::yaml_file_to_json(yaml_file);
        if (!out.contains("root_path"))
            out["root_path"] = yaml_file;
    }
    catch (const std::exception &e)
    {
        spdlog::error("Error parsing YAML file {}: {}", yaml_file, e.what());
        return false;
    }
    return true;
}

/**
 * @brief Executes a forward simulation based on the provided parameters.
 *
 * @param command_line Parsed command-line arguments.
 * @param hdf5_file Path to the HDF5 input file (if any).
 * @param output_dir Directory where output files will be stored.
 * @param max_threads Maximum number of threads to use.
 * @param is_strict Flag to enable or disable strict validation.
 * @param fallback_solver Flag to enable solver fallback.
 * @param log_level Logging level for simulation output.
 * @param in_args JSON object containing input parameters.
 * @return int Exit status code.
 */
int forward_simulation(const CLI::App &command_line,
                       const std::string &hdf5_file,
                       const std::string output_dir,
                       const unsigned max_threads,
                       const bool is_strict,
                       const bool fallback_solver,
                       const spdlog::level::level_enum &log_level,
                       json &in_args);

/**
 * @brief Executes an optimization simulation based on the provided parameters.
 *
 * @param command_line Parsed command-line arguments.
 * @param max_threads Maximum number of threads to use.
 * @param is_strict Flag to enable or disable strict validation.
 * @param log_level Logging level for simulation output.
 * @param opt_args JSON object containing optimization parameters.
 * @return int Exit status code.
 */
int optimization_simulation(const CLI::App &command_line,
                            const unsigned max_threads,
                            const bool is_strict,
                            const spdlog::level::level_enum &log_level,
                            json &opt_args);

int main(int argc, char **argv)
{
    using namespace polyfem;

    CLI::App command_line{"PolyFEM - A simulation tool for FEM."};

    command_line.ignore_case();
    command_line.ignore_underscore();

    unsigned max_threads = std::numeric_limits<unsigned>::max();
    command_line.add_option("--max_threads", max_threads, "Maximum number of threads to use for computation.");

    auto input = command_line.add_option_group("Input Files", "Specify the input files for the simulation.");

    std::string json_file;
    input->add_option("-j,--json", json_file, "Path to the JSON input file.")
        ->check(CLI::ExistingFile)
        ->description("This file contains simulation parameters in JSON format.");

    std::string yaml_file;
    input->add_option("-y,--yaml", yaml_file, "Path to the YAML input file.")
        ->check(CLI::ExistingFile)
        ->description("This file contains simulation parameters in YAML format.");

    std::string hdf5_file;
    input->add_option("--hdf5", hdf5_file, "Path to the HDF5 input file.")
        ->check(CLI::ExistingFile)
        ->description("This file contains mesh and simulation parameters.");

    input->require_option(1);

    std::string output_dir;
    command_line.add_option("-o,--output_dir", output_dir, "Directory where the output files will be stored.")
        ->check(CLI::ExistingDirectory | CLI::NonexistentPath);

    bool is_strict = true;
    command_line.add_flag("-s,--strict_validation,!--ns,!--no_strict_validation", is_strict,
                          "Enable or disable strict validation of input files.");

    bool fallback_solver = false;
    command_line.add_flag("--enable_overwrite_solver", fallback_solver,
                          "Enable solver fallback if specified solver is not available.");

    const std::vector<std::pair<std::string, spdlog::level::level_enum>> SPDLOG_LEVEL_NAMES_TO_LEVELS = {
        {"trace", spdlog::level::trace},
        {"debug", spdlog::level::debug},
        {"info", spdlog::level::info},
        {"warning", spdlog::level::warn},
        {"error", spdlog::level::err},
        {"critical", spdlog::level::critical},
        {"off", spdlog::level::off}
    };

    spdlog::level::level_enum log_level = spdlog::level::debug;
    command_line.add_option("--log_level", log_level, "Set the logging level for the simulation output.")
        ->transform(CLI::CheckedTransformer(SPDLOG_LEVEL_NAMES_TO_LEVELS, CLI::ignore_case));

    CLI11_PARSE(command_line, argc, argv);

    json in_args = json({});

    if (!json_file.empty() || !yaml_file.empty())
    {
        bool ok = false;
        if (!json_file.empty())
        {
            ok = load_json(json_file, in_args);
        }
        else if (!yaml_file.empty())
        {
            ok = load_yaml(yaml_file, in_args);
        }

        if (!ok)
        {
            log_and_throw_error(fmt::format("Unable to open or parse input file: {}", json_file.empty() ? yaml_file : json_file));
        }

        if (in_args.contains("states"))
        {
            return optimization_simulation(command_line, max_threads, is_strict, log_level, in_args);
        }
        else
        {
            return forward_simulation(command_line, "", output_dir, max_threads, is_strict, fallback_solver, log_level, in_args);
        }
    }
    else
    {
        return forward_simulation(command_line, hdf5_file, output_dir, max_threads, is_strict, fallback_solver, log_level, in_args);
    }
}

int forward_simulation(const CLI::App &command_line,
                       const std::string &hdf5_file,
                       const std::string output_dir,
                       const unsigned max_threads,
                       const bool is_strict,
                       const bool fallback_solver,
                       const spdlog::level::level_enum &log_level,
                       json &in_args)
{
    std::vector<std::string> names;
    std::vector<Eigen::MatrixXi> cells;
    std::vector<Eigen::MatrixXd> vertices;

    if (in_args.empty() && hdf5_file.empty())
    {
        logger().error("No input file specified!");
        return command_line.exit(CLI::RequiredError("--json or --hdf5"));
    }

    if (in_args.empty() && !hdf5_file.empty())
    {
        using MatrixXl = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;

        try
        {
            h5pp::File file(hdf5_file, h5pp::FileAccess::READONLY);
            std::string json_string = file.readDataset<std::string>("json");

            in_args = json::parse(json_string);
            in_args["root_path"] = hdf5_file;

            names = file.findGroups("", "/meshes");
            cells.resize(names.size());
            vertices.resize(names.size());

            for (size_t i = 0; i < names.size(); ++i)
            {
                const std::string &name = names[i];
                cells[i] = file.readDataset<MatrixXl>("/meshes/" + name + "/c").cast<int>();
                vertices[i] = file.readDataset<Eigen::MatrixXd>("/meshes/" + name + "/v");
            }
        }
        catch (const std::exception &e)
        {
            logger().error("Error reading HDF5 file {}: {}", hdf5_file, e.what());
            return EXIT_FAILURE;
        }
    }

    json tmp = json::object();
    if (has_arg(command_line, "log_level"))
        tmp["/output/log/level"_json_pointer] = static_cast<int>(log_level);
    if (has_arg(command_line, "max_threads"))
        tmp["/solver/max_threads"_json_pointer] = max_threads;
    if (has_arg(command_line, "output_dir"))
        tmp["/output/directory"_json_pointer] = std::filesystem::absolute(output_dir);
    if (has_arg(command_line, "enable_overwrite_solver"))
        tmp["/solver/linear/enable_overwrite_solver"_json_pointer] = fallback_solver;

    assert(tmp.is_object());
    in_args.merge_patch(tmp);

    State state;
    state.init(in_args, is_strict);

    state.load_mesh(/*non_conforming=*/false, names, cells, vertices);

    if (state.mesh == nullptr)
    {
        logger().error("Mesh was not loaded successfully.");
        return EXIT_FAILURE;
    }

    state.stats.compute_mesh_stats(*state.mesh);

    state.build_basis();

    state.assemble_rhs();
    state.assemble_mass_mat();

    Eigen::MatrixXd sol, pressure;
    state.solve_problem(sol, pressure);

    state.compute_errors(sol);

    logger().info("Total simulation time: {}s", state.timings.total_time());

    state.save_json(sol);
    state.export_data(sol, pressure);

    return EXIT_SUCCESS;
}

int optimization_simulation(const CLI::App &command_line,
                            const unsigned max_threads,
                            const bool is_strict,
                            const spdlog::level::level_enum &log_level,
                            json &opt_args)
{
    json tmp = json::object();
    if (has_arg(command_line, "log_level"))
        tmp["/output/log/level"_json_pointer] = static_cast<int>(log_level);
    if (has_arg(command_line, "max_threads"))
        tmp["/solver/max_threads"_json_pointer] = max_threads;
    opt_args.merge_patch(tmp);

    OptState opt_state;
    opt_state.init(opt_args, is_strict);

    opt_state.create_states(
        opt_state.args["compute_objective"].get<bool>() ?
            polyfem::solver::CacheLevel::Solution :
            polyfem::solver::CacheLevel::Derivatives,
        opt_state.args["solver"]["max_threads"].get<int>());

    opt_state.init_variables();
    opt_state.create_problem();

    Eigen::VectorXd x;
    opt_state.initial_guess(x);

    if (opt_state.args["compute_objective"].get<bool>())
    {
        logger().info("Objective value: {}", opt_state.eval(x));
        return EXIT_SUCCESS;
    }

    opt_state.solve(x);
    return EXIT_SUCCESS;
}
