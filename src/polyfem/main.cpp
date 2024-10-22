#include <filesystem>
#include <CLI/CLI.hpp>
#include <h5pp/h5pp.h>
#include <polyfem/State.hpp>
#include <polyfem/OptState.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/io/YamlToJson.hpp>

using namespace polyfem;
using namespace solver;

// Helper function to check if an argument is passed to the CLI
bool has_arg(const CLI::App &command_line, const std::string &value) {
    const auto *opt = command_line.get_option_no_throw(value.size() == 1 ? ("-" + value) : ("--" + value));
    if (!opt)
        return false;

    return opt->count() > 0;
}

// Load input from a JSON file
bool load_json(const std::string &json_file, json &out) {
    std::ifstream file(json_file);
    if (!file.is_open())
        return false;

    file >> out;
    if (!out.contains("root_path"))
        out["root_path"] = json_file;

    return true;
}

// Load input from a YAML file
bool load_yaml(const std::string &yaml_file, json &out) {
    try {
        out = io::yaml_file_to_json(yaml_file);
        if (!out.contains("root_path"))
            out["root_path"] = yaml_file;
    } catch (...) {
        return false;
    }
    return true;
}

// Function for forward simulation
int forward_simulation(const CLI::App &command_line, const std::string &hdf5_file, const std::string output_dir, 
                       const unsigned max_threads, const bool is_strict, const bool fallback_solver,
                       const spdlog::level::level_enum &log_level, json &in_args);

// Function for optimization simulation
int optimization_simulation(const CLI::App &command_line, const unsigned max_threads, const bool is_strict, 
                            const spdlog::level::level_enum &log_level, json &opt_args);

int main(int argc, char **argv) {
    using namespace polyfem;

    CLI::App command_line{"PolyFEM - A simulation tool for FEM."};

    // Set up command-line options and flags
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

    input->require_option(1);  // At least one input file is required

    std::string output_dir;
    command_line.add_option("-o,--output_dir", output_dir, "Directory where the output files will be stored.")
                ->check(CLI::ExistingDirectory | CLI::NonexistentPath);

    bool is_strict = true;
    command_line.add_flag("-s,--strict_validation,!--ns,!--no_strict_validation", is_strict, 
                          "Enable or disable strict validation of input files.");

    bool fallback_solver = false;
    command_line.add_flag("--enable_overwrite_solver", fallback_solver, 
                          "Enable solver fallback if specified solver is not available.");

    // Log levels and transformation
    const std::vector<std::pair<std::string, spdlog::level::level_enum>> SPDLOG_LEVEL_NAMES_TO_LEVELS = {
        {"trace", spdlog::level::trace}, {"debug", spdlog::level::debug}, {"info", spdlog::level::info},
        {"warning", spdlog::level::warn}, {"error", spdlog::level::err}, {"critical", spdlog::level::critical},
        {"off", spdlog::level::off}
    };

    spdlog::level::level_enum log_level = spdlog::level::info;
    command_line.add_option("--log_level", log_level, "Set the logging level for the simulation output.")
                ->transform(CLI::CheckedTransformer(SPDLOG_LEVEL_NAMES_TO_LEVELS, CLI::ignore_case));

    CLI11_PARSE(command_line, argc, argv);

    // Help handling
    if (command_line.get_subcommands().size() == 0) {
        std::cout << command_line.help() << std::endl;
        return EXIT_SUCCESS;
    }

    json in_args = json({});

    // Load input files
    if (!json_file.empty() || !yaml_file.empty()) {
        const bool ok = !json_file.empty() ? load_json(json_file, in_args) : load_yaml(yaml_file, in_args);

        if (!ok)
            log_and_throw_error(fmt::format("Unable to open input file: {}", json_file));

        if (in_args.contains("states"))
            return optimization_simulation(command_line, max_threads, is_strict, log_level, in_args);
        else
            return forward_simulation(command_line, "", output_dir, max_threads, is_strict, fallback_solver, log_level, in_args);
    } else {
        return forward_simulation(command_line, hdf5_file, output_dir, max_threads, is_strict, fallback_solver, log_level, in_args);
    }
}

int forward_simulation(const CLI::App &command_line, const std::string &hdf5_file, const std::string output_dir, 
                       const unsigned max_threads, const bool is_strict, const bool fallback_solver, 
                       const spdlog::level::level_enum &log_level, json &in_args) {

    std::vector<std::string> names;
    std::vector<Eigen::MatrixXi> cells;
    std::vector<Eigen::MatrixXd> vertices;

    if (in_args.empty() && hdf5_file.empty()) {
        logger().error("No input file specified!");
        return command_line.exit(CLI::RequiredError("--json or --hdf5"));
    }

    if (in_args.empty() && !hdf5_file.empty()) {
        using MatrixXl = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;

        h5pp::File file(hdf5_file, h5pp::FileAccess::READONLY);
        std::string json_string = file.readDataset<std::string>("json");

        in_args = json::parse(json_string);
        in_args["root_path"] = hdf5_file;

        names = file.findGroups("", "/meshes");
        cells.resize(names.size());
        vertices.resize(names.size());

        for (int i = 0; i < names.size(); ++i) {
            const std::string &name = names[i];
            cells[i] = file.readDataset<MatrixXl>("/meshes/" + name + "/c").cast<int>();
            vertices[i] = file.readDataset<Eigen::MatrixXd>("/meshes/" + name + "/v");
        }
    }

    json tmp = json::object();
    if (has_arg(command_line, "log_level"))
        tmp["/output/log/level"_json_pointer] = int(log_level);
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

    if (state.mesh == nullptr) {
        return EXIT_FAILURE;  // Error if mesh was not loaded
    }

    state.stats.compute_mesh_stats(*state.mesh);
    state.build_basis();
    state.assemble_rhs();
    state.assemble_mass_mat();

    Eigen::MatrixXd sol, pressure;
    state.solve_problem(sol, pressure);
    state.compute_errors(sol);

    logger().info("total time: {}s", state.timings.total_time());

    state.save_json(sol);
    state.export_data(sol, pressure);

    return EXIT_SUCCESS;
}

int optimization_simulation(const CLI::App &command_line, const unsigned max_threads, const bool is_strict, 
                            const spdlog::level::level_enum &log_level, json &opt_args) {

    json tmp = json::object();
    if (has_arg(command_line, "log_level"))
        tmp["/output/log/level"_json_pointer] = int(log_level);
    if (has_arg(command_line, "max_threads"))
        tmp["/solver/max_threads"_json_pointer] = max_threads;

    opt_args.merge_patch(tmp);

    OptState opt_state;
    opt_state.init(opt_args, is_strict);

    opt_state.create_states(opt_state.args["compute_objective"].get<bool>() ? polyfem::solver::CacheLevel::Solution : polyfem::solver::CacheLevel::Derivatives, 
                            opt_state.args["solver"]["max_threads"].get<int>());
    opt_state.init_variables();
    opt_state.create_problem();

    Eigen::VectorXd x;
    opt_state.initial_guess(x);

    if (opt_state.args["compute_objective"].get<bool>()) {
        logger().info("Objective is {}", opt_state.eval(x));
        return EXIT_SUCCESS;
    }

    opt_state.solve(x);
    return EXIT_SUCCESS;
}
