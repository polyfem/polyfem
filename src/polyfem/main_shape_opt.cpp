#include <filesystem>

#include <CLI/CLI.hpp>
#include <polyfem/Optimizations.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polyfem/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/Problem.hpp>
#include <polyfem/ImplicitTimeIntegrator.hpp>
#include <polyfem/AssemblerUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/ElasticityUtils.hpp>

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <algorithm>

using namespace polyfem;
using namespace polysolve;
using namespace Eigen;
using namespace std::filesystem;

bool has_arg(const CLI::App &command_line, const std::string &value)
{
	const auto *opt = command_line.get_option_no_throw(value.size() == 1 ? ("-" + value) : ("--" + value));
	if (!opt)
		return false;

	return opt->count() > 0;
}

void vector2matrix(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat)
{
	int size = sqrt(vec.size());
	assert(size * size == vec.size());

	mat.resize(size, size);
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			mat(i, j) = vec(i * size + j);
}

int main(int argc, char **argv)
{
	using namespace std::filesystem;

	CLI::App command_line{"polyfem"};
	// Eigen::setNbThreads(1);

	// Input
	std::string mesh_file = "";
	std::string json_file = "";
	std::string febio_file = "";

	// Output
	std::string output_dir = "";
	std::string output_json = "";
	std::string output_vtu = "";
	std::string screenshot = "";

	// Problem
	std::string problem_name = "";
	std::string scalar_formulation = "";
	std::string tensor_formulation = "";
	std::string time_integrator_name = "";
	std::string solver = "";

	std::string bc_method = "";

	int discr_order = 1;

	int n_refs = 0;
	bool use_splines = false;
	bool count_flipped_els = false;
	bool skip_normalization = false;
	bool no_ui = false;
	bool p_ref = false;
	bool force_linear = false;
	bool isoparametric = false;
	bool serendipity = false;
	bool project_to_psd = false;
	bool export_material_params = false;
	bool save_solve_sequence_debug = false;
	bool compute_errors = false;

	std::string log_file = "";
	bool is_quiet = false;
	bool lump_mass_mat = false;
	spdlog::level::level_enum log_level = spdlog::level::debug;
	int nl_solver_rhs_steps = 1;
	int cache_size = -1;
	size_t max_threads = std::numeric_limits<size_t>::max();
	double f_delta = 0;

	bool use_al = false;
	int min_component = -1;

	double vis_mesh_res = -1;

	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	command_line.add_option("-j,--json", json_file, "Simulation json file")->check(CLI::ExistingFile);
	command_line.add_option("-m,--mesh", mesh_file, "Mesh path")->check(CLI::ExistingFile);

	// for debugging
	command_line.add_option("--n_refs", n_refs, "Number of refinements");
	command_line.add_flag("--not_norm", skip_normalization, "Skips mesh normalization");

	const ProblemFactory &p_factory = ProblemFactory::factory();
	command_line.add_option("--problem", problem_name, "Problem name")
		->check(CLI::IsMember(p_factory.get_problem_names()));

	const auto ta = AssemblerUtils::tensor_assemblers();
	command_line.add_option("--tform", tensor_formulation, "Tensor formulation")->check(CLI::IsMember(ta));

	const std::vector<std::string> solvers = LinearSolver::availableSolvers();
	command_line.add_option("--solver", solver, "Linear solver to use")->check(CLI::IsMember(solvers));

	command_line.add_flag("--al", use_al, "Use augmented lagrangian");
	command_line.add_flag("--count_flipped_els", count_flipped_els, "Count flippsed elements");
	command_line.add_flag("--lin_geom", force_linear, "Force use linear geometric mapping");
	command_line.add_option("--vis_mesh_res", vis_mesh_res, "Vis mesh resolution");
	command_line.add_flag("--project_to_psd", project_to_psd, "Project local matrices to psd");
	command_line.add_flag("--lump_mass_mat", lump_mass_mat, "Lump the mass matrix");

	const std::vector<std::string> bc_methods = {"", "sample", "lsq"}; //, "integrate"};
	command_line.add_option("--bc_method", bc_method, "Method used for boundary conditions")->check(CLI::IsMember(bc_methods));

	command_line.add_option("--cache_size", cache_size, "Size of the cached assembly values");
	command_line.add_option("--min_component", min_component, "Mimimum number of faces in connected compoment for contact");

	command_line.add_option("--log_file", log_file, "Log to a file");
	const std::vector<std::pair<std::string, spdlog::level::level_enum>>
		SPDLOG_LEVEL_NAMES_TO_LEVELS = {
			{"trace", spdlog::level::trace},
			{"debug", spdlog::level::debug},
			{"info", spdlog::level::info},
			{"warning", spdlog::level::warn},
			{"error", spdlog::level::err},
			{"critical", spdlog::level::critical},
			{"off", spdlog::level::off}};
	command_line.add_option("--log_level", log_level, "Log level")
		->transform(CLI::CheckedTransformer(SPDLOG_LEVEL_NAMES_TO_LEVELS, CLI::ignore_case));

	command_line.add_flag("--export_material_params", export_material_params, "Export material parameters");

	const auto &time_integrator_names = ImplicitTimeIntegrator::get_time_integrator_names();
	command_line.add_option("--time_integrator", time_integrator_name, "Time integrator name")
		->check(CLI::IsMember(time_integrator_names));

	// CCD
	int ccd_max_iterations;
	std::string broad_phase_method;
	const std::vector<std::string> ccd_methods = {"", "brute_force", "spatial_hash", "hash_grid"};
	command_line.add_option("--ccd_max_iterations", ccd_max_iterations, "Max number of CCD iterations");
	command_line.add_option("--broad_phase_method", broad_phase_method, "CCD Method")
		->check(CLI::IsMember({"brute_force", "hash_grid", "spatial_hash"}));

	CLI11_PARSE(command_line, argc, argv);

	if (!screenshot.empty())
	{
		no_ui = false;
	}

	json in_args = json({});

	if (!json_file.empty())
	{
		std::ifstream file(json_file);

		if (file.is_open())
			file >> in_args;
		else
			logger().error("unable to open {} file", json_file);
		file.close();

		in_args["root_path"] = json_file;

		if (in_args.contains("default_params"))
		{
			apply_default_params(in_args);
			in_args.erase("default_params"); // Remove this so state does not redo the apply
		}
	}
	else
	{
		in_args["root_path"] = mesh_file;
		if (has_arg(command_line, "mesh") || has_arg(command_line, "m"))
			in_args["mesh"] = mesh_file;

		if (has_arg(command_line, "lin_geom"))
			in_args["force_linear_geometry"] = force_linear;

		if (has_arg(command_line, "n_refs"))
			in_args["n_refs"] = n_refs;

		if (!problem_name.empty())
			in_args["problem"] = problem_name;

		if (!time_integrator_name.empty())
			in_args["time_integrator"] = time_integrator_name;

		if (has_arg(command_line, "not_norm"))
			in_args["normalize_mesh"] = !skip_normalization;

		if (has_arg(command_line, "project_to_psd"))
			in_args["project_to_psd"] = project_to_psd;

		if (has_arg(command_line, "al"))
			in_args["use_al"] = use_al;

		if (!scalar_formulation.empty())
			in_args["scalar_formulation"] = scalar_formulation;
		if (!tensor_formulation.empty())
			in_args["tensor_formulation"] = tensor_formulation;

		if (has_arg(command_line, "p") || has_arg(command_line, "q"))
			in_args["discr_order"] = discr_order;

		if (has_arg(command_line, "spline"))
			in_args["use_spline"] = use_splines;

		if (has_arg(command_line, "count_flipped_els"))
			in_args["count_flipped_els"] = count_flipped_els;

		if (has_arg(command_line, "p_ref"))
			in_args["use_p_ref"] = p_ref;
		if (has_arg(command_line, "isoparametric"))
			in_args["iso_parametric"] = isoparametric;
		if (has_arg(command_line, "serendipity"))
			in_args["serendipity"] = serendipity;

		if (has_arg(command_line, "n_incr_load"))
			in_args["nl_solver_rhs_steps"] = nl_solver_rhs_steps;
		if (has_arg(command_line, "save_incr_load"))
			in_args["save_solve_sequence_debug"] = save_solve_sequence_debug;

		if (has_arg(command_line, "export_material_params"))
			in_args["export"]["material_params"] = true;

		if (has_arg(command_line, "f_delta"))
		{
			in_args["solver_params"] = {};
			in_args["solver_params"]["fDelta"] = f_delta;
		}
	}

	if (!bc_method.empty())
		in_args["bc_method"] = bc_method;

	if (!in_args.contains("lump_mass_matrix") && has_arg(command_line, "lump_mass_mat"))
		in_args["lump_mass_matrix"] = lump_mass_mat;

	if (has_arg(command_line, "compute_errors"))
		in_args["compute_error"] = compute_errors;

	if (!solver.empty())
		in_args["solver_type"] = solver;
	if (cache_size >= 0)
		in_args["cache_size"] = cache_size;
	if (!output_vtu.empty())
	{
		in_args["export"]["vis_mesh"] = output_vtu;
		in_args["export"]["wire_mesh"] = StringUtils::replace_ext(output_vtu, "obj");
	}

	if (has_arg(command_line, "ccd_max_iterations"))
		in_args["solver_params"]["ccd_max_iterations"] = ccd_max_iterations;

	if (has_arg(command_line, "broad_phase_method"))
		in_args["solver_params"]["broad_phase_method"] = broad_phase_method;

	if (min_component > 0)
		in_args["min_component"] = min_component;

	if (vis_mesh_res > 0)
		in_args["vismesh_rel_area"] = vis_mesh_res;

	if (!in_args.contains("rhs_solver_type") && in_args.contains("solver_type"))
	{
		in_args["rhs_solver_type"] = in_args["solver_type"];
	}

	// create solver
	State state(max_threads);
	state.init_logger(log_file, log_level, is_quiet);
	state.init(in_args, output_dir);

	if (state.args["has_collision"] && !state.args.contains("barrier_stiffness"))
	{
		logger().error("Not fixing the barrier stiffness!");
		return EXIT_FAILURE;
	}

	// load mesh
	if (!febio_file.empty())
		state.load_febio(febio_file, in_args);
	else
		state.load_mesh();

	if (state.mesh == nullptr)
		return EXIT_FAILURE;

	const int p = state.args.contains("functional_power") ? state.args["functional_power"].get<int>() : 2;
	std::shared_ptr<CompositeFunctional> func = CompositeFunctional::create("Stress");
	func->set_power(p);

	state.compute_mesh_stats();
	state.build_basis();

	shape_optimization(state, func, state.args["optimization"]);

	return EXIT_SUCCESS;
}
