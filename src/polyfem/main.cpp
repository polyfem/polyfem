#include <CLI/CLI.hpp>
#include <polyfem/State.hpp>
#ifndef POLYFEM_NO_UI
#include <polyfem/viewer/UIState.hpp>
#endif

#include <polysolve/LinearSolver.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/assembler/Problem.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/problem/ProblemFactory.hpp>

#include <polyfem/utils/JSONUtils.hpp>

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

#include <filesystem>

using namespace polyfem;
using namespace polyfem::problem;
using namespace polyfem::assembler;
using namespace polyfem::utils;
using namespace polysolve;
using namespace Eigen;
using namespace HighFive;

bool has_arg(const CLI::App &command_line, const std::string &value)
{
	const auto *opt = command_line.get_option_no_throw(value.size() == 1 ? ("-" + value) : ("--" + value));
	if (!opt)
		return false;

	return opt->count() > 0;
}

int main(int argc, char **argv)
{
	using namespace std::filesystem;

	CLI::App command_line{"polyfem"};
	// Eigen::setNbThreads(1);

	// Input
	std::string mesh_file = "";
	std::string json_file = "";
	std::string hdf5_file = "";
	std::string febio_file = "";

	// Output
	std::string output_dir = "";
	std::string output_json = "";
	std::string output_vtu = "";
	std::string screenshot = "";

	// Problem
	std::string problem_name = "";
	std::string formulation = "";
	std::string time_integrator_name = "";
	std::string solver = "";

	std::string bc_method = "";

	int discr_order = 1;

	int n_refs = 0;
	bool use_splines = false;
	bool count_flipped_els = true;
	bool skip_normalization = true;
	bool no_ui = false;
	bool p_ref = false;
	bool force_linear = false;
	bool isoparametric = false;
	bool serendipity = false;
	bool export_material_params = false;
	bool save_solve_sequence_debug = false;
	bool compute_errors = false;

	std::string log_file = "";
	bool is_quiet = false;
	bool stop_after_build_basis = false;
	bool lump_mass_mat = false;
	spdlog::level::level_enum log_level = spdlog::level::debug;
	int cache_size = -1;
	size_t max_threads = std::numeric_limits<size_t>::max();
	double f_delta = 0;

	bool use_al = false;
	int min_component = -1;

	double vis_mesh_res = -1;

	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	command_line.add_option("-j,--json", json_file, "Simulation json file")->check(CLI::ExistingFile);
	command_line.add_option("--hdf5", hdf5_file, "Simulation hdf5 file")->check(CLI::ExistingFile);
	command_line.add_option("-m,--mesh", mesh_file, "Mesh path")->check(CLI::ExistingFile);
	command_line.add_option("-b,--febio", febio_file, "FEBio file path")->check(CLI::ExistingFile);

	// for debugging
	command_line.add_option("--n_refs", n_refs, "Number of refinements");
	command_line.add_flag("--not_norm", skip_normalization, "Skips mesh normalization");

	const ProblemFactory &p_factory = ProblemFactory::factory();
	command_line.add_option("--problem", problem_name, "Problem name")
		->check(CLI::IsMember(p_factory.get_problem_names()));

	const auto sa = AssemblerUtils::scalar_assemblers();
	const auto ta = AssemblerUtils::tensor_assemblers();
	std::vector forms = sa;
	forms.insert(forms.end(), ta.begin(), ta.end());
	command_line.add_option("--PDE", formulation, "PDE")->check(CLI::IsMember(forms));

	const std::vector<std::string> solvers = LinearSolver::availableSolvers();
	command_line.add_option("--solver", solver, "Linear solver to use")->check(CLI::IsMember(solvers));

	command_line.add_flag("--al", use_al, "Use augmented lagrangian");
	command_line.add_option("-q,-p", discr_order, "Discretization order");
	command_line.add_flag("--p_ref", p_ref, "Use p refimenet");
	command_line.add_flag("--spline", use_splines, "Use spline for quad/hex meshes");
	command_line.add_flag("--count_flipped_els", count_flipped_els, "Count flippsed elements");
	command_line.add_flag("--lin_geom", force_linear, "Force use linear geometric mapping");
	command_line.add_flag("--isoparametric", isoparametric, "Force use isoparametric basis");
	command_line.add_flag("--serendipity", serendipity, "Use of serendipity elements, only for Q2");
	command_line.add_flag("--stop_after_build_basis", stop_after_build_basis, "Stop after build bases");
	command_line.add_option("--vis_mesh_res", vis_mesh_res, "Vis mesh resolution");
	command_line.add_flag("--save_solve_sequence_debug", save_solve_sequence_debug, "Save incremental steps");
	command_line.add_flag("--lump_mass_mat", lump_mass_mat, "Lump the mass matrix");
	command_line.add_flag("--compute_errors", compute_errors, "Computes the errors");

	const std::vector<std::string> bc_methods = {"", "sample", "lsq"}; //, "integrate"};
	command_line.add_option("--bc_method", bc_method, "Method used for boundary conditions")->check(CLI::IsMember(bc_methods));

	command_line.add_option("--cache_size", cache_size, "Size of the cached assembly values");
	command_line.add_option("--min_component", min_component, "Mimimum number of faces in connected compoment for contact");

	// disable out
	command_line.add_flag("--cmd,--ngui,!--gui", no_ui, "Runs in command line mode, no GUI");

	// IO
	command_line.add_option("-o,--output_dir", output_dir, "Directory for output files")->check(CLI::ExistingDirectory | CLI::NonexistentPath);
	command_line.add_option("--output,--output_json", output_json, "Output json file");
	command_line.add_option("--vtu,--output_vtu", output_vtu, "Output VTU file");
	command_line.add_option("--screenshot", screenshot, "screenshot (disabled)");
	command_line.add_option("--f_delta", f_delta, "non linear tolerance");

	command_line.add_flag("--quiet", is_quiet, "Disable cout for logging");
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

	const auto &time_integrator_names = time_integrator::ImplicitTimeIntegrator::get_time_integrator_names();
	command_line.add_option("--time_integrator", time_integrator_name, "Time integrator name")
		->check(CLI::IsMember(time_integrator_names));

	// CCD
	int ccd_max_iterations;
	std::string broad_phase_method;
	command_line.add_option("--ccd_max_iterations", ccd_max_iterations, "Max number of CCD iterations");
#ifdef IPC_TOOLKIT_WITH_CUDA
	command_line.add_option("--bp,--broad_phase_method", broad_phase_method, "CCD Method")
		->check(CLI::IsMember({"brute_force", "BF",
							   "hash_grid", "HG",
							   "spatial_hash", "SH",
							   "sweep_and_tiniest_queue", "STQ",
							   "sweep_and_tiniest_queue_gpu", "STQ_GPU"}));
#else
	command_line.add_option("--bp,--broad_phase_method", broad_phase_method, "CCD Method")
		->check(CLI::IsMember({"brute_force", "BF",
							   "hash_grid", "HG",
							   "spatial_hash", "SH",
							   "sweep_and_tiniest_queue", "STQ"}));
#endif

	CLI11_PARSE(command_line, argc, argv);

	std::vector<std::string> names;
	std::vector<Eigen::MatrixXi> cells;
	std::vector<Eigen::MatrixXd> vertices;

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

		if (in_args.contains("defaults"))
		{
			apply_default_params(in_args);
			in_args.erase("defaults"); // Remove this so state does not redo the apply
		}
	}
	else if (!hdf5_file.empty())
	{
		File file(hdf5_file, File::ReadOnly);
		std::string json_string = H5Easy::load<std::string>(file, "json");

		in_args = json::parse(json_string);
		in_args["root_path"] = hdf5_file;

		Group meshes = file.getGroup("meshes");
		names = meshes.listObjectNames();
		cells.resize(names.size());
		vertices.resize(names.size());

		for (size_t i = 0; i < names.size(); ++i)
		{
			const auto &s = names[i];
			const auto &tmp = meshes.getGroup(s);

			tmp.getDataSet("c").read(cells[i]);
			tmp.getDataSet("v").read(vertices[i]);
		}
	}
	else
	{
		in_args["root_path"] = mesh_file;
		if (has_arg(command_line, "mesh") || has_arg(command_line, "m"))
			in_args["mesh"] = mesh_file;

		//TODO
		// if (has_arg(command_line, "lin_geom"))
		// 	in_args["force_linear_geometry"] = force_linear;

		// if (has_arg(command_line, "n_refs"))
		// 	in_args["n_refs"] = n_refs;

		// if (has_arg(command_line, "not_norm"))
		// 	in_args["normalize_mesh"] = !skip_normalization;

		if (!problem_name.empty())
		{
			in_args["preset_problem"] = {};
			in_args["preset_problem"]["name"] = problem_name;
		}

		if (!time_integrator_name.empty())
		{
			in_args["time"] = {};
			in_args["time"]["integrator"] = time_integrator_name;
		}

		if (!formulation.empty())
		{
			in_args["materials"] = {};
			in_args["materials"]["type"] = formulation;
		}

		if (has_arg(command_line, "p") || has_arg(command_line, "q"))
		{
			in_args["space"] = {};
			in_args["space"]["discr_order"] = discr_order;
		}

		if (has_arg(command_line, "spline"))
		{
			if (!in_args.contains("space"))
				in_args["space"] = {};
			in_args["space"]["advanced"] = {};
			in_args["space"]["advanced"]["use_spline"] = use_splines;
		}

		if (has_arg(command_line, "count_flipped_els"))
		{
			if (!in_args.contains("space"))
				in_args["space"] = {};
			if (!in_args["space"].contains("advanced"))
				in_args["space"]["advanced"] = {};
			in_args["space"]["advanced"]["count_flipped_els"] = count_flipped_els;
		}

		if (has_arg(command_line, "p_ref"))
		{
			if (!in_args.contains("space"))
				in_args["space"] = {};
			in_args["space"]["use_p_ref"] = p_ref;
		}
		if (has_arg(command_line, "isoparametric"))
		{
			if (!in_args.contains("space"))
				in_args["space"] = {};
			if (!in_args["space"].contains("advanced"))
				in_args["space"]["advanced"] = {};
			in_args["space"]["advanced"]["iso_parametric"] = isoparametric;
		}
		if (has_arg(command_line, "serendipity"))
		{
			if (!in_args.contains("space"))
				in_args["space"] = {};
			if (!in_args["space"].contains("advanced"))
				in_args["space"]["advanced"] = {};
			in_args["space"]["advanced"]["serendipity"] = serendipity;
		}

		if (has_arg(command_line, "save_solve_sequence_debug"))
		{
			in_args["output"] = {};
			in_args["output"]["advanced"] = {};
			in_args["output"]["advanced"]["save_solve_sequence_debug"] = save_solve_sequence_debug;
		}

		if (has_arg(command_line, "export_material_params"))
		{
			if (!in_args.contains("output"))
				in_args["output"]["options"] = {};
			in_args["output"]["options"]["material"] = true;
		}

		if (has_arg(command_line, "f_delta"))
		{
			in_args["solver"] = {};
			in_args["solver"]["nonlinear"] = {};
			in_args["solver"]["nonlinear"]["fDelta"] = f_delta;
		}
	}

	if (!bc_method.empty())
	{
		if (!in_args.contains("space"))
			in_args["space"] = {};
		if (!in_args["space"].contains("advanced"))
			in_args["space"]["advanced"] = {};
		in_args["space"]["advanced"]["bc_method"] = bc_method;
	}

	if (!in_args.contains("lump_mass_matrix") && has_arg(command_line, "lump_mass_mat"))
	{
		if (!in_args.contains("solver"))
			in_args["solver"] = {};
		if (!in_args["solver"].contains("advanced"))
			in_args["solver"]["advanced"] = {};
		in_args["solver"]["advanced"]["lump_mass_matrix"] = lump_mass_mat;
	}

	if (has_arg(command_line, "compute_errors"))
	{
		if (!in_args.contains("output"))
			in_args["output"] = {};
		if (!in_args["output"].contains("advanced"))
			in_args["output"]["advanced"] = {};
		in_args["output"]["advanced"]["compute_error"] = compute_errors;
	}

	if (!output_json.empty())
	{
		if (!in_args.contains("output"))
			in_args["output"] = {};
		in_args["output"]["json"] = output_json;
	}
	if (!solver.empty())
	{
		if (!in_args.contains("solver"))
			in_args["solver"] = {};
		if (!in_args["solver"].contains("linear"))
			in_args["solver"]["linear"] = {};
		in_args["solver"]["linear"]["solver"] = solver;
	}
	if (cache_size >= 0)
	{
		if (!in_args.contains("solver"))
			in_args["solver"] = {};
		if (!in_args["solver"].contains("advanced"))
			in_args["solver"]["advanced"] = {};
		in_args["solver"]["advanced"]["cache_size"] = cache_size;
	}
	if (!output_vtu.empty())
	{
		if (!in_args.contains("output"))
			in_args["output"] = {};
		if (!in_args["output"].contains("paraview"))
			in_args["output"]["paraview"] = {};
		in_args["output"]["paraview"]["file_name"] = output_vtu;
	}

	if (has_arg(command_line, "ccd_max_iterations"))
	{
		if (!in_args.contains("solver"))
			in_args["solver"] = {};
		in_args["solver"]["contact"] = {};
		in_args["solver"]["contact"]["CCD"] = {};
		in_args["solver"]["contact"]["CCD"]["max_iterations"] = ccd_max_iterations;
	}

	if (has_arg(command_line, "broad_phase_method"))
	{
		if (!in_args.contains("solver"))
			in_args["solver"] = {};
		if (!in_args["solver"].contains("contact"))
			in_args["solver"]["contact"] = {};
		if (!in_args["solver"]["contact"].contains("CCD"))
			in_args["solver"]["contact"]["CCD"] = {};
		in_args["solver"]["contact"]["CCD"]["broad_phase"] = broad_phase_method;
	}

	//TODO
	// if (min_component > 0)
	// 	in_args["min_component"] = min_component;

	if (vis_mesh_res > 0)
	{
		if (!in_args.contains("output"))
			in_args["output"] = {};
		if (!in_args["output"].contains("paraview"))
			in_args["output"]["paraview"] = {};
		in_args["vismesh_rel_area"] = vis_mesh_res;
	}

	if (!output_dir.empty())
	{
		create_directories(output_dir);
	}

	if (!output_json.empty())
	{
		const auto parent = path(output_json).parent_path();
		if (!parent.empty())
			create_directories(parent);
	}

#ifndef POLYFEM_NO_UI
	if (no_ui)
	{
#endif
		State state(max_threads);
		state.init_logger(log_file, log_level, is_quiet);
		state.init(in_args, output_dir);

		if (!febio_file.empty())
			state.load_febio(febio_file, in_args);
		else
			state.load_mesh(false, names, cells, vertices);

		// Mesh was not loaded successfully; load_mesh() logged the error.
		if (state.mesh == nullptr)
		{
			// Cannot proceed without a mesh.
			return EXIT_FAILURE;
		}

		state.compute_mesh_stats();

		state.build_basis();

		if (stop_after_build_basis)
			return EXIT_SUCCESS;

		state.assemble_rhs();
		state.assemble_stiffness_mat();

		state.solve_problem();

		state.compute_errors();

		state.save_json();
		state.export_data();
#ifndef POLYFEM_NO_UI
	}
	else
	{
		UIState::ui_state().launch(log_file, log_level, is_quiet, in_args, febio_file);
	}
#endif

	return EXIT_SUCCESS;
}
