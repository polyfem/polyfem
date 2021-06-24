#include <CLI/CLI.hpp>
#include <polyfem/State.hpp>
#ifndef POLYFEM_NO_UI
#include <polyfem/UIState.hpp>
#endif

#include <polysolve/LinearSolver.hpp>
#include <polyfem/StringUtils.hpp>
#include <polyfem/Logger.hpp>

#include <polyfem/Problem.hpp>
#include <polyfem/ImplicitTimeIntegrator.hpp>
#include <polyfem/AssemblerUtils.hpp>

#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <ghc/fs_std.hpp> // filesystem

using namespace polyfem;
using namespace polysolve;
using namespace Eigen;

int main(int argc, char **argv)
{
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
	std::string problem_name = "Franke";
	std::string scalar_formulation = "Laplacian";
	std::string tensor_formulation = "LinearElasticity"; // "SaintVenant";
	std::string time_integrator_name = "ImplicitEuler";
	// std::string mixed_formulation = "Stokes"; // "SaintVenant";
	std::string solver = "";

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

	std::string log_file = "";
	bool is_quiet = false;
	bool stop_after_build_basis = false;
	int log_level = 1;
	int nl_solver_rhs_steps = 1;
	int cache_size = -1;

	bool use_al = false;
	int min_component = -1;

	double vis_mesh_res = -1;

	command_line.add_option("-j,--json", json_file, "Simulation json file")->check(CLI::ExistingFile);
	command_line.add_option("-m,--mesh", mesh_file, "Mesh path")->check(CLI::ExistingFile);
	command_line.add_option("-b,--febio", febio_file, "FEBio file path")->check(CLI::ExistingFile);

	// for debugging
	command_line.add_option("--n_refs", n_refs, "Number of refinements");
	command_line.add_flag("--not_norm", skip_normalization, "Skips mesh normalization");

	const ProblemFactory &p_factory = ProblemFactory::factory();
	command_line.add_set("--problem", problem_name, std::set<std::string>(p_factory.get_problem_names().begin(), p_factory.get_problem_names().end()), "Problem name");

	const auto sa = AssemblerUtils::scalar_assemblers();
	const auto ta = AssemblerUtils::tensor_assemblers();
	command_line.add_set("--sform", scalar_formulation, std::set<std::string>(sa.begin(), sa.end()), "Scalar formulation");
	command_line.add_set("--tform", tensor_formulation, std::set<std::string>(ta.begin(), ta.end()), "Tensor formulation");
	// command_line.add_set("--mform", mixed_formulation, std::set<std::string>(assembler.mixed_assemblers().begin(), assembler.mixed_assemblers().end()),  "Mixed formulation");

	const std::vector<std::string> solvers = LinearSolver::availableSolvers();
	command_line.add_set("--solver", solver, std::set<std::string>(solvers.begin(), solvers.end()), "Linear solver to use");

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
	command_line.add_flag("--project_to_psd", project_to_psd, "Project local matrices to psd");
	command_line.add_option("--n_incr_load", nl_solver_rhs_steps, "Number of incremeltal load");
	command_line.add_flag("--save_incr_load", save_solve_sequence_debug, "Save incremental steps");

	command_line.add_option("--cache_size", cache_size, "Size of the cached assembly values");
	command_line.add_option("--min_component", min_component, "Mimimum number of faces in connected compoment for contact");

	// disable out
	command_line.add_flag("--cmd,--ngui", no_ui, "Runs in command line mode, no GUI");

	// IO
	command_line.add_option("-o,--output_dir", output_dir, "Directory for output files")->check(CLI::ExistingDirectory | CLI::NonexistentPath);
	command_line.add_option("--output,--output_json", output_json, "Output json file");
	command_line.add_option("--vtu,--output_vtu", output_vtu, "Output VTU file");
	command_line.add_option("--screenshot", screenshot, "screenshot (disabled)");

	command_line.add_flag("--quiet", is_quiet, "Disable cout for logging");
	command_line.add_option("--log_file", log_file, "Log to a file");
	command_line.add_option("--log_level", log_level, "Log level 0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical, 6=off");

	command_line.add_flag("--export_material_params", export_material_params, "Export material parameters");

	const auto &time_integrator_names = ImplicitTimeIntegrator::get_time_integrator_names();
	command_line.add_set("--time_integrator", time_integrator_name, std::set<std::string>(time_integrator_names.begin(), time_integrator_names.end()), "Time integrator name");

	try
	{
		command_line.parse(argc, argv);
	}
	catch (const CLI::ParseError &e)
	{
		return command_line.exit(e);
	}

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

		// If the mesh path does not exist make it relative to the json_file
		if (in_args.contains("mesh"))
		{
			in_args["mesh"] = resolve_path(in_args["mesh"], json_file);
		}

		// If the meshes paths does not exist make it relative to the json_file
		if (in_args.contains("meshes") && !in_args["meshes"].empty())
		{
			for (json &json_mesh : in_args["meshes"])
			{
				if (json_mesh.contains("mesh"))
				{
					json_mesh["mesh"] = resolve_path(json_mesh["mesh"], json_file);
				}
			}
		}
	}
	else
	{
		in_args["root_path"] = mesh_file;
		in_args["mesh"] = mesh_file;
		in_args["force_linear_geometry"] = force_linear;
		in_args["n_refs"] = n_refs;
		in_args["problem"] = problem_name;
		in_args["time_integrator"] = time_integrator_name;
		in_args["normalize_mesh"] = !skip_normalization;
		in_args["project_to_psd"] = project_to_psd;

		if (use_al)
			in_args["use_al"] = use_al;

		in_args["scalar_formulation"] = scalar_formulation;
		in_args["tensor_formulation"] = tensor_formulation;
		// in_args["mixed_formulation"] = mixed_formulation;

		in_args["discr_order"] = discr_order;
		in_args["use_spline"] = use_splines;
		in_args["count_flipped_els"] = count_flipped_els;

		in_args["use_p_ref"] = p_ref;
		in_args["iso_parametric"] = isoparametric;
		in_args["serendipity"] = serendipity;

		in_args["nl_solver_rhs_steps"] = nl_solver_rhs_steps;
		in_args["save_solve_sequence_debug"] = save_solve_sequence_debug;

		if (export_material_params)
			in_args["export"]["material_params"] = true;
	}

	if (!output_json.empty())
		in_args["output"] = output_json;
	if (!solver.empty())
		in_args["solver_type"] = solver;
	if (cache_size >= 0)
		in_args["cache_size"] = cache_size;
	if (!output_vtu.empty())
	{
		in_args["export"]["vis_mesh"] = output_vtu;
		in_args["export"]["wire_mesh"] = StringUtils::replace_ext(output_vtu, "obj");
	}

	if (min_component > 0)
		in_args["min_component"] = min_component;

	if (vis_mesh_res > 0)
		in_args["vismesh_rel_area"] = vis_mesh_res;

	if (!output_dir.empty())
	{
		fs::create_directories(output_dir);
	}

	if (!output_json.empty())
	{
		fs::create_directories(fs::path(output_json).parent_path());
	}

#ifndef POLYFEM_NO_UI
	if (no_ui)
	{
#endif
		State state;
		state.init_logger(log_file, log_level, is_quiet);
		state.init(in_args, output_dir);

		if (!febio_file.empty())
			state.load_febio(febio_file);
		else
			state.load_mesh();

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
