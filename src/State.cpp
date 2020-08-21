#include <polyfem/State.hpp>

#include <polyfem/StringUtils.hpp>

#include <polyfem/Mesh2D.hpp>
#include <polyfem/Mesh3D.hpp>
#include <polyfem/FEBioReader.hpp>

#include <polyfem/FEBasis2d.hpp>
#include <polyfem/FEBasis3d.hpp>

#include <polyfem/SpectralBasis2d.hpp>

#include <polyfem/SplineBasis2d.hpp>
#include <polyfem/SplineBasis3d.hpp>

#include <polyfem/EdgeSampler.hpp>
#include <polyfem/BoundarySampler.hpp>

#include <polyfem/PolygonalBasis2d.hpp>
#include <polyfem/PolygonalBasis3d.hpp>

#include <polyfem/MVPolygonalBasis2d.hpp>

#include <polyfem/AssemblerUtils.hpp>
#include <polyfem/RhsAssembler.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/RefElementSampler.hpp>

#include <polyfem/Common.hpp>

#include <polyfem/VTUWriter.hpp>
#include <polyfem/MeshUtils.hpp>

#include <polyfem/NLProblem.hpp>
#include <polyfem/LbfgsSolver.hpp>
#include <polyfem/SparseNewtonDescentSolver.hpp>
#include <polyfem/NavierStokesSolver.hpp>
#include <polyfem/TransientNavierStokesSolver.hpp>
#include <polyfem/OperatorSplittingSolver.hpp>

#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <polyfem/HexQuadrature.hpp>
#include <polyfem/QuadQuadrature.hpp>
#include <polyfem/TetQuadrature.hpp>
#include <polyfem/TriQuadrature.hpp>
#include <polyfem/BDF.hpp>

#include <polyfem/Logger.hpp>

#ifdef POLYFEM_WITH_TBB
#include <tbb/task_scheduler_init.h>
#endif

#include <igl/Timer.h>
#include <igl/remove_unreferenced.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/isolines.h>
#include <igl/write_triangle_mesh.h>

#include <igl/per_face_normals.h>
#include <igl/AABB.h>
#include <igl/in_element.h>

#include <unsupported/Eigen/SparseExtra>


#include <iostream>
#include <algorithm>
#include <memory>
#include <math.h>

#include <polyfem/autodiff.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
DECLARE_DIFFSCALAR_BASE();

using namespace Eigen;

extern "C" size_t getPeakRSS();

namespace polyfem
{
	using namespace polysolve;
namespace
{
template <typename V1, typename V2>
double angle2(const V1 &v1, const V2 &v2)
{
	assert(v1.size() == 2);
	assert(v2.size() == 2);
	return std::abs(atan2(v1(0) * v2(1) - v1(1) * v2(0), v1.dot(v2)));
}

template <typename V1, typename V2>
double angle3(const V1 &v1, const V2 &v2)
{
	assert(v1.size() == 3);
	assert(v2.size() == 3);
	return std::abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));
}

class GeoLoggerForward : public GEO::LoggerClient
{
	std::shared_ptr<spdlog::logger> logger_;

public:
	template <typename T>
	GeoLoggerForward(T logger) : logger_(logger) {}

private:
	std::string truncate(const std::string &msg)
	{
		static size_t prefix_len = GEO::CmdLine::ui_feature(" ", false).size();
		return msg.substr(prefix_len, msg.size() - 1 - prefix_len);
	}

protected:
	void div(const std::string &title) override
	{
		logger_->trace(title.substr(0, title.size() - 1));
	}

	void out(const std::string &str) override
	{
		logger_->info(truncate(str));
	}

	void warn(const std::string &str) override
	{
		logger_->warn(truncate(str));
	}

	void err(const std::string &str) override
	{
		logger_->error(truncate(str));
	}

	void status(const std::string &str) override
	{
		// Errors and warnings are also dispatched as status by geogram, but without
		// the "feature" header. We thus forward them as trace, to avoid duplicated
		// logger info...
		logger_->trace(str.substr(0, str.size() - 1));
	}
};

} // namespace

State::State()
{
#ifndef WIN32
	setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif

	GEO::initialize();

#ifdef USE_TBB
	const size_t MB = 1024 * 1024;
	const size_t stack_size = 64 * MB;
	unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
	tbb::task_scheduler_init scheduler(num_threads, stack_size);
#endif

	// Import standard command line arguments, and custom ones
	GEO::CmdLine::import_arg_group("standard");
	GEO::CmdLine::import_arg_group("pre");
	GEO::CmdLine::import_arg_group("algo");

	problem = ProblemFactory::factory().get_problem("Linear");

	use_avg_pressure = true;

	this->args = {
		{"mesh", ""},
		{"force_linear_geometry", false},
		{"bc_tag", ""},
		{"boundary_id_threshold", -1.0},
		{"n_refs", 0},
		{"vismesh_rel_area", 0.00001},
		{"refinenemt_location", 0.5},
		{"n_boundary_samples", 6},
		{"problem", "Franke"},
		{"normalize_mesh", true},

		{"curved_mesh_size", false},

		{"count_flipped_els", false},

		{"tend", 1},
		{"time_steps", 10},

		{"scalar_formulation", "Laplacian"},
		{"tensor_formulation", "LinearElasticity"},

		{"B", 3},
		{"h1_formula", false},

		{"BDF_order", 1},
		{"quadrature_order", 4},
		{"discr_order", 1},
		{"poly_bases", "MFSHarmonic"},
		{"serendipity", false},
		{"discr_order_max", autogen::MAX_P_BASES},
		{"pressure_discr_order", 1},
		{"use_p_ref", false},
		{"use_spline", false},
		{"iso_parametric", false},
		{"integral_constraints", 2},

		{"fit_nodes", false},

		{"n_harmonic_samples", 10},

		{"solver_type", LinearSolver::defaultSolver()},
		{"precond_type", LinearSolver::defaultPrecond()},

		{"solver_params", json({})},
		{"line_search", "armijo"},
		{"nl_solver", "newton"},
		{"nl_solver_rhs_steps", 1},
		{"save_solve_sequence", false},
		{"save_solve_sequence_debug", false},
		{"save_time_sequence", true},
		{"skip_frame", 1},

		{"force_no_ref_for_harmonic", false},

		{"rhs_path", ""},

		{"particle", false},
		{"density", false},
		{"advection_order", 1},
		{"advection_RK", 1},

		{"params", {{"lambda", 0.32967032967032966}, {"mu", 0.3846153846153846}, {"k", 1.0}, {"elasticity_tensor", json({})},
					// {"young", 1.0},
					// {"nu", 0.3},
					{"alphas", {2.13185026692482, -0.600299816209491}},
					{"mus", {0.00407251192475097, 0.000167202574129608}},
					{"Ds", {9.4979, 1000000}}}},

		{"problem_params", json({})},

		{"output", ""},
		// {"solution", ""},
		// {"stiffness_mat_save_path", ""},

		{"export", {
			{"sol_at_node", -1},
			{"vis_mesh", ""},
			{"paraview", ""},
			{"vis_boundary_only", false},
			{"material_params", false},
			{"nodes", ""},
			{"wire_mesh", ""},
			{"iso_mesh", ""},
			{"spectrum", false},
			{"solution", ""},
			{"full_mat", ""},
			{"stiffness_mat", ""},
			{"solution_mat", ""},
			{"stress_mat", ""},
			{"mises", ""}
		}}};
}

void State::init_logger(const std::string &log_file, int log_level, const bool is_quiet)
{
	Logger::init(!is_quiet, log_file);
	log_level = std::max(0, std::min(6, log_level));
	spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
	spdlog::flush_every(std::chrono::seconds(3));

	GEO::Logger *geo_logger = GEO::Logger::instance();
	geo_logger->unregister_all_clients();
	geo_logger->register_client(new GeoLoggerForward(logger().clone("geogram")));
	geo_logger->set_pretty(false);
}

void State::init_logger(std::ostream &os, int log_level)
{
	Logger::init(os);
	log_level = std::max(0, std::min(6, log_level));
	spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
	spdlog::flush_every(std::chrono::seconds(3));

	GEO::Logger *geo_logger = GEO::Logger::instance();
	geo_logger->unregister_all_clients();
	geo_logger->register_client(new GeoLoggerForward(logger().clone("geogram")));
	geo_logger->set_pretty(false);
}

void State::init_logger(std::vector<spdlog::sink_ptr> &sinks, int log_level)
{
	Logger::init(sinks);
	log_level = std::max(0, std::min(6, log_level));
	spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));

	GEO::Logger *geo_logger = GEO::Logger::instance();
	geo_logger->unregister_all_clients();
	geo_logger->register_client(new GeoLoggerForward(logger().clone("geogram")));
	geo_logger->set_pretty(false);
}

void State::sol_to_pressure()
{
	if (n_pressure_bases <= 0)
	{
		logger().error("No pressure bases defined!");
		return;
	}

	// assert(problem->is_mixed());
	assert(AssemblerUtils::instance().is_mixed(formulation()));
	Eigen::MatrixXd tmp = sol;

	int fluid_offset = use_avg_pressure ? (AssemblerUtils::instance().is_fluid(formulation()) ? 1 : 0) : 0;
	sol = tmp.block(0, 0, tmp.rows() - n_pressure_bases - fluid_offset, tmp.cols());
	assert(sol.size() == n_bases * (problem->is_scalar() ? 1 : mesh->dimension()));
	pressure = tmp.block(tmp.rows() - n_pressure_bases - fluid_offset, 0, n_pressure_bases, tmp.cols());
	assert(pressure.size() == n_pressure_bases);
}

void State::compute_mesh_size(const Mesh &mesh_in, const std::vector<ElementBases> &bases_in, const int n_samples)
{
	Eigen::MatrixXd samples_simplex, samples_cube, mapped, p0, p1, p;

	mesh_size = 0;
	average_edge_length = 0;
	min_edge_length = std::numeric_limits<double>::max();

	if (!args["curved_mesh_size"])
	{
		mesh_in.get_edges(p0, p1);
		p = p0 - p1;
		min_edge_length = p.rowwise().norm().minCoeff();
		average_edge_length = p.rowwise().norm().mean();
		mesh_size = p.rowwise().norm().maxCoeff();

		logger().info("hmin: {}", min_edge_length);
		logger().info("hmax: {}", mesh_size);
		logger().info("havg: {}", average_edge_length);

		return;
	}

	if (mesh_in.is_volume())
	{
		EdgeSampler::sample_3d_simplex(n_samples, samples_simplex);
		EdgeSampler::sample_3d_cube(n_samples, samples_cube);
	}
	else
	{
		EdgeSampler::sample_2d_simplex(n_samples, samples_simplex);
		EdgeSampler::sample_2d_cube(n_samples, samples_cube);
	}

	int n = 0;
	for (size_t i = 0; i < bases_in.size(); ++i)
	{
		if (mesh_in.is_polytope(i))
			continue;
		int n_edges;

		if (mesh_in.is_simplex(i))
		{
			n_edges = mesh_in.is_volume() ? 6 : 3;
			bases_in[i].eval_geom_mapping(samples_simplex, mapped);
		}
		else
		{
			n_edges = mesh_in.is_volume() ? 12 : 4;
			bases_in[i].eval_geom_mapping(samples_cube, mapped);
		}

		for (int j = 0; j < n_edges; ++j)
		{
			double current_edge = 0;
			for (int k = 0; k < n_samples - 1; ++k)
			{
				p0 = mapped.row(j * n_samples + k);
				p1 = mapped.row(j * n_samples + k + 1);
				p = p0 - p1;

				current_edge += p.norm();
			}

			mesh_size = std::max(current_edge, mesh_size);
			min_edge_length = std::min(current_edge, min_edge_length);
			average_edge_length += current_edge;
			++n;
		}
	}

	average_edge_length /= n;

	logger().info("hmin: {}", min_edge_length);
	logger().info("hmax: {}", mesh_size);
	logger().info("havg: {}", average_edge_length);
}

void State::save_json()
{
	const std::string out_path = args["output"];
	if (!out_path.empty())
	{
		std::ofstream out(out_path);
		save_json(out);
		out.close();
	}
}

void State::save_json(std::ostream &out)
{
	using json = nlohmann::json;
	json j;
	save_json(j);
	out << j.dump(4) << std::endl;
}

void State::save_json(nlohmann::json &j)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (sol.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}

	logger().info("Saving json...");

	j["args"] = args;
	j["quadrature_order"] = args["quadrature_order"];
	j["mesh_path"] = mesh_path();
	j["discr_order"] = args["discr_order"];
	j["geom_order"] = mesh->orders().size() > 0 ? mesh->orders().maxCoeff() : 1;
	j["geom_order_min"] = mesh->orders().size() > 0 ? mesh->orders().minCoeff() : 1;
	j["discr_order_min"] = disc_orders.minCoeff();
	j["discr_order_max"] = disc_orders.maxCoeff();
	j["harmonic_samples_res"] = args["n_harmonic_samples"];
	j["use_splines"] = args["use_spline"];
	j["iso_parametric"] = iso_parametric();
	j["problem"] = problem->name();
	j["mat_size"] = mat_size;
	j["solver_type"] = args["solver_type"];
	j["precond_type"] = args["precond_type"];
	j["line_search"] = args["line_search"];
	j["nl_solver"] = args["nl_solver"];
	j["params"] = args["params"];

	j["refinenemt_location"] = args["refinenemt_location"];

	j["num_boundary_samples"] = args["n_boundary_samples"];
	j["num_refs"] = args["n_refs"];
	j["num_bases"] = n_bases;
	j["num_pressure_bases"] = n_pressure_bases;
	j["num_non_zero"] = nn_zero;
	j["num_flipped"] = n_flipped;
	j["num_dofs"] = num_dofs;
	j["num_vertices"] = mesh->n_vertices();
	j["num_elements"] = mesh->n_elements();

	j["num_p1"] = (disc_orders.array() == 1).count();
	j["num_p2"] = (disc_orders.array() == 2).count();
	j["num_p3"] = (disc_orders.array() == 3).count();
	j["num_p4"] = (disc_orders.array() == 4).count();
	j["num_p5"] = (disc_orders.array() == 5).count();

	j["mesh_size"] = mesh_size;
	j["max_angle"] = max_angle;

	j["sigma_max"] = sigma_max;
	j["sigma_min"] = sigma_min;
	j["sigma_avg"] = sigma_avg;

	j["min_edge_length"] = min_edge_length;
	j["average_edge_length"] = average_edge_length;

	j["err_l2"] = l2_err;
	j["err_h1"] = h1_err;
	j["err_h1_semi"] = h1_semi_err;
	j["err_linf"] = linf_err;
	j["err_linf_grad"] = grad_max_err;
	j["err_lp"] = lp_err;

	j["spectrum"] = {spectrum(0), spectrum(1), spectrum(2), spectrum(3)};
	j["spectrum_condest"] = std::abs(spectrum(3)) / std::abs(spectrum(0));

	// j["errors"] = errors;

	j["time_building_basis"] = building_basis_time;
	j["time_loading_mesh"] = loading_mesh_time;
	j["time_computing_poly_basis"] = computing_poly_basis_time;
	j["time_assembling_stiffness_mat"] = assembling_stiffness_mat_time;
	j["time_assigning_rhs"] = assigning_rhs_time;
	j["time_solving"] = solving_time;
	j["time_computing_errors"] = computing_errors_time;

	j["solver_info"] = solver_info;

	j["count_simplex"] = simplex_count;
	j["count_regular"] = regular_count;
	j["count_regular_boundary"] = regular_boundary_count;
	j["count_simple_singular"] = simple_singular_count;
	j["count_multi_singular"] = multi_singular_count;
	j["count_boundary"] = boundary_count;
	j["count_non_regular_boundary"] = non_regular_boundary_count;
	j["count_non_regular"] = non_regular_count;
	j["count_undefined"] = undefined_count;
	j["count_multi_singular_boundary"] = multi_singular_boundary_count;

	j["is_simplicial"] = mesh->n_elements() == simplex_count;

	j["peak_memory"] = getPeakRSS() / (1024 * 1024);

	const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();

	std::vector<double> mmin(actual_dim);
	std::vector<double> mmax(actual_dim);

	for (int d = 0; d < actual_dim; ++d)
	{
		mmin[d] = std::numeric_limits<double>::max();
		mmax[d] = -std::numeric_limits<double>::max();
	}

	for (int i = 0; i < sol.size(); i += actual_dim)
	{
		for (int d = 0; d < actual_dim; ++d)
		{
			mmin[d] = std::min(mmin[d], sol(i + d));
			mmax[d] = std::max(mmax[d], sol(i + d));
		}
	}

	std::vector<double> sol_at_node(actual_dim);

	if (args["export"]["sol_at_node"] >= 0)
	{
		const int node_id = args["export"]["sol_at_node"];

		for (int d = 0; d < actual_dim; ++d)
		{
			sol_at_node[d] = sol(node_id * actual_dim + d);
		}
	}

	j["sol_at_node"] = sol_at_node;
	j["sol_min"] = mmin;
	j["sol_max"] = mmax;

#ifdef POLYFEM_WITH_TBB
	j["num_threads"] = tbb::task_scheduler_init::default_num_threads();
#else
	j["num_threads"] = 1;
#endif

	j["formulation"] = formulation();

	logger().info("done");
}

double get_opt_p(bool h1_formula, double B,
				 double h_ref, int p_ref, double rho_ref,
				 double h, double rho, int p_max)
{
	const double sigma_ref = rho_ref / h_ref;
	const double sigma = rho / h;

	const double ptmp = h1_formula ? (std::log(B * std::pow(h_ref, p_ref + 1) * rho / (h * rho_ref)) / std::log(h)) : (std::log(B * std::pow(h_ref, p_ref + 1) * sigma * sigma / sigma_ref / sigma_ref) - std::log(h)) / std::log(h);
	// (std::log(B*std::pow(h_ref, p_ref + 2)*rho*rho / (h * h *rho_ref*rho_ref))/std::log(h));

	return std::min(std::max(p_ref, (int)std::round(ptmp)), p_max);
}

void State::p_refinement(const Mesh2D &mesh2d)
{
	max_angle = 0;
	// static const int max_angles = 5;
	// static const double angles[max_angles] = {0, 170./180.*M_PI, 179./180.*M_PI, 179.9/180.* M_PI, M_PI};

	Eigen::MatrixXd p0, p1;
	mesh2d.get_edges(p0, p1);
	const auto tmp = p0 - p1;
	const double h_ref = tmp.rowwise().norm().mean();
	const double B = args["B"];
	const bool h1_formula = args["h1_formula"];
	const int p_ref = args["discr_order"];
	const double rho_ref = sqrt(3.0) / 6.0 * h_ref;
	const int p_max = std::min(autogen::MAX_P_BASES, args["discr_order_max"].get<int>());

	sigma_avg = 0;
	sigma_max = 0;
	sigma_min = std::numeric_limits<double>::max();

	for (int f = 0; f < mesh2d.n_faces(); ++f)
	{
		if (!mesh2d.is_simplex(f))
			continue;

		auto v0 = mesh2d.point(mesh2d.face_vertex(f, 0));
		auto v1 = mesh2d.point(mesh2d.face_vertex(f, 1));
		auto v2 = mesh2d.point(mesh2d.face_vertex(f, 2));

		const RowVectorNd e0 = v1 - v0;
		const RowVectorNd e1 = v2 - v1;
		const RowVectorNd e2 = v0 - v2;

		const double e0n = e0.norm();
		const double e1n = e1.norm();
		const double e2n = e2.norm();

		const double alpha0 = angle2(e0, -e2);
		const double alpha1 = angle2(e1, -e0);
		const double alpha2 = angle2(e2, -e1);

		const double P = e0n + e1n + e2n;
		const double A = std::abs(e1(0) * e2(1) - e1(1) * e2(0)) / 2;
		const double rho = 2 * A / P;
		const double hp = std::max(e0n, std::max(e1n, e2n));
		const double sigma = rho / hp;

		sigma_avg += sigma;
		sigma_max = std::max(sigma_max, sigma);
		sigma_min = std::min(sigma_min, sigma);

		const int p = get_opt_p(h1_formula, B, h_ref, p_ref, rho_ref, hp, rho, p_max);

		if (p > disc_orders[f])
			disc_orders[f] = p;
		auto index = mesh2d.get_index_from_face(f);

		for (int lv = 0; lv < 3; ++lv)
		{
			auto nav = mesh2d.switch_face(index);

			if (nav.face >= 0)
			{
				if (p > disc_orders[nav.face])
					disc_orders[nav.face] = p;
			}

			index = mesh2d.next_around_face(index);
		}

		max_angle = std::max(max_angle, alpha0);
		max_angle = std::max(max_angle, alpha1);
		max_angle = std::max(max_angle, alpha2);
	}

	sigma_avg /= mesh2d.n_faces();
	max_angle = max_angle / M_PI * 180.;
	logger().info("using B={} with {} estimate max_angle {}", B, (h1_formula ? "H1" : "L2"), max_angle);
	logger().info("average sigma: {}", sigma_avg);
	logger().info("min sigma: {}", sigma_min);
	logger().info("max sigma: {}", sigma_max);

	logger().info("num_p1 {}", (disc_orders.array() == 1).count());
	logger().info("num_p2 {}", (disc_orders.array() == 2).count());
	logger().info("num_p3 {}", (disc_orders.array() == 3).count());
	logger().info("num_p4 {}", (disc_orders.array() == 4).count());
	logger().info("num_p5 {}", (disc_orders.array() == 5).count());
}

void State::p_refinement(const Mesh3D &mesh3d)
{
	max_angle = 0;

	Eigen::MatrixXd p0, p1;
	mesh3d.get_edges(p0, p1);
	const auto tmp = p0 - p1;
	const double h_ref = tmp.rowwise().norm().mean();
	const double B = args["B"];
	const bool h1_formula = args["h1_formula"];
	const int p_ref = args["discr_order"];
	const double rho_ref = sqrt(6.) / 12. * h_ref;
	const int p_max = std::min(autogen::MAX_P_BASES, args["discr_order_max"].get<int>());

	sigma_avg = 0;
	sigma_max = 0;
	sigma_min = std::numeric_limits<double>::max();

	for (int c = 0; c < mesh3d.n_cells(); ++c)
	{
		if (!mesh3d.is_simplex(c))
			continue;

		const auto v0 = mesh3d.point(mesh3d.cell_vertex(c, 0));
		const auto v1 = mesh3d.point(mesh3d.cell_vertex(c, 1));
		const auto v2 = mesh3d.point(mesh3d.cell_vertex(c, 2));
		const auto v3 = mesh3d.point(mesh3d.cell_vertex(c, 3));

		Eigen::Matrix<double, 6, 3> e;
		e.row(0) = v0 - v1;
		e.row(1) = v1 - v2;
		e.row(2) = v2 - v0;

		e.row(3) = v0 - v3;
		e.row(4) = v1 - v3;
		e.row(5) = v2 - v3;

		Eigen::Matrix<double, 6, 1> en = e.rowwise().norm();

		Eigen::Matrix<double, 3 * 4, 1> alpha;
		alpha(0) = angle3(e.row(0), -e.row(1));
		alpha(1) = angle3(e.row(1), -e.row(2));
		alpha(2) = angle3(e.row(2), -e.row(0));
		alpha(3) = angle3(e.row(0), -e.row(4));
		alpha(4) = angle3(e.row(4), e.row(3));
		alpha(5) = angle3(-e.row(3), -e.row(0));
		alpha(6) = angle3(-e.row(4), -e.row(1));
		alpha(7) = angle3(e.row(1), -e.row(5));
		alpha(8) = angle3(e.row(5), e.row(4));
		alpha(9) = angle3(-e.row(2), -e.row(5));
		alpha(10) = angle3(e.row(5), e.row(3));
		alpha(11) = angle3(-e.row(3), e.row(2));

		const double S = (e.row(0).cross(e.row(1)).norm() + e.row(0).cross(e.row(4)).norm() + e.row(4).cross(e.row(1)).norm() + e.row(2).cross(e.row(5)).norm()) / 2;
		const double V = std::abs(e.row(3).dot(e.row(2).cross(-e.row(0)))) / 6;
		const double rho = 3 * V / S;
		const double hp = en.maxCoeff();

		sigma_avg += rho / hp;
		sigma_max = std::max(sigma_max, rho / hp);
		sigma_min = std::min(sigma_min, rho / hp);

		const int p = get_opt_p(h1_formula, B, h_ref, p_ref, rho_ref, hp, rho, p_max);

		if (p > disc_orders[c])
			disc_orders[c] = p;

		for (int le = 0; le < 6; ++le)
		{
			const int e_id = mesh3d.cell_edge(c, le);
			const auto cells = mesh3d.edge_neighs(e_id);

			for (auto c_id : cells)
			{
				if (p > disc_orders[c_id])
					disc_orders[c_id] = p;
			}
		}

		max_angle = std::max(max_angle, alpha.maxCoeff());
	}

	max_angle = max_angle / M_PI * 180.;
	sigma_avg /= mesh3d.n_elements();

	logger().info("using B={} with {} estimate max_angle {}", B, (h1_formula ? "H1" : "L2"), max_angle);
	logger().info("average sigma: {}", sigma_avg);
	logger().info("min sigma: {}", sigma_min);
	logger().info("max sigma: {}", sigma_max);

	logger().info("num_p1 {}", (disc_orders.array() == 1).count());
	logger().info("num_p2 {}", (disc_orders.array() == 2).count());
	logger().info("num_p3 {}", (disc_orders.array() == 3).count());
	logger().info("num_p4 {}", (disc_orders.array() == 4).count());
	logger().info("num_p5 {}", (disc_orders.array() == 5).count());
}

void State::interpolate_boundary_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (fun.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}
	assert(mesh->is_volume());

	const Mesh3D &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());

	Eigen::MatrixXd points, uv;
	Eigen::VectorXd weights;

	int actual_dim = 1;
	if (!problem->is_scalar())
		actual_dim = 3;

	igl::AABB<Eigen::MatrixXd, 3> tree;
	tree.init(pts, faces);

	const auto &gbases = iso_parametric() ? bases : geom_bases;
	result.resize(faces.rows(), actual_dim);
	result.setConstant(std::numeric_limits<double>::quiet_NaN());

	int counter = 0;

	for (int e = 0; e < mesh3d.n_elements(); ++e)
	{
		const ElementBases &gbs = gbases[e];
		const ElementBases &bs = bases[e];

		for (int lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
		{
			const int face_id = mesh3d.cell_face(e, lf);
			if (!mesh3d.is_boundary_face(face_id))
				continue;

			if (mesh3d.is_simplex(e))
				BoundarySampler::quadrature_for_tri_face(lf, 4, face_id, mesh3d, uv, points, weights);
			else if (mesh3d.is_cube(e))
				BoundarySampler::quadrature_for_quad_face(lf, 4, face_id, mesh3d, uv, points, weights);
			else
				assert(false);

			ElementAssemblyValues vals;
			vals.compute(e, true, points, bs, gbs);
			RowVectorNd loc_val(actual_dim);
			loc_val.setZero();

			// UIState::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(1,0,0));

			// const auto nodes = bs.local_nodes_for_primitive(face_id, mesh3d);

			// for(long n = 0; n < nodes.size(); ++n)
			for (size_t j = 0; j < bs.bases.size(); ++j)
			{
				// const auto &b = bs.bases[nodes(n)];
				// const AssemblyValues &v = vals.basis_values[nodes(n)];
				const AssemblyValues &v = vals.basis_values[j];
				for (int d = 0; d < actual_dim; ++d)
				{
					for (size_t g = 0; g < v.global.size(); ++g)
					{
						loc_val(d) += (v.global[g].val * v.val.array() * fun(v.global[g].index * actual_dim + d) * weights.array()).sum();
					}
				}
			}

			int I;
			Eigen::RowVector3d C;
			const Eigen::RowVector3d bary = mesh3d.face_barycenter(face_id);

			const double dist = tree.squared_distance(pts, faces, bary, I, C);
			assert(dist < 1e-16);

			assert(std::isnan(result(I, 0)));
			if (compute_avg)
				result.row(I) = loc_val / weights.sum();
			else
				result.row(I) = loc_val;
			++counter;
		}
	}

	assert(counter == result.rows());
}

void State::interpolate_boundary_function_at_vertices(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, MatrixXd &result)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (fun.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}
	if (!mesh->is_volume())
	{
		logger().error("This function works only on volumetric meshes!");
		return;
	}

	assert(mesh->is_volume());

	const Mesh3D &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());

	Eigen::MatrixXd points;

	int actual_dim = 1;
	if (!problem->is_scalar())
		actual_dim = 3;

	igl::AABB<Eigen::MatrixXd, 3> tree;
	tree.init(pts, faces);

	const auto &gbases = iso_parametric() ? bases : geom_bases;
	result.resize(pts.rows(), actual_dim);
	result.setZero();

	for (int e = 0; e < mesh3d.n_elements(); ++e)
	{
		const ElementBases &gbs = gbases[e];
		const ElementBases &bs = bases[e];

		for (int lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
		{
			const int face_id = mesh3d.cell_face(e, lf);
			if (!mesh3d.is_boundary_face(face_id))
				continue;

			if (mesh3d.is_simplex(e))
				autogen::p_nodes_3d(1, points);
			else if (mesh3d.is_cube(e))
				autogen::q_nodes_3d(1, points);
			else
				assert(false);

			ElementAssemblyValues vals;
			vals.compute(e, true, points, bs, gbs);
			MatrixXd loc_val(points.rows(), actual_dim);
			loc_val.setZero();

			// UIState::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(1,0,0));

			for (size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				const AssemblyValues &v = vals.basis_values[j];

				for (int d = 0; d < actual_dim; ++d)
				{
					for (size_t ii = 0; ii < b.global().size(); ++ii)
						loc_val.col(d) += b.global()[ii].val * v.val * fun(b.global()[ii].index * actual_dim + d);
				}
			}

			int I;
			Eigen::RowVector3d C;
			const Eigen::RowVector3d bary = mesh3d.face_barycenter(face_id);

			const double dist = tree.squared_distance(pts, faces, bary, I, C);
			assert(dist < 1e-16);

			for (int lv_id = 0; lv_id < faces.cols(); ++lv_id)
			{
				const int v_id = faces(I, lv_id);
				const auto p = pts.row(v_id);
				const auto &mapped = vals.val;

				bool found = false;

				for (int n = 0; n < mapped.rows(); ++n)
				{
					if ((p - mapped.row(n)).norm() < 1e-10)
					{
						result.row(v_id) = loc_val.row(n);
						found = true;
						break;
					}
				}

				assert(found);
			}
		}
	}
}

void State::interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result)
{
	interpolate_boundary_tensor_function(pts, faces, fun, Eigen::MatrixXd::Zero(pts.rows(), pts.cols()), compute_avg, result);
}

void State::interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const MatrixXd &disp, const bool compute_avg, MatrixXd &result)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (fun.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}
	if (disp.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}
	if (!mesh->is_volume())
	{
		logger().error("This function works only on volumetric meshes!");
		return;
	}
	if (problem->is_scalar())
	{
		logger().error("Define a tensor problem!");
		return;
	}

	assert(mesh->is_volume());
	assert(!problem->is_scalar());

	const Mesh3D &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());

	MatrixXd normals;
	igl::per_face_normals((pts + disp).eval(), faces, normals);
	// std::cout<<normals<<std::endl;

	Eigen::MatrixXd points, uv;
	Eigen::VectorXd weights;

	const int actual_dim = 3;

	igl::AABB<Eigen::MatrixXd, 3> tree;
	tree.init(pts, faces);

	const auto &gbases = iso_parametric() ? bases : geom_bases;
	result.resize(faces.rows(), actual_dim);
	result.setConstant(std::numeric_limits<double>::quiet_NaN());

	int counter = 0;

	const auto &assembler = AssemblerUtils::instance();

	for (int e = 0; e < mesh3d.n_elements(); ++e)
	{
		const ElementBases &gbs = gbases[e];
		const ElementBases &bs = bases[e];

		for (int lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
		{
			const int face_id = mesh3d.cell_face(e, lf);
			if (!mesh3d.is_boundary_face(face_id))
				continue;

			if (mesh->is_simplex(e))
				BoundarySampler::quadrature_for_tri_face(lf, 4, face_id, mesh3d, uv, points, weights);
			else if (mesh->is_cube(e))
				BoundarySampler::quadrature_for_quad_face(lf, 4, face_id, mesh3d, uv, points, weights);
			else
				assert(false);

			// ElementAssemblyValues vals;
			// vals.compute(e, true, points, bs, gbs);
			Eigen::MatrixXd loc_val;

			// UIState::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(1,0,0));

			// const auto nodes = bs.local_nodes_for_primitive(face_id, mesh3d);
			assembler.compute_tensor_value(formulation(), e, bs, gbs, points, fun, loc_val);
			Eigen::VectorXd tmp(loc_val.cols());
			for (int d = 0; d < loc_val.cols(); ++d)
				tmp(d) = (loc_val.col(d).array() * weights.array()).sum();
			const Eigen::MatrixXd tensor = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 3, 3);

			int I;
			Eigen::RowVector3d C;
			const Eigen::RowVector3d bary = mesh3d.face_barycenter(face_id);

			const double dist = tree.squared_distance(pts, faces, bary, I, C);
			assert(dist < 1e-16);

			assert(std::isnan(result(I, 0)));
			result.row(I) = normals.row(I) * tensor;
			if (compute_avg)
				result.row(I) /= weights.sum();
			++counter;
		}
	}

	assert(counter == result.rows());
}

void State::average_grad_based_function(const int n_points, const MatrixXd &fun, MatrixXd &result_scalar, MatrixXd &result_tensor, const bool boundary_only)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (fun.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}
	if (problem->is_scalar())
	{
		logger().error("Define a tensor problem!");
		return;
	}

	assert(!problem->is_scalar());
	const int actual_dim = mesh->dimension();

	MatrixXd avg_scalar(n_bases, 1);
	// MatrixXd avg_tensor(n_points * actual_dim*actual_dim, 1);
	MatrixXd areas(n_bases, 1);
	avg_scalar.setZero();
	// avg_tensor.setZero();
	areas.setZero();

	const auto &assembler = AssemblerUtils::instance();

	Eigen::MatrixXd local_val;
	const auto &gbases = iso_parametric() ? bases : geom_bases;

	ElementAssemblyValues vals;
	for (int i = 0; i < int(bases.size()); ++i)
	{
		const ElementBases &bs = bases[i];
		const ElementBases &gbs = gbases[i];
		Eigen::MatrixXd local_pts;

		if (mesh->is_simplex(i))
		{
			if (mesh->dimension() == 3)
				autogen::p_nodes_3d(bs.bases.front().order(), local_pts);
			else
				autogen::p_nodes_2d(bs.bases.front().order(), local_pts);
		}
		else
		{
			if (mesh->dimension() == 3)
				autogen::q_nodes_3d(bs.bases.front().order(), local_pts);
			else
				autogen::q_nodes_2d(bs.bases.front().order(), local_pts);
		}
		// else if(mesh->is_cube(i))
		// 	local_pts = sampler.cube_points();
		// // else
		// 	// local_pts = vis_pts_poly[i];

		vals.compute(i, actual_dim == 3, bases[i], gbases[i]);
		const Quadrature &quadrature = vals.quadrature;
		const double area = (vals.det.array() * quadrature.weights.array()).sum();

		assembler.compute_scalar_value(formulation(), i, bs, gbs, local_pts, fun, local_val);
		// assembler.compute_tensor_value(formulation(), i, bs, gbs, local_pts, fun, local_val);

		for (size_t j = 0; j < bs.bases.size(); ++j)
		{
			const Basis &b = bs.bases[j];
			if (b.global().size() > 1)
				continue;

			auto &global = b.global().front();
			areas(global.index) += area;
			avg_scalar(global.index) += local_val(j) * area;
		}
	}

	avg_scalar.array() /= areas.array();

	interpolate_function(n_points, 1, bases, avg_scalar, result_scalar, boundary_only);
	// interpolate_function(n_points, actual_dim*actual_dim, bases, avg_tensor, result_tensor, boundary_only);
}

void State::compute_vertex_values(int actual_dim,
								  const std::vector<ElementBases> &basis,
								  const MatrixXd &fun,
								  Eigen::MatrixXd &result)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (fun.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}
	if (!mesh->is_volume())
	{
		logger().error("This function works only on volumetric meshes!");
		return;
	}

	if (!mesh)
	{
		return;
	}
	if (!mesh->is_volume())
	{
		return;
	}
	const Mesh3D &mesh3d = *dynamic_cast<const Mesh3D *>(mesh.get());

	result.resize(mesh3d.n_vertices(), actual_dim);
	result.setZero();

	// std::array<int, 8> get_ordered_vertices_from_hex(const int element_index) const;
	// std::array<int, 4> get_ordered_vertices_from_tet(const int element_index) const;

	const auto &sampler = RefElementSampler::sampler();
	std::vector<AssemblyValues> tmp;
	std::vector<bool> marked(mesh3d.n_vertices(), false);
	for (int i = 0; i < int(basis.size()); ++i)
	{
		const ElementBases &bs = basis[i];
		MatrixXd local_pts;
		std::vector<int> vertices;

		if (mesh->is_simplex(i))
		{
			local_pts = sampler.simplex_corners();
			auto vtx = mesh3d.get_ordered_vertices_from_tet(i);
			vertices.assign(vtx.begin(), vtx.end());
		}
		else if (mesh->is_cube(i))
		{
			local_pts = sampler.cube_corners();
			auto vtx = mesh3d.get_ordered_vertices_from_hex(i);
			vertices.assign(vtx.begin(), vtx.end());
		}
		//TODO poly?
		assert((int)vertices.size() == (int)local_pts.rows());

		MatrixXd local_res = MatrixXd::Zero(local_pts.rows(), actual_dim);
		bs.evaluate_bases(local_pts, tmp);
		for (size_t j = 0; j < bs.bases.size(); ++j)
		{
			const Basis &b = bs.bases[j];

			for (int d = 0; d < actual_dim; ++d)
			{
				for (size_t ii = 0; ii < b.global().size(); ++ii)
					local_res.col(d) += b.global()[ii].val * tmp[j].val * fun(b.global()[ii].index * actual_dim + d);
			}
		}

		for (size_t lv = 0; lv < vertices.size(); ++lv)
		{
			int v = vertices[lv];
			if (marked[v])
			{
				assert((result.row(v) - local_res.row(lv)).norm() < 1e-6);
			}
			else
			{
				result.row(v) = local_res.row(lv);
				marked[v] = true;
			}
		}
	}
}

void flattened_tensor_coeffs(const MatrixXd &S, MatrixXd &X)
{
	if (S.cols() == 4)
	{
		X.resize(S.rows(), 3);
		X.col(0) = S.col(0);
		X.col(1) = S.col(3);
		X.col(2) = S.col(1);
	}
	else if (S.cols() == 9)
	{
		// [S11, S22, S33, S12, S13, S23]
		X.resize(S.rows(), 6);
		X.col(0) = S.col(0);
		X.col(1) = S.col(4);
		X.col(2) = S.col(8);
		X.col(3) = S.col(1);
		X.col(4) = S.col(2);
		X.col(5) = S.col(5);
	}
	else
	{
		logger().error("Invalid tensor dimensions.");
	}
}

void State::compute_stress_at_quadrature_points(const MatrixXd &fun, Eigen::MatrixXd &result, Eigen::VectorXd &von_mises)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (fun.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}
	if (problem->is_scalar())
	{
		logger().error("Define a tensor problem!");
		return;
	}

	const int actual_dim = mesh->dimension();
	assert(!problem->is_scalar());

	const auto &assembler = AssemblerUtils::instance();

	Eigen::MatrixXd local_val, local_stress, local_mises;
	const auto &gbases = iso_parametric() ? bases : geom_bases;

	int num_quadr_pts = 0;
	result.resize(disc_orders.sum(), actual_dim == 2 ? 3 : 6);
	result.setZero();
	von_mises.resize(disc_orders.sum(), 1);
	von_mises.setZero();
	for (int e = 0; e < mesh->n_elements(); ++e)
	{
		// Compute quadrature points for element
		Quadrature quadr;
		if (mesh->is_simplex(e))
		{
			if (mesh->is_volume())
			{
				TetQuadrature f;
				f.get_quadrature(disc_orders(e), quadr);
			}
			else
			{
				TriQuadrature f;
				f.get_quadrature(disc_orders(e), quadr);
			}
		}
		else if (mesh->is_cube(e))
		{
			if (mesh->is_volume())
			{
				HexQuadrature f;
				f.get_quadrature(disc_orders(e), quadr);
			}
			else
			{
				QuadQuadrature f;
				f.get_quadrature(disc_orders(e), quadr);
			}
		}
		else
		{
			continue;
		}

		assembler.compute_scalar_value(formulation(), e, bases[e], gbases[e],
									   quadr.points, fun, local_mises);
		assembler.compute_tensor_value(formulation(), e, bases[e], gbases[e],
									   quadr.points, fun, local_val);

		if (num_quadr_pts + local_val.rows() >= result.rows())
		{
			result.conservativeResize(
				std::max(num_quadr_pts + local_val.rows() + 1, 2 * result.rows()),
				result.cols());
			von_mises.conservativeResize(result.rows(), von_mises.cols());
		}
		flattened_tensor_coeffs(local_val, local_stress);
		result.block(num_quadr_pts, 0, local_stress.rows(), local_stress.cols()) = local_stress;
		von_mises.block(num_quadr_pts, 0, local_mises.rows(), local_mises.cols()) = local_mises;
		num_quadr_pts += local_val.rows();
	}
	result.conservativeResize(num_quadr_pts, result.cols());
	von_mises.conservativeResize(num_quadr_pts, von_mises.cols());
}

void State::interpolate_function(const int n_points, const MatrixXd &fun, MatrixXd &result, const bool boundary_only)
{
	int actual_dim = 1;
	if (!problem->is_scalar())
		actual_dim = mesh->dimension();
	interpolate_function(n_points, actual_dim, bases, fun, result, boundary_only);
}

void State::interpolate_function(const int n_points, const int actual_dim, const std::vector<ElementBases> &basis, const MatrixXd &fun, MatrixXd &result, const bool boundary_only)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (fun.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}

	std::vector<AssemblyValues> tmp;

	result.resize(n_points, actual_dim);

	int index = 0;
	const auto &sampler = RefElementSampler::sampler();

	Eigen::MatrixXi vis_faces_poly;

	for (int i = 0; i < int(basis.size()); ++i)
	{
		const ElementBases &bs = basis[i];
		MatrixXd local_pts;

		if (boundary_only && mesh->is_volume() && !mesh->is_boundary_element(i))
			continue;

		if (mesh->is_simplex(i))
			local_pts = sampler.simplex_points();
		else if (mesh->is_cube(i))
			local_pts = sampler.cube_points();
		else
		{
			if (mesh->is_volume())
				sampler.sample_polyhedron(polys_3d[i].first, polys_3d[i].second, local_pts, vis_faces_poly);
			else
				sampler.sample_polygon(polys[i], local_pts, vis_faces_poly);
		}

		MatrixXd local_res = MatrixXd::Zero(local_pts.rows(), actual_dim);
		bs.evaluate_bases(local_pts, tmp);
		for (size_t j = 0; j < bs.bases.size(); ++j)
		{
			const Basis &b = bs.bases[j];

			for (int d = 0; d < actual_dim; ++d)
			{
				for (size_t ii = 0; ii < b.global().size(); ++ii)
					local_res.col(d) += b.global()[ii].val * tmp[j].val * fun(b.global()[ii].index * actual_dim + d);
			}
		}

		result.block(index, 0, local_res.rows(), actual_dim) = local_res;
		index += local_res.rows();
	}
}

void State::compute_scalar_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool boundary_only)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (fun.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}

	result.resize(n_points, 1);
	assert(!problem->is_scalar());

	int index = 0;
	const auto &sampler = RefElementSampler::sampler();
	const auto &assembler = AssemblerUtils::instance();

	Eigen::MatrixXi vis_faces_poly;
	Eigen::MatrixXd local_val;
	const auto &gbases = iso_parametric() ? bases : geom_bases;

	for (int i = 0; i < int(bases.size()); ++i)
	{
		if (boundary_only && mesh->is_volume() && !mesh->is_boundary_element(i))
			continue;

		const ElementBases &bs = bases[i];
		const ElementBases &gbs = gbases[i];
		Eigen::MatrixXd local_pts;

		if (mesh->is_simplex(i))
			local_pts = sampler.simplex_points();
		else if (mesh->is_cube(i))
			local_pts = sampler.cube_points();
		else
		{
			if (mesh->is_volume())
				sampler.sample_polyhedron(polys_3d[i].first, polys_3d[i].second, local_pts, vis_faces_poly);
			else
				sampler.sample_polygon(polys[i], local_pts, vis_faces_poly);
		}

		assembler.compute_scalar_value(formulation(), i, bs, gbs, local_pts, fun, local_val);

		result.block(index, 0, local_val.rows(), 1) = local_val;
		index += local_val.rows();
	}
}

void State::compute_tensor_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool boundary_only)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (fun.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}

	const int actual_dim = mesh->dimension();
	result.resize(n_points, actual_dim * actual_dim);
	assert(!problem->is_scalar());

	int index = 0;
	const auto &sampler = RefElementSampler::sampler();
	const auto &assembler = AssemblerUtils::instance();

	Eigen::MatrixXi vis_faces_poly;
	Eigen::MatrixXd local_val;
	const auto &gbases = iso_parametric() ? bases : geom_bases;

	for (int i = 0; i < int(bases.size()); ++i)
	{
		if (boundary_only && mesh->is_volume() && !mesh->is_boundary_element(i))
			continue;

		const ElementBases &bs = bases[i];
		const ElementBases &gbs = gbases[i];
		Eigen::MatrixXd local_pts;

		if (mesh->is_simplex(i))
			local_pts = sampler.simplex_points();
		else if (mesh->is_cube(i))
			local_pts = sampler.cube_points();
		else
		{
			if (mesh->is_volume())
				sampler.sample_polyhedron(polys_3d[i].first, polys_3d[i].second, local_pts, vis_faces_poly);
			else
				sampler.sample_polygon(polys[i], local_pts, vis_faces_poly);
		}

		assembler.compute_tensor_value(formulation(), i, bs, gbs, local_pts, fun, local_val);

		result.block(index, 0, local_val.rows(), local_val.cols()) = local_val;
		index += local_val.rows();
	}
}

void State::get_sidesets(Eigen::MatrixXd &pts, Eigen::MatrixXi &faces, Eigen::MatrixXd &sidesets)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}

	if (mesh->is_volume())
	{
		const Mesh3D &tmp_mesh = *dynamic_cast<Mesh3D *>(mesh.get());
		int n_pts = 0;
		int n_faces = 0;
		for (int f = 0; f < tmp_mesh.n_faces(); ++f)
		{
			if (tmp_mesh.get_boundary_id(f) > 0)
			{
				n_pts += tmp_mesh.n_face_vertices(f) + 1;
				n_faces += tmp_mesh.n_face_vertices(f);
			}
		}

		pts.resize(n_pts, 3);
		faces.resize(n_faces, 3);
		sidesets.resize(n_pts, 1);

		n_pts = 0;
		n_faces = 0;
		for (int f = 0; f < tmp_mesh.n_faces(); ++f)
		{
			const int sideset = tmp_mesh.get_boundary_id(f);
			if (sideset > 0)
			{
				const int n_face_vertices = tmp_mesh.n_face_vertices(f);

				for (int i = 0; i < n_face_vertices; ++i)
				{
					if (n_face_vertices == 3)
						faces.row(n_faces) << ((i + 1) % n_face_vertices + n_pts), (i + n_pts), (n_pts + n_face_vertices);
					else
						faces.row(n_faces) << (i + n_pts), ((i + 1) % n_face_vertices + n_pts), (n_pts + n_face_vertices);
					++n_faces;
				}

				for (int i = 0; i < n_face_vertices; ++i)
				{
					pts.row(n_pts) = tmp_mesh.point(tmp_mesh.face_vertex(f, i));
					sidesets(n_pts) = sideset;

					++n_pts;
				}

				pts.row(n_pts) = tmp_mesh.face_barycenter(f);
				sidesets(n_pts) = sideset;
				++n_pts;
			}
		}
	}
	else
	{
		const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
		int n_siteset = 0;
		for (int e = 0; e < tmp_mesh.n_edges(); ++e)
		{
			if (tmp_mesh.get_boundary_id(e) > 0)
				++n_siteset;
		}

		pts.resize(n_siteset * 2, 2);
		faces.resize(n_siteset, 2);
		sidesets.resize(n_siteset, 1);

		n_siteset = 0;
		for (int e = 0; e < tmp_mesh.n_edges(); ++e)
		{
			const int sideset = tmp_mesh.get_boundary_id(e);
			if (sideset > 0)
			{
				pts.row(2 * n_siteset) = tmp_mesh.point(tmp_mesh.edge_vertex(e, 0));
				pts.row(2 * n_siteset + 1) = tmp_mesh.point(tmp_mesh.edge_vertex(e, 1));
				faces.row(n_siteset) << 2 * n_siteset, 2 * n_siteset + 1;
				sidesets(n_siteset) = sideset;
				++n_siteset;
			}
		}

		pts.conservativeResize(n_siteset * 2, 3);
		pts.col(2).setZero();
	}
}

void State::load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd &)> &boundary_marker, bool skip_boundary_sideset)
{
	bases.clear();
	pressure_bases.clear();
	geom_bases.clear();
	boundary_nodes.clear();
	local_boundary.clear();
	local_neumann_boundary.clear();
	polys.clear();
	poly_edge_to_data.clear();
	parent_elements.clear();

	stiffness.resize(0, 0);
	rhs.resize(0, 0);
	sol.resize(0, 0);
	pressure.resize(0, 0);

	n_bases = 0;
	n_pressure_bases = 0;

	igl::Timer timer;
	timer.start();
	logger().info("Loading mesh...");
	mesh = Mesh::create(meshin);
	if (!mesh)
	{
		logger().error("Unable to load the mesh");
		return;
	}

	if (args["normalize_mesh"])
	{
		mesh->normalize();
	}
	RowVectorNd min, max;
	mesh->bounding_box(min, max);

	if (min.size() == 2)
		logger().info("mesh bb min [{} {}], max [{} {}]", min(0), min(1), max(0), max(1));
	else
		logger().info("mesh bb min [{} {} {}], max [{} {} {}]", min(0), min(1), min(2), max(0), max(1), max(2));

	int n_refs = args["n_refs"];

	if (n_refs <= 0 && args["poly_bases"] == "MFSHarmonic" && mesh->has_poly()){
		 if(args["force_no_ref_for_harmonic"])
		 	logger().warn("Using harmonic bases without refinement");
		else
			n_refs = 1;
	}

	if (n_refs > 0)
		mesh->refine(n_refs, args["refinenemt_location"], parent_elements);

	if (!skip_boundary_sideset)
		mesh->compute_boundary_ids(boundary_marker);

	timer.stop();
	logger().info(" took {}s", timer.getElapsedTime());

	RefElementSampler::sampler().init(mesh->is_volume(), mesh->n_elements(), args["vismesh_rel_area"]);
}

void State::load_mesh()
{
	bases.clear();
	pressure_bases.clear();
	geom_bases.clear();
	boundary_nodes.clear();
	local_boundary.clear();
	local_neumann_boundary.clear();
	polys.clear();
	poly_edge_to_data.clear();
	parent_elements.clear();

	stiffness.resize(0, 0);
	rhs.resize(0, 0);
	sol.resize(0, 0);
	pressure.resize(0, 0);

	n_bases = 0;
	n_pressure_bases = 0;

	igl::Timer timer;
	timer.start();
	logger().info("Loading mesh...");

	if (!mesh || !mesh_path().empty())
	{
		mesh = Mesh::create(mesh_path());
	}
	if (!mesh)
	{
		logger().error("unable to load the mesh!");
		return;
	}

	// if(!flipped_elements.empty())
	// {
	// 	mesh->compute_elements_tag();
	// 	for(auto el_id : flipped_elements)
	// 		mesh->set_tag(el_id, ElementType::InteriorPolytope);
	// }

	if (args["normalize_mesh"])
		mesh->normalize();

	RowVectorNd min, max;
	mesh->bounding_box(min, max);

	if (min.size() == 2)
		logger().info("mesh bb min [{}, {}], max [{}, {}]", min(0), min(1), max(0), max(1));
	else
		logger().info("mesh bb min [{}, {}, {}], max [{}, {}, {}]", min(0), min(1), min(2), max(0), max(1), max(2));

	int n_refs = args["n_refs"];

	if (n_refs <= 0 && args["poly_bases"] == "MFSHarmonic" && mesh->has_poly())
	{
		if (args["force_no_ref_for_harmonic"])
			logger().warn("Using harmonic bases without refinement");
		else
			n_refs = 1;
	}

	if (n_refs > 0)
		mesh->refine(n_refs, args["refinenemt_location"], parent_elements);

	// mesh->set_tag(1712, ElementType::InteriorPolytope);

	const std::string bc_tag_path = args["bc_tag"];

	double boundary_id_threshold = args["boundary_id_threshold"];
	if (boundary_id_threshold <= 0)
		boundary_id_threshold = mesh->is_volume() ? 1e-2 : 1e-7;

	if (!mesh->has_boundary_ids())
	{
		if (bc_tag_path.empty())
			mesh->compute_boundary_ids(boundary_id_threshold);
		else
			mesh->load_boundary_ids(bc_tag_path);
	}

	timer.stop();
	logger().info(" took {}s", timer.getElapsedTime());

	RefElementSampler::sampler().init(mesh->is_volume(), mesh->n_elements(), args["vismesh_rel_area"]);

	// const double poly_percentage = 0.05;
	// const double poly_percentage = 0;
	// const double perturb_t = 0.3;

	// if(poly_percentage > 0)
	// {
	// 	const int n_poly = std::max(1., mesh->n_elements()*poly_percentage);
	// 	int counter = 0;
	// 	srand(11);

	// 	for(int trial = 0; trial < n_poly*10; ++trial)
	// 	{
	// 		int el_id = rand() % mesh->n_elements();

	// 		auto tags = mesh->elements_tag();

	// 		if(mesh->is_volume())
	// 		{
	// 			assert(false);
	// 		}
	// 		else
	// 		{
	// 			const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
	// 			auto index = tmp_mesh.get_index_from_face(el_id);

	// 			bool stop = false;

	// 			for(int i = 0; i < tmp_mesh.n_face_vertices(el_id); ++i)
	// 			{
	// 				if(tmp_mesh.is_boundary_edge(index.edge))
	// 				{
	// 					stop = true;
	// 					break;
	// 				}

	// 				const auto neigh_index = tmp_mesh.switch_face(index);
	// 				if(tags[neigh_index.face] != ElementType::RegularInteriorCube)
	// 				{
	// 					stop = true;
	// 					break;
	// 				}

	// 				const auto f1 = tmp_mesh.switch_face(tmp_mesh.switch_edge(neigh_index						 )).face;
	// 				const auto f2 = tmp_mesh.switch_face(tmp_mesh.switch_edge(tmp_mesh.switch_vertex(neigh_index))).face;
	// 				if((f1 >= 0 && tags[f1] != ElementType::RegularInteriorCube) || (f2 >= 0 && tags[f2] != ElementType::RegularInteriorCube ))
	// 				{
	// 					stop = true;
	// 					break;
	// 				}

	// 				index = tmp_mesh.next_around_face(index);
	// 			}

	// 			if(stop) continue;
	// 		}

	// 		mesh->set_tag(el_id, ElementType::InteriorPolytope);
	// 		++counter;

	// 		mesh->update_elements_tag();

	// 		if(counter >= n_poly)
	// 			break;

	// 	}

	// 	if(perturb_t > 0)
	// 	{
	// 		if(mesh->is_volume())
	// 		{
	// 			assert(false);
	// 		}
	// 		else
	// 		{
	// 			Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
	// 			for(int el_id = 0; el_id < tmp_mesh.n_elements(); ++el_id)
	// 			{
	// 				if(!tmp_mesh.is_polytope(el_id))
	// 					continue;

	// 				const int rand_index = rand() % tmp_mesh.n_face_vertices(el_id);
	// 				auto index = tmp_mesh.get_index_from_face(el_id);
	// 				for(int r = 0; r < rand_index; ++r)
	// 					index = tmp_mesh.next_around_face(index);

	// 				const auto v1 = tmp_mesh.point(index.vertex);
	// 				const auto v2 = tmp_mesh.point(tmp_mesh.next_around_face(tmp_mesh.next_around_face(index)).vertex);

	// 				const double t = perturb_t + ((double) rand() / (RAND_MAX)) * 0.2 - 0.1;
	// 				const RowVectorNd v = t * v1 + (1-t) * v2;
	// 				tmp_mesh.set_point(index.vertex, v);
	// 			}
	// 		}
	// 	}
	// }
}

void State::load_febio(const std::string &path)
{
	FEBioReader::load(path, *this, "solution.txt");
}

void State::compute_mesh_stats()
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}

	bases.clear();
	pressure_bases.clear();
	geom_bases.clear();
	boundary_nodes.clear();
	local_boundary.clear();
	local_neumann_boundary.clear();
	polys.clear();
	poly_edge_to_data.clear();

	stiffness.resize(0, 0);
	rhs.resize(0, 0);
	sol.resize(0, 0);
	pressure.resize(0, 0);

	n_bases = 0;
	n_pressure_bases = 0;

	simplex_count = 0;
	regular_count = 0;
	regular_boundary_count = 0;
	simple_singular_count = 0;
	multi_singular_count = 0;
	boundary_count = 0;
	non_regular_boundary_count = 0;
	non_regular_count = 0;
	undefined_count = 0;
	multi_singular_boundary_count = 0;

	const auto &els_tag = mesh->elements_tag();

	for (size_t i = 0; i < els_tag.size(); ++i)
	{
		const ElementType type = els_tag[i];

		switch (type)
		{
		case ElementType::Simplex:
			simplex_count++;
			break;
		case ElementType::RegularInteriorCube:
			regular_count++;
			break;
		case ElementType::RegularBoundaryCube:
			regular_boundary_count++;
			break;
		case ElementType::SimpleSingularInteriorCube:
			simple_singular_count++;
			break;
		case ElementType::MultiSingularInteriorCube:
			multi_singular_count++;
			break;
		case ElementType::SimpleSingularBoundaryCube:
			boundary_count++;
			break;
		case ElementType::InterfaceCube:
		case ElementType::MultiSingularBoundaryCube:
			multi_singular_boundary_count++;
			break;
		case ElementType::BoundaryPolytope:
			non_regular_boundary_count++;
			break;
		case ElementType::InteriorPolytope:
			non_regular_count++;
			break;
		case ElementType::Undefined:
			undefined_count++;
			break;
		}
	}

	logger().info("simplex_count: \t{}", simplex_count);
	logger().info("regular_count: \t{}", regular_count);
	logger().info("regular_boundary_count: \t{}", regular_boundary_count);
	logger().info("simple_singular_count: \t{}", simple_singular_count);
	logger().info("multi_singular_count: \t{}", multi_singular_count);
	logger().info("boundary_count: \t{}", boundary_count);
	logger().info("multi_singular_boundary_count: \t{}", multi_singular_boundary_count);
	logger().info("non_regular_count: \t{}", non_regular_count);
	logger().info("non_regular_boundary_count: \t{}", non_regular_boundary_count);
	logger().info("undefined_count: \t{}", undefined_count);
	logger().info("total count:\t {}", mesh->n_elements());
}

void compute_integral_constraints(
	const Mesh3D &mesh,
	const int n_bases,
	const std::vector<ElementBases> &bases,
	const std::vector<ElementBases> &gbases,
	Eigen::MatrixXd &basis_integrals)
{
	if (!mesh.is_volume())
	{
		logger().error("Works only on volumetric meshes!");
		return;
	}
	assert(mesh.is_volume());

	basis_integrals.resize(n_bases, 9);
	basis_integrals.setZero();
	Eigen::MatrixXd rhs(n_bases, 9);
	rhs.setZero();

	const int n_elements = mesh.n_elements();
	for (int e = 0; e < n_elements; ++e)
	{
		// if (mesh.is_polytope(e)) {
		// 	continue;
		// }
		// ElementAssemblyValues vals = values[e];
		// const ElementAssemblyValues &gvals = gvalues[e];
		ElementAssemblyValues vals;
		vals.compute(e, mesh.is_volume(), bases[e], gbases[e]);

		// Computes the discretized integral of the PDE over the element
		const int n_local_bases = int(vals.basis_values.size());
		for (int j = 0; j < n_local_bases; ++j)
		{
			const AssemblyValues &v = vals.basis_values[j];
			const double integral_100 = (v.grad_t_m.col(0).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			const double integral_010 = (v.grad_t_m.col(1).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			const double integral_001 = (v.grad_t_m.col(2).array() * vals.det.array() * vals.quadrature.weights.array()).sum();

			const double integral_110 = ((vals.val.col(1).array() * v.grad_t_m.col(0).array() + vals.val.col(0).array() * v.grad_t_m.col(1).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();
			const double integral_011 = ((vals.val.col(2).array() * v.grad_t_m.col(1).array() + vals.val.col(1).array() * v.grad_t_m.col(2).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();
			const double integral_101 = ((vals.val.col(0).array() * v.grad_t_m.col(2).array() + vals.val.col(2).array() * v.grad_t_m.col(0).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();

			const double integral_200 = 2 * (vals.val.col(0).array() * v.grad_t_m.col(0).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			const double integral_020 = 2 * (vals.val.col(1).array() * v.grad_t_m.col(1).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			const double integral_002 = 2 * (vals.val.col(2).array() * v.grad_t_m.col(2).array() * vals.det.array() * vals.quadrature.weights.array()).sum();

			const double area = (v.val.array() * vals.det.array() * vals.quadrature.weights.array()).sum();

			for (size_t ii = 0; ii < v.global.size(); ++ii)
			{
				basis_integrals(v.global[ii].index, 0) += integral_100 * v.global[ii].val;
				basis_integrals(v.global[ii].index, 1) += integral_010 * v.global[ii].val;
				basis_integrals(v.global[ii].index, 2) += integral_001 * v.global[ii].val;

				basis_integrals(v.global[ii].index, 3) += integral_110 * v.global[ii].val;
				basis_integrals(v.global[ii].index, 4) += integral_011 * v.global[ii].val;
				basis_integrals(v.global[ii].index, 5) += integral_101 * v.global[ii].val;

				basis_integrals(v.global[ii].index, 6) += integral_200 * v.global[ii].val;
				basis_integrals(v.global[ii].index, 7) += integral_020 * v.global[ii].val;
				basis_integrals(v.global[ii].index, 8) += integral_002 * v.global[ii].val;

				rhs(v.global[ii].index, 6) += -2.0 * area * v.global[ii].val;
				rhs(v.global[ii].index, 7) += -2.0 * area * v.global[ii].val;
				rhs(v.global[ii].index, 8) += -2.0 * area * v.global[ii].val;
			}
		}
	}

	basis_integrals -= rhs;
}

void State::build_basis()
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}

	bases.clear();
	pressure_bases.clear();
	geom_bases.clear();
	boundary_nodes.clear();
	local_boundary.clear();
	local_neumann_boundary.clear();
	polys.clear();
	poly_edge_to_data.clear();
	stiffness.resize(0, 0);
	rhs.resize(0, 0);
	sol.resize(0, 0);
	pressure.resize(0, 0);

	n_bases = 0;
	n_pressure_bases = 0;

	sigma_avg = 0;
	sigma_max = 0;
	sigma_min = 0;

	disc_orders.resize(mesh->n_elements());
	disc_orders.setConstant(args["discr_order"]);

	auto &assembler = AssemblerUtils::instance();
	const auto params = build_json_params();
	assembler.set_parameters(params);
	problem->init(*mesh);

	logger().info("Building {} basis...", (iso_parametric() ? "isoparametric" : "not isoparametric"));
	const bool has_polys = non_regular_count > 0 || non_regular_boundary_count > 0 || undefined_count > 0;

	local_boundary.clear();
	local_neumann_boundary.clear();
	std::map<int, InterfaceData> poly_edge_to_data_geom; //temp dummy variable

	const int base_p = args["discr_order"];
	disc_orders.setConstant(base_p);

	Eigen::MatrixXi geom_disc_orders;
	if (!iso_parametric())
	{
		if (args["force_linear_geometry"] || mesh->orders().size() <= 0)
		{
			geom_disc_orders.resizeLike(disc_orders);
			geom_disc_orders.setConstant(1);
		}
		else
			geom_disc_orders = mesh->orders();
	}

	igl::Timer timer;
	timer.start();
	if (args["use_p_ref"])
	{
		if (mesh->is_volume())
			p_refinement(*dynamic_cast<Mesh3D *>(mesh.get()));
		else
			p_refinement(*dynamic_cast<Mesh2D *>(mesh.get()));

		logger().info("min p: {} max p: {}", disc_orders.minCoeff(), disc_orders.maxCoeff());
	}

	if (mesh->is_volume())
	{
		const Mesh3D &tmp_mesh = *dynamic_cast<Mesh3D *>(mesh.get());
		if (args["use_spline"])
		{
			if (!iso_parametric())
			{
				logger().error("Splines must be isoparametric, ignoring...");
				// FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], geom_disc_orders, has_polys, geom_bases, local_boundary, poly_edge_to_data_geom);
				SplineBasis3d::build_bases(tmp_mesh, args["quadrature_order"], geom_bases, local_boundary, poly_edge_to_data);
			}

			n_bases = SplineBasis3d::build_bases(tmp_mesh, args["quadrature_order"], bases, local_boundary, poly_edge_to_data);

			if (iso_parametric() && args["fit_nodes"])
				SplineBasis3d::fit_nodes(tmp_mesh, n_bases, bases);
		}
		else
		{
			if (!iso_parametric())
				FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], geom_disc_orders, false, has_polys, true, geom_bases, local_boundary, poly_edge_to_data_geom);

			n_bases = FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, args["serendipity"], has_polys, false, bases, local_boundary, poly_edge_to_data);
		}

		// if(problem->is_mixed())
		if (assembler.is_mixed(formulation()))
		{
			n_pressure_bases = FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], int(args["pressure_discr_order"]), false, has_polys, false, pressure_bases, local_boundary, poly_edge_to_data_geom);
		}
	}
	else
	{
		const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
		if (args["use_spline"])
		{

			if (!iso_parametric())
			{
				logger().error("Splines must be isoparametric, ignoring...");
				// FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, has_polys, geom_bases, local_boundary, poly_edge_to_data_geom);
				n_bases = SplineBasis2d::build_bases(tmp_mesh, args["quadrature_order"], geom_bases, local_boundary, poly_edge_to_data);
			}

			n_bases = SplineBasis2d::build_bases(tmp_mesh, args["quadrature_order"], bases, local_boundary, poly_edge_to_data);

			if (iso_parametric() && args["fit_nodes"])
				SplineBasis2d::fit_nodes(tmp_mesh, n_bases, bases);
		}
		else
		{
			if (!iso_parametric())
				FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], geom_disc_orders, false, has_polys, true, geom_bases, local_boundary, poly_edge_to_data_geom);

			n_bases = FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, args["serendipity"], has_polys, false, bases, local_boundary, poly_edge_to_data);
		}

		// if(problem->is_mixed())
		if (assembler.is_mixed(formulation()))
		{
			n_pressure_bases = FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], int(args["pressure_discr_order"]), false, has_polys, false, pressure_bases, local_boundary, poly_edge_to_data_geom);
		}
	}
	timer.stop();

	build_polygonal_basis();


	auto &gbases = iso_parametric() ? bases : geom_bases;

	n_flipped = 0;

	if (args["count_flipped_els"])
	{
		logger().info("Counting flipped elements...");
		const auto &els_tag = mesh->elements_tag();

		// flipped_elements.clear();
		for (size_t i = 0; i < gbases.size(); ++i)
		{
			if (mesh->is_polytope(i))
				continue;

			ElementAssemblyValues vals;
			if (!vals.is_geom_mapping_positive(mesh->is_volume(), gbases[i]))
			{
				++n_flipped;

				std::string type = "";
				switch (els_tag[i])
				{
				case ElementType::Simplex: type = "Simplex"; break;
				case ElementType::RegularInteriorCube: type = "RegularInteriorCube"; break;
				case ElementType::RegularBoundaryCube: type = "RegularBoundaryCube"; break;
				case ElementType::SimpleSingularInteriorCube: type = "SimpleSingularInteriorCube"; break;
				case ElementType::MultiSingularInteriorCube: type = "MultiSingularInteriorCube"; break;
				case ElementType::SimpleSingularBoundaryCube: type = "SimpleSingularBoundaryCube"; break;
				case ElementType::InterfaceCube: type = "InterfaceCube"; break;
				case ElementType::MultiSingularBoundaryCube: type = "MultiSingularBoundaryCube"; break;
				case ElementType::BoundaryPolytope: type = "BoundaryPolytope"; break;
				case ElementType::InteriorPolytope: type = "InteriorPolytope"; break;
				case ElementType::Undefined: type = "Undefined"; break;
				}

				logger().info("element {} is flipped, type {}", i, type);

				// if(!parent_elements.empty())
				// 	flipped_elements.push_back(parent_elements[i]);
			}
		}

		logger().info(" done");
	}

	// dynamic_cast<Mesh3D *>(mesh.get())->save({56}, 1, "mesh.HYBRID");

	// std::sort(flipped_elements.begin(), flipped_elements.end());
	// auto it = std::unique(flipped_elements.begin(), flipped_elements.end());
	// flipped_elements.resize(std::distance(flipped_elements.begin(), it));

	logger().info("Extracting boundary mesh...");
	extract_boundary_mesh();
	logger().info("Done!");

	problem->setup_bc(*mesh, bases, local_boundary, boundary_nodes, local_neumann_boundary);

	//add a pressure node to avoid singular solution
	if (assembler.is_mixed(formulation())) // && !assembler.is_fluid(formulation()))
	{
		if (!use_avg_pressure){
			const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
			boundary_nodes.push_back(n_bases * problem_dim + 0);

			// boundary_nodes.push_back(n_bases * problem_dim + 1);
			// boundary_nodes.push_back(n_bases * problem_dim + 2);
			// boundary_nodes.push_back(n_bases * problem_dim + 3);
			// boundary_nodes.push_back(n_bases * problem_dim + 3);
			// boundary_nodes.push_back(n_bases * problem_dim + 215);
		}
	}


	const auto &curret_bases = iso_parametric() ? bases : geom_bases;
	const int n_samples = 10;
	compute_mesh_size(*mesh, curret_bases, n_samples);

	building_basis_time = timer.getElapsedTime();
	logger().info(" took {}s", building_basis_time);

	logger().info("flipped elements {}", n_flipped);
	logger().info("h: {}", mesh_size);
	logger().info("n bases: {}", n_bases);
	logger().info("n pressure bases: {}", n_pressure_bases);
}

void State::build_polygonal_basis()
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (n_bases <= 0)
	{
		logger().error("Build the bases first!");
		return;
	}

	stiffness.resize(0, 0);
	rhs.resize(0, 0);
	sol.resize(0, 0);
	pressure.resize(0, 0);

	igl::Timer timer;
	timer.start();
	logger().info("Computing polygonal basis...");

	// std::sort(boundary_nodes.begin(), boundary_nodes.end());

	//mixed not supports polygonal bases
	assert(n_pressure_bases == 0 || poly_edge_to_data.size() == 0);

	int new_bases = 0;

	if (iso_parametric())
	{
		if (mesh->is_volume())
		{
			if (args["poly_bases"] == "MeanValue")
				logger().error("MeanValue bases not supported in 3D");
			new_bases = PolygonalBasis3d::build_bases(formulation(), args["n_harmonic_samples"], *dynamic_cast<Mesh3D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, bases, poly_edge_to_data, polys_3d);
		}
		else
		{
			if (args["poly_bases"] == "MeanValue")
			{
				new_bases = MVPolygonalBasis2d::build_bases(formulation(), *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["quadrature_order"], bases, bases, poly_edge_to_data, local_boundary, polys);
			}
			else
				new_bases = PolygonalBasis2d::build_bases(formulation(), args["n_harmonic_samples"], *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, bases, poly_edge_to_data, polys);
		}
	}
	else
	{
		if (mesh->is_volume())
		{
			if (args["poly_bases"] == "MeanValue")
				logger().error("MeanValue bases not supported in 3D");
			new_bases = PolygonalBasis3d::build_bases(formulation(), args["n_harmonic_samples"], *dynamic_cast<Mesh3D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, geom_bases, poly_edge_to_data, polys_3d);
		}
		else
		{
			if (args["poly_bases"] == "MeanValue")
				new_bases = MVPolygonalBasis2d::build_bases(formulation(), *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["quadrature_order"], bases, geom_bases, poly_edge_to_data, local_boundary, polys);
			else
				new_bases = PolygonalBasis2d::build_bases(formulation(), args["n_harmonic_samples"], *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, geom_bases, poly_edge_to_data, polys);
		}
	}

	timer.stop();
	computing_poly_basis_time = timer.getElapsedTime();
	logger().info(" took {}s", computing_poly_basis_time);

	n_bases += new_bases;
}

void State::extract_boundary_mesh()
{
	if(mesh->is_volume())
	{
		const Mesh3D &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());

		std::map<int, int> global_to_small;
		std::vector<RowVectorNd> vertices;
		std::vector<std::tuple<int, int, int>> tris;
		boundary_to_global.clear();
		int index = 0;

		for (auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &b = bases[lb.element_id()];

			for (int j = 0; j < lb.size(); ++j)
			{
				const int eid = lb.global_primitive_id(j);
				const int lid = lb[j];
				const auto nodes = b.local_nodes_for_primitive(eid, mesh3d);

				if (!mesh->is_simplex(lb.element_id()))
				{
					logger().warn("skipping element {} since it is not a simplex", eid);
					continue;
				}

				std::vector<int> loc_nodes;

				for (long n = 0; n < nodes.size(); ++n)
				{
					auto &bs = b.bases[nodes(n)];
					const auto &glob = bs.global();
					if (glob.size() != 1)
						continue;


					int gindex = glob.front().index;
					int ii = 0;
					const auto it = global_to_small.find(gindex);
					if (it == global_to_small.end())
					{
						global_to_small[gindex] = index;
						vertices.push_back(glob.front().node);
						ii = index;
						assert(boundary_to_global.size() == index);
						boundary_to_global.push_back(gindex);

						++index;
					}
					else
						ii = it->second;

					loc_nodes.push_back(ii);
				}

				if(loc_nodes.size() == 3)
				{
					tris.emplace_back(loc_nodes[0], loc_nodes[1], loc_nodes[2]);
				}
				else if (loc_nodes.size() == 6)
				{
					tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[5]);
					tris.emplace_back(loc_nodes[3], loc_nodes[1], loc_nodes[4]);
					tris.emplace_back(loc_nodes[4], loc_nodes[2], loc_nodes[5]);
					tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[5]);
				}
				else if (loc_nodes.size() == 10)
				{
					tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[8]);
					tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[9]);
					tris.emplace_back(loc_nodes[4], loc_nodes[1], loc_nodes[5]);
					tris.emplace_back(loc_nodes[5], loc_nodes[6], loc_nodes[9]);
					tris.emplace_back(loc_nodes[6], loc_nodes[2], loc_nodes[7]);
					tris.emplace_back(loc_nodes[7], loc_nodes[8], loc_nodes[9]);
					tris.emplace_back(loc_nodes[8], loc_nodes[3], loc_nodes[9]);
					tris.emplace_back(loc_nodes[9], loc_nodes[4], loc_nodes[5]);
					tris.emplace_back(loc_nodes[6], loc_nodes[7], loc_nodes[9]);
				}
				else if (loc_nodes.size() == 15)
				{
					tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[11]);
					tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[12]);
					tris.emplace_back(loc_nodes[3], loc_nodes[12], loc_nodes[11]);
					tris.emplace_back(loc_nodes[12], loc_nodes[10], loc_nodes[11]);
					tris.emplace_back(loc_nodes[4], loc_nodes[5], loc_nodes[13]);
					tris.emplace_back(loc_nodes[4], loc_nodes[13], loc_nodes[12]);
					tris.emplace_back(loc_nodes[12], loc_nodes[13], loc_nodes[14]);
					tris.emplace_back(loc_nodes[12], loc_nodes[14], loc_nodes[10]);
					tris.emplace_back(loc_nodes[14], loc_nodes[9], loc_nodes[10]);
					tris.emplace_back(loc_nodes[5], loc_nodes[1], loc_nodes[6]);
					tris.emplace_back(loc_nodes[5], loc_nodes[6], loc_nodes[13]);
					tris.emplace_back(loc_nodes[6], loc_nodes[7], loc_nodes[13]);
					tris.emplace_back(loc_nodes[13], loc_nodes[7], loc_nodes[14]);
					tris.emplace_back(loc_nodes[7], loc_nodes[8], loc_nodes[14]);
					tris.emplace_back(loc_nodes[14], loc_nodes[8], loc_nodes[9]);
					tris.emplace_back(loc_nodes[8], loc_nodes[2], loc_nodes[9]);
				}
				else
				{
					std::cout << loc_nodes.size() << std::endl;
					assert(false);
				}
			}
		}

		boundary_nodes_pos.resize(vertices.size(), 3);
		boundary_elements.resize(tris.size(), 3);

		for (int i = 0; i < vertices.size(); ++i)
		{
			boundary_nodes_pos.row(i) << vertices[i](0), vertices[i](2), vertices[i](1);
		}

		for (int i = 0; i < tris.size(); ++i)
		{
			boundary_elements.row(i) << std::get<0>(tris[i]), std::get<1>(tris[i]), std::get<2>(tris[i]);
		}
	}
	else
	{
		const Mesh2D &mesh2d = *dynamic_cast<Mesh2D *>(mesh.get());

		std::map<int, int> global_to_small;
		std::vector<RowVectorNd> vertices;
		std::vector<std::pair<int, int>> edges;
		boundary_to_global.clear();
		int index = 0;

		for (auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &b = bases[lb.element_id()];

			for(int j = 0; j < lb.size(); ++j)
			{
				const int eid = lb.global_primitive_id(j);
				const int lid = lb[j];
				const auto nodes = b.local_nodes_for_primitive(eid, mesh2d);

				int prev_node = -1;

				for (long n = 0; n < nodes.size(); ++n)
				{
					auto &bs = b.bases[nodes(n)];
					const auto &glob = bs.global();
					if(glob.size() != 1) continue;

					int gindex = glob.front().index;
					int ii = 0;
					const auto it = global_to_small.find(gindex);
					if(it == global_to_small.end())
					{
						global_to_small[gindex] = index;
						vertices.push_back(glob.front().node);
						ii = index;
						assert(boundary_to_global.size() == index);
						boundary_to_global.push_back(gindex);

						++index;
					}
					else
						ii = it->second;

					if(prev_node >= 0)
						edges.emplace_back(prev_node, ii);
					prev_node = ii;
				}
			}
		}

		boundary_nodes_pos.resize(vertices.size(), 3);
		boundary_elements.resize(edges.size(), 2);

		for (int i = 0; i < vertices.size(); ++i)
		{
			boundary_nodes_pos.row(i) << vertices[i](0), vertices[i](1), 0;
		}

		for (int i = 0; i < edges.size(); ++i)
		{
			boundary_elements.row(i) << edges[i].first, edges[i].second;
		}
	}
}

json State::build_json_params()
{
	json params = args["params"];
	params["size"] = mesh->dimension();

	return params;
}

void State::assemble_stiffness_mat()
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (n_bases <= 0)
	{
		logger().error("Build the bases first!");
		return;
	}
	if (formulation() == "OperatorSplitting")
	{
		stiffness.resize(1,1);
		assembling_stiffness_mat_time = 0;
		return;
	}

	stiffness.resize(0, 0);
	sol.resize(0, 0);
	pressure.resize(0, 0);

	igl::Timer timer;
	timer.start();
	logger().info("Assembling stiffness mat...");

	auto &assembler = AssemblerUtils::instance();

	// if(problem->is_mixed())
	if (assembler.is_mixed(formulation()))
	{
		if (assembler.is_linear(formulation()))
		{
			StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
			assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, bases, iso_parametric() ? bases : geom_bases, velocity_stiffness);
			assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, iso_parametric() ? bases : geom_bases, mixed_stiffness);
			assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, iso_parametric() ? bases : geom_bases, pressure_stiffness);

			const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();

			AssemblerUtils::merge_mixed_matrices(n_bases, n_pressure_bases, problem_dim, use_avg_pressure ? assembler.is_fluid(formulation()) : false,
												 velocity_stiffness, mixed_stiffness, pressure_stiffness,
												 stiffness);

			if (problem->is_time_dependent())
			{
				StiffnessMatrix velocity_mass;
				assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, bases, iso_parametric() ? bases : geom_bases, velocity_mass);

				std::vector<Eigen::Triplet<double>> mass_blocks;
				mass_blocks.reserve(velocity_mass.nonZeros());

				for (int k = 0; k < velocity_mass.outerSize(); ++k)
				{
					for (StiffnessMatrix::InnerIterator it(velocity_mass, k); it; ++it)
					{
						mass_blocks.emplace_back(it.row(), it.col(), it.value());
					}
				}

				mass.resize(stiffness.rows(), stiffness.cols());
				mass.setFromTriplets(mass_blocks.begin(), mass_blocks.end());
				mass.makeCompressed();
			}
		}
	}
	else
	{
		assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, bases, iso_parametric() ? bases : geom_bases, stiffness);
		if (problem->is_time_dependent())
		{
			assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, bases, iso_parametric() ? bases : geom_bases, mass);
		}
	}

	timer.stop();
	assembling_stiffness_mat_time = timer.getElapsedTime();
	logger().info(" took {}s", assembling_stiffness_mat_time);

	nn_zero = stiffness.nonZeros();
	num_dofs = stiffness.rows();
	mat_size = (long long)stiffness.rows() * (long long)stiffness.cols();
	logger().info("sparsity: {}/{}", nn_zero, mat_size);
}

void State::assemble_rhs()
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (n_bases <= 0)
	{
		logger().error("Build the bases first!");
		return;
	}
	if (formulation() == "OperatorSplitting")
	{
		rhs.resize(1,1);
		assigning_rhs_time = 0;
		return;
	}

	igl::Timer timer;
	const std::string rhs_path = args["rhs_path"];

	auto p_params = args["problem_params"];
	p_params["formulation"] = formulation();
	{
		RowVectorNd min, max, delta;
		mesh->bounding_box(min, max);
		delta = (max - min) / 2. + min;
		if (mesh->is_volume())
			p_params["bbox_center"] = {delta(0), delta(1), delta(2)};
		else
			p_params["bbox_center"] = {delta(0), delta(1)};
	}
	problem->set_parameters(p_params);

	// const auto params = build_json_params();
	auto &assembler = AssemblerUtils::instance();
	// assembler.set_parameters(params);

	// stiffness.resize(0, 0);
	rhs.resize(0, 0);
	sol.resize(0, 0);
	pressure.resize(0, 0);

	timer.start();
	logger().info("Assigning rhs...");

	const int size = problem->is_scalar() ? 1 : mesh->dimension();

	RhsAssembler rhs_assembler(*mesh, n_bases, size, bases, iso_parametric() ? bases : geom_bases, formulation(), *problem);
	if (!rhs_path.empty() || rhs_in.size() > 0)
	{
		logger().debug("Loading rhs...");

		if (rhs_in.size())
			rhs = rhs_in;
		else
			read_matrix(args["rhs_path"], rhs);

		StiffnessMatrix tmp_mass;
		assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, bases, iso_parametric() ? bases : geom_bases, tmp_mass);
		rhs = tmp_mass * rhs;
		logger().debug("done!");
	}
	else
	{
		rhs_assembler.assemble(rhs);
		rhs *= -1;
	}

	if (formulation() != "Bilaplacian")
		rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs);
	else
		rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], std::vector<LocalBoundary>(), rhs);

	// if(problem->is_mixed())
	if (assembler.is_mixed(formulation()))
	{
		const int prev_size = rhs.size();
		const int n_larger = n_pressure_bases + (use_avg_pressure ? (assembler.is_fluid(formulation()) ? 1 : 0) : 0);
		rhs.conservativeResize(prev_size + n_larger, rhs.cols());
		//Divergence free rhs
		if (formulation() != "Bilaplacian" || local_neumann_boundary.empty())
		{
			rhs.block(prev_size, 0, n_larger, rhs.cols()).setZero();
		}
		else
		{
			Eigen::MatrixXd tmp(n_pressure_bases, 1);
			tmp.setZero();
			RhsAssembler rhs_assembler1(*mesh, n_pressure_bases, size, pressure_bases, iso_parametric() ? bases : geom_bases, formulation(), *problem);
			rhs_assembler1.set_bc(std::vector<LocalBoundary>(), std::vector<int>(), args["n_boundary_samples"], local_neumann_boundary, tmp);
			rhs.block(prev_size, 0, n_larger, rhs.cols()) = tmp;
		}
	}

	timer.stop();
	assigning_rhs_time = timer.getElapsedTime();
	logger().info(" took {}s", assigning_rhs_time);
}

void State::solve_problem()
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (n_bases <= 0)
	{
		logger().error("Build the bases first!");
		return;
	}

	const auto &assembler = AssemblerUtils::instance();

	if (assembler.is_linear(formulation()) && stiffness.rows() <= 0)
	{
		logger().error("Assemble the stiffness matrix first!");
		return;
	}
	if (rhs.size() <= 0)
	{
		logger().error("Assemble the rhs first!");
		return;
	}

	sol.resize(0, 0);
	pressure.resize(0, 0);
	spectrum.setZero();

	igl::Timer timer;
	timer.start();
	logger().info("Solving {} with", formulation());

	const json &params = solver_params();

	const std::string full_mat_path = args["export"]["full_mat"];
	if (!full_mat_path.empty())
	{
		Eigen::saveMarket(stiffness, full_mat_path);
	}

	if (problem->is_time_dependent())
	{

		const double tend = args["tend"];
		const int time_steps = args["time_steps"];
		const double dt = tend / time_steps;

		const auto &gbases = iso_parametric() ? bases : geom_bases;
		RhsAssembler rhs_assembler(*mesh, n_bases, problem->is_scalar() ? 1 : mesh->dimension(), bases, gbases, formulation(), *problem);
		rhs_assembler.initial_solution(sol);

		Eigen::MatrixXd current_rhs = rhs;

		Eigen::MatrixXd density(sol.size() / mesh->dimension(), 1);
		if (args["density"])
		{
			for (int i = 0; i < density.rows(); i++)
			{
				bool flag = false;
				for (int d = 0; d < mesh->dimension(); d++)
				{
					if (abs(sol(mesh->dimension()*i+d)) > 1e-8)
					{
						flag = true;
						break;
					}
				}
				if (flag) density(i) = 1;
				else density(i) = 0;
			}
		}

		if (formulation() == "OperatorSplitting")
		{
			const int dim = mesh->dimension();
			const int n_el = int(bases.size());				// number of elements
			const int shape = gbases[0].bases.size();		// number of geometry vertices in an element
			const double viscosity_ = build_json_params()["viscosity"];

			StiffnessMatrix stiffness_viscosity, mixed_stiffness;
			// coefficient matrix of viscosity
			assembler.assemble_problem("Laplacian", mesh->is_volume(), n_bases, bases, gbases, stiffness_viscosity);
			assembler.assemble_mass_matrix("Laplacian", mesh->is_volume(), n_bases, bases, gbases, mass);

			// coefficient matrix of pressure projection
			assembler.assemble_problem("Laplacian", mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, stiffness);
			
			// matrix used to calculate divergence of velocity
			assembler.assemble_mixed_problem("Stokes", mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases, mixed_stiffness);
			mixed_stiffness = mixed_stiffness.transpose();

			// barycentric coordinates of FEM nodes
			Eigen::MatrixXd local_pts;
			if (dim == 2)
			{
				if (shape == 3)
					autogen::p_nodes_2d(args["discr_order"], local_pts);
				else
					autogen::q_nodes_2d(args["discr_order"], local_pts);
			}
			else
			{
				if (shape == 4)
					autogen::p_nodes_3d(args["discr_order"], local_pts);
				else
					autogen::q_nodes_3d(args["discr_order"], local_pts);
			}

			std::vector<int> bnd_nodes;
			bnd_nodes.reserve(boundary_nodes.size() / dim);
			for (auto it = boundary_nodes.begin(); it != boundary_nodes.end(); it++)
			{
				if (!(*it % dim)) continue;
				bnd_nodes.push_back(*it / dim);
			}

			OperatorSplittingSolver ss(*mesh, shape, n_el, local_boundary, bnd_nodes, mass, stiffness_viscosity, stiffness, dt, viscosity_, args["solver_type"], args["precond_type"], params, args["export"]["stiffness_mat"]);

			/* initialize solution */

			ss.initialize_solution(*mesh, gbases, bases, problem, sol, local_pts);
			pressure = Eigen::MatrixXd::Zero(n_pressure_bases, 1);

			/* export to vtu */
			if (args["save_time_sequence"])
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu("step_" + std::to_string(0) + ".vtu", 0.);
			}

			for (int t = 1; t <= time_steps; t++)
			{
				double time = t * dt;
				logger().info("{}/{} steps, t={}s", t, time_steps, time);
				
				/* advection */
				if(args["particle"])
					ss.advection_FLIP(*mesh, gbases, bases, sol, dt, local_pts, args["advection_order"]);
				else
				{
					if(args["density"])
						ss.advection_with_density(*mesh, gbases, bases, sol, density, dt, local_pts, args["advection_order"], args["advection_RK"]);
					else
						ss.advection(*mesh, gbases, bases, sol, dt, local_pts, args["advection_order"], args["advection_RK"]);
				}

				/* apply boundary condition */
				// ss.set_bc(*mesh, bnd_nodes, local_boundary, gbases, bases, sol, local_pts, problem, time);
				rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, time);

				/* viscosity */
				if(viscosity_ > 0)
					ss.solve_diffusion_1st(mass, bnd_nodes, sol);

				/* external force */
				ss.external_force(*mesh, gbases, bases, dt, sol, local_pts, problem, time);
				
				/* incompressibility */
				ss.solve_pressure(mixed_stiffness, sol, pressure);
				
				ss.projection(*mesh, n_bases, gbases, bases, pressure_bases, local_pts, pressure, sol);

				/* apply boundary condition */
				// ss.set_bc(*mesh, bnd_nodes, local_boundary, gbases, bases, sol, local_pts, problem, time);
				rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, time);

				/* export to vtu */
				if (args["save_time_sequence"] && !(t % (int)args["skip_frame"]))
				{
					if (!solve_export_to_file)
						solution_frames.emplace_back();
					save_vtu("step_" + std::to_string(t) + ".vtu", time);
				}
			}
		}
		else if (formulation() == "NavierStokes")
		{
			StiffnessMatrix velocity_mass;
			assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, bases, gbases, velocity_mass);

			StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;

			const int prev_size = sol.size();
			sol.conservativeResize(rhs.size(), sol.cols());
			//Zero initial pressure
			sol.block(prev_size, 0, n_pressure_bases, sol.cols()).setZero();
			sol(sol.size()-1) = 0;
			Eigen::VectorXd c_sol = sol;

			Eigen::VectorXd prev_sol;

			int BDF_order = args["BDF_order"];
			// int aux_steps = BDF_order-1;
			BDF bdf(BDF_order);
			bdf.new_solution(c_sol);

			sol = c_sol;
			sol_to_pressure();
			if (args["save_time_sequence"]){
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu("step_" + std::to_string(0) + ".vtu", 0);
				// save_wire("step_" + std::to_string(0) + ".obj");
			}

			assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, bases, gbases, velocity_stiffness);
			assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases, mixed_stiffness);
			assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_stiffness);

			TransientNavierStokesSolver ns_solver(solver_params(), build_json_params(), solver_type(), precond_type());
			const int n_larger = n_pressure_bases + (use_avg_pressure ? 1 : 0);

			for (int t = 1; t <= time_steps; ++t)
			{
				double time = t * dt;
				double current_dt = dt;

				logger().info("{}/{} steps, dt={}s t={}s", t, time_steps, current_dt, time);

				bdf.rhs(prev_sol);
				rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs, time, current_rhs);
				rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, current_rhs, time);

				const int prev_size = current_rhs.size();
				if (prev_size != rhs.size())
				{
					current_rhs.conservativeResize(prev_size + n_larger, current_rhs.cols());
					current_rhs.block(prev_size, 0, n_larger, current_rhs.cols()).setZero();
				}

				ns_solver.minimize(*this, bdf.alpha(), current_dt, prev_sol,
								   velocity_stiffness, mixed_stiffness, pressure_stiffness,
								   velocity_mass, current_rhs, c_sol);
				bdf.new_solution(c_sol);
				sol = c_sol;
				sol_to_pressure();

				if (args["save_time_sequence"] && !(t % (int)args["skip_frame"]))
				{
					if (!solve_export_to_file)
						solution_frames.emplace_back();
					save_vtu("step_" + std::to_string(t) + ".vtu", time);
					// save_wire("step_" + std::to_string(t) + ".obj");
				}
			}
		}
		else //if (formulation() != "NavierStokes")
		{
			if (assembler.is_mixed(formulation()))
			{
				pressure.resize(n_pressure_bases, 1);
				pressure.setZero();
			}

			auto solver = LinearSolver::create(args["solver_type"], args["precond_type"]);
			solver->setParameters(params);
			logger().info("{}...", solver->name());

			if (args["save_time_sequence"])
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu("step_" + std::to_string(0) + ".vtu", 0);
				save_wire("step_" + std::to_string(0) + ".obj");
			}

			if (assembler.is_mixed(formulation()))
			{
				pressure.resize(0, 0);
				const int prev_size = sol.size();
				sol.conservativeResize(rhs.size(), sol.cols());
				//Zero initial pressure
				sol.block(prev_size, 0, n_pressure_bases, sol.cols()).setZero();
				sol(sol.size()-1) = 0;
			}

			if (problem->is_scalar() || assembler.is_mixed(formulation()))
			{
				StiffnessMatrix A;
				Eigen::VectorXd b, x;


				const int BDF_order = args["BDF_order"];
				// const int aux_steps = BDF_order-1;
				BDF bdf(BDF_order);
				x = sol;
				bdf.new_solution(x);

				const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
				const int precond_num = problem_dim * n_bases;

				for (int t = 1; t <= time_steps; ++t)
				{
					double time = t * dt;
					double current_dt = dt;

					logger().info("{}/{} {}s", t, time_steps, time);
					rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs, time, current_rhs);
					rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, current_rhs, time);

					if (assembler.is_mixed(formulation()))
					{
						//divergence free
						int fluid_offset = use_avg_pressure ? (assembler.is_fluid(formulation()) ? 1 : 0) : 0;

						current_rhs.block(current_rhs.rows() - n_pressure_bases - use_avg_pressure, 0, n_pressure_bases + use_avg_pressure, current_rhs.cols()).setZero();
					}

					A = (bdf.alpha() / current_dt) * mass + stiffness;
					bdf.rhs(x);
					b = (mass * x) / current_dt;
					for (int i : boundary_nodes)
						b[i] = 0;
					b += current_rhs;

					spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], t == time_steps && args["export"]["spectrum"]);
					bdf.new_solution(x);
					sol = x;

					if (assembler.is_mixed(formulation()))
					{
						sol_to_pressure();
					}

					if (args["save_time_sequence"])
					{
						if (!solve_export_to_file)
							solution_frames.emplace_back();

						save_vtu("step_" + std::to_string(t) + ".vtu", time);
						save_wire("step_" + std::to_string(t) + ".obj");
					}
				}
			}
			else //tensor time dependent
			{
				Eigen::MatrixXd velocity, acceleration;
				rhs_assembler.initial_velocity(velocity);
				rhs_assembler.initial_acceleration(acceleration);

				const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
				const int precond_num = problem_dim * n_bases;

				if (assembler.is_linear(formulation()))
				{
					//Newmark
					const double gamma = 0.5;
					const double beta = 0.25;
					// makes the algorithm implicit and equivalent to the trapezoidal rule (unconditionally stable).

					Eigen::MatrixXd temp, b;
					StiffnessMatrix A;
					Eigen::VectorXd x, btmp;

					for (int t = 1; t <= time_steps; ++t)
					{
						const double dt2 = dt * dt;

						const Eigen::MatrixXd aOld = acceleration;
						const Eigen::MatrixXd vOld = velocity;
						const Eigen::MatrixXd uOld = sol;

						if (!problem->is_linear_in_time())
						{
							rhs_assembler.assemble(current_rhs, dt * t);
							current_rhs *= -1;
						}
						temp = -(uOld + dt * vOld + ((1 / 2. - beta) * dt2) * aOld);
						b = stiffness * temp + current_rhs;

						rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, b, dt * t);

						A = stiffness * beta * dt2 + mass;
						btmp = b;
						spectrum = dirichlet_solve(*solver, A, btmp, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], t == 1 && args["export"]["spectrum"]);
						acceleration = x;

						sol += dt * vOld + dt2 * ((1 / 2.0 - beta) * aOld + beta * acceleration);
						velocity += dt * ((1 - gamma) * aOld + gamma * acceleration);

						rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, dt * t);
						rhs_assembler.set_velocity_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, velocity, dt * t);
						rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, acceleration, dt * t);

						if (args["save_time_sequence"])
						{
							if (!solve_export_to_file)
								solution_frames.emplace_back();
							save_vtu("step_" + std::to_string(t) + ".vtu", dt * t);
							save_wire("step_" + std::to_string(t) + ".obj");
						}

						logger().info("{}/{}", t, time_steps);
					}
				}
				else //if (!assembler.is_linear(formulation()))
				{
					// {
					// 	boundary_nodes.clear();
					// 	NLProblem nl_problem(*this, rhs_assembler, 0);
					// 	tmp_sol = rhs;

					// 	// tmp_sol.setRandom();
					// 	tmp_sol.setOnes();
					// 	tmp_sol /=10000.;

					// 	velocity.setZero();
					// 	VectorXd xxx=tmp_sol;
					// 	velocity = tmp_sol;
					// 	nl_problem.init_timestep(xxx, velocity, dt);


					// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad;
					// 	nl_problem.gradient(tmp_sol, actual_grad);

					// 	StiffnessMatrix hessian;
					// 	Eigen::MatrixXd expected_hessian;
					// 	nl_problem.hessian(tmp_sol, hessian);

					// 	Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);
					// 	// std::cout << "hhh\n"<< actual_hessian<<std::endl;

					// 	for (int i = 0; i < actual_hessian.rows(); ++i)
					// 	{
					// 		double hhh = 1e-6;
					// 		VectorXd xp = tmp_sol; xp(i) += hhh;
					// 		VectorXd xm = tmp_sol; xm(i) -= hhh;

					// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
					// 		nl_problem.gradient(xp, tmp_grad_p);

					// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
					// 		nl_problem.gradient(xm, tmp_grad_m);

					// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m)/(hhh*2.);

					// 		const double vp = nl_problem.value(xp);
					// 		const double vm = nl_problem.value(xm);

					// 		const double fd = (vp-vm)/(hhh*2.);
					// 		const double  diff = std::abs(actual_grad(i) - fd);
					// 		if(diff > 1e-6)
					// 			std::cout<<"diff grad "<<i<<": "<<actual_grad(i)<<" vs "<<fd <<" error: " <<diff<<" rrr: "<<actual_grad(i)/fd<<std::endl;

					// 		for(int j = 0; j < actual_hessian.rows(); ++j)
					// 		{
					// 			const double diff = std::abs(actual_hessian(i,j) - fd_h(j));

					// 			if(diff > 1e-5)
					// 				std::cout<<"diff H "<<i<<", "<<j<<": "<<actual_hessian(i,j)<<" vs "<<fd_h(j)<<" error: " <<diff<<" rrr: "<<actual_hessian(i,j)/fd_h(j)<<std::endl;

					// 		}
					// 	}

					// 	// std::cout<<"diff grad max "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
					// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
					// 	exit(0);
					// }

					const int full_size = n_bases * mesh->dimension();
					const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();
					VectorXd tmp_sol;

					NLProblem nl_problem(*this, rhs_assembler, 0);
					nl_problem.init_timestep(sol, velocity, dt);
					nl_problem.full_to_reduced(sol, tmp_sol);

					for (int t = 1; t <= time_steps; ++t)
					{
						cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
						nlsolver.setLineSearch(args["line_search"]);
						nlsolver.minimize(nl_problem, tmp_sol);

						if (nlsolver.error_code() == -10)
						{
							logger().error("Unable to solve t={}", t*dt);
							break;
						}

						nlsolver.getInfo(solver_info);
						nl_problem.reduced_to_full(tmp_sol, sol);
						if (assembler.is_mixed(formulation()))
						{
							sol_to_pressure();
						}

						rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, dt * t);

						nl_problem.update_quantities(t*dt, sol);



						if (args["save_time_sequence"])
						{
							if (!solve_export_to_file)
								solution_frames.emplace_back();
							save_vtu("step_" + std::to_string(t) + ".vtu", dt * t);
							save_wire("step_" + std::to_string(t) + ".obj");
						}

						logger().info("{}/{}", t, time_steps);
					}
				}
			}
		}
	}
	else //if(!problem->is_time_dependent())
	{
		if (assembler.is_linear(formulation()))
		{
			auto solver = LinearSolver::create(args["solver_type"], args["precond_type"]);
			solver->setParameters(params);
			StiffnessMatrix A;
			Eigen::VectorXd b;
			logger().info("{}...", solver->name());

			const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
			const int precond_num = problem_dim * n_bases;

			A = stiffness;
			Eigen::VectorXd x;
			b = rhs;
			spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], args["export"]["spectrum"]);
			sol = x;
			solver->getInfo(solver_info);

			logger().debug("Solver error: {}", (A * sol - b).norm());

			if (assembler.is_mixed(formulation()))
			{
				sol_to_pressure();
			}
		}
		else //if (!assembler.is_linear(formulation()))
		{
			if (formulation() == "NavierStokes")
			{
				auto params = build_json_params();
				const double viscosity = params.count("viscosity") ? double(params["viscosity"]) : 1.;
				NavierStokesSolver ns_solver(viscosity, solver_params(), build_json_params(), solver_type(), precond_type());
				Eigen::VectorXd x;
				ns_solver.minimize(*this, rhs, x);
				sol = x;
				sol_to_pressure();
			}
			else
			{
				const int full_size = n_bases * mesh->dimension();
				const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();
				const double tend = args["tend"];

				const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
				const int precond_num = problem_dim * n_bases;

				int steps = args["nl_solver_rhs_steps"];
				if (steps <= 0)
				{
					RowVectorNd min, max;
					mesh->bounding_box(min, max);
					steps = problem->n_incremental_load_steps((max - min).norm());
				}
				steps = std::max(steps, 1);

				RhsAssembler rhs_assembler(*mesh, n_bases, mesh->dimension(), bases, iso_parametric() ? bases : geom_bases, formulation(), *problem);

				StiffnessMatrix nlstiffness;
				auto solver = LinearSolver::create(args["solver_type"], args["precond_type"]);
				Eigen::VectorXd x, b;
				Eigen::MatrixXd grad;
				Eigen::MatrixXd prev_rhs;

				VectorXd tmp_sol;

				double step_t = 1.0 / steps;
				double t = step_t;
				double prev_t = 0;

				sol.resizeLike(rhs);
				sol.setZero();

				prev_rhs.resizeLike(rhs);
				prev_rhs.setZero();

				x.resizeLike(sol);
				x.setZero();

				b.resizeLike(sol);
				b.setZero();

				if (args["save_solve_sequence"])
				{
					if (!solve_export_to_file)
						solution_frames.emplace_back();
					save_vtu("step_" + std::to_string(prev_t) + ".vtu",tend);
					// save_wire("step_" + std::to_string(prev_t) + ".obj");
				}

				const auto &gbases = iso_parametric() ? bases : geom_bases;
				igl::Timer update_timer;
				while (t <= 1)
				{
					if (step_t < 1e-10)
					{
						logger().error("Step too small, giving up");
						break;
					}

					logger().info("t: {} prev: {} step: {}", t, prev_t, step_t);

					NLProblem nl_problem(*this, rhs_assembler, t);

					logger().debug("Updating starting point...");
					update_timer.start();
					{
						nl_problem.hessian_full(sol, nlstiffness);
						nl_problem.gradient_no_rhs(sol, grad);

						b = grad;
						for (int bId : boundary_nodes)
							b(bId) = -(nl_problem.current_rhs()(bId) - prev_rhs(bId));
						dirichlet_solve(*solver, nlstiffness, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], args["export"]["spectrum"]);
						// logger().debug("Solver error: {}", (nlstiffness * sol - b).norm());
						x = sol - x;
						nl_problem.full_to_reduced(x, tmp_sol);
					}
					update_timer.stop();
					logger().debug("done!, took {}s", update_timer.getElapsedTime());

					if (args["save_solve_sequence_debug"])
					{
						Eigen::MatrixXd xxx = sol;
						sol = x;
						if (assembler.is_mixed(formulation()))
							sol_to_pressure();
						if (!solve_export_to_file)
							solution_frames.emplace_back();

						save_vtu("step_s_" + std::to_string(t) + ".vtu", tend);
						// save_wire("step_s_" + std::to_string(t) + ".obj");

						sol = xxx;
					}

					bool has_nan = false;
					for (int k = 0; k < tmp_sol.size(); ++k)
					{
						if (std::isnan(tmp_sol[k]))
						{
							has_nan = true;
							break;
						}
					}

					if (has_nan)
					{
						do
						{
							step_t /= 2;
							t = prev_t + step_t;
						} while (t >= 1);
						continue;
					}

					if (args["nl_solver"] == "newton")
					{
						cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
						nlsolver.setLineSearch(args["line_search"]);
						nlsolver.minimize(nl_problem, tmp_sol);

						if (nlsolver.error_code() == -10) //Nan
						{
							do
							{
								step_t /= 2;
								t = prev_t + step_t;
							} while (t >= 1);
							continue;
						}
						else
						{
							prev_t = t;
							step_t *= 2;
						}

						if (step_t > 1.0 / steps)
							step_t = 1.0 / steps;

						nlsolver.getInfo(solver_info);
					}
					else if (args["nl_solver"] == "lbfgs")
					{
						cppoptlib::LbfgsSolverL2<NLProblem> nlsolver;
						nlsolver.setLineSearch(args["line_search"]);
						nlsolver.setDebug(cppoptlib::DebugLevel::High);
						nlsolver.minimize(nl_problem, tmp_sol);

						prev_t = t;
					}
					else
					{
						throw std::invalid_argument("[State] invalid solver type for non-linear problem");
					}

					t = prev_t + step_t;
					if ((prev_t < 1 && t > 1) || abs(t - 1) < 1e-10)
						t = 1;

					nl_problem.reduced_to_full(tmp_sol, sol);

					// std::ofstream of("sol.txt");
					// of<<sol<<std::endl;
					// of.close();
					prev_rhs = nl_problem.current_rhs();
					if (args["save_solve_sequence"])
					{
						if (!solve_export_to_file)
							solution_frames.emplace_back();
						save_vtu("step_" + std::to_string(prev_t) + ".vtu", tend);
						// save_wire("step_" + std::to_string(prev_t) + ".obj");
					}
				}

				if (assembler.is_mixed(formulation()))
				{
					sol_to_pressure();
				}

				// {
				// 	boundary_nodes.clear();
				// 	NLProblem nl_problem(*this, rhs_assembler, t);
				// 	tmp_sol = rhs;

				// 	// tmp_sol.setRandom();
				// 	tmp_sol.setOnes();
				// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad;
				// 	nl_problem.gradient(tmp_sol, actual_grad);

				// 	StiffnessMatrix hessian;
				// 	// Eigen::MatrixXd expected_hessian;
				// 	nl_problem.hessian(tmp_sol, hessian);
				// 	// nl_problem.finiteGradient(tmp_sol, expected_grad, 0);

				// 	Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);
				// 	// std::cout << "hhh\n"<< actual_hessian<<std::endl;

				// 		for (int i = 0; i < actual_hessian.rows(); ++i)
				// 	{
				// 		double hhh = 1e-7;
				// 		VectorXd xp = tmp_sol; xp(i) += hhh;
				// 		VectorXd xm = tmp_sol; xm(i) -= hhh;

				// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
				// 		nl_problem.gradient(xp, tmp_grad_p);

				// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
				// 		nl_problem.gradient(xm, tmp_grad_m);

				// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m)/(hhh*2.);

				// 		const double vp = nl_problem.value(xp);
				// 		const double vm = nl_problem.value(xm);

				// 		const double fd = (vp-vm)/(hhh*2.);
				// 		const double  diff = std::abs(actual_grad(i) - fd);
				// 		// if(diff > 1e-6)
				// 		// 	std::cout<<"diff grad "<<i<<": "<<actual_grad(i)<<" vs "<<fd <<" error: " <<diff<<" rrr: "<<actual_grad(i)/fd<<std::endl;

				// 		for(int j = 0; j < actual_hessian.rows(); ++j)
				// 		{
				// 			const double diff = std::abs(actual_hessian(i,j) - fd_h(j));

				// 			if(diff > 1e-5)
				// 				std::cout<<"diff H "<<i<<", "<<j<<": "<<actual_hessian(i,j)<<" vs "<<fd_h(j)<<" error: " <<diff<<" rrr: "<<actual_hessian(i,j)/fd_h(j)<<std::endl;

				// 		}
				// 	}

				// 	// std::cout<<"diff grad max "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
				// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
				// 	exit(0);
				// }

				// NLProblem::reduced_to_full_aux(full_size, reduced_size, tmp_sol, rhs, sol);
			}
		}
	}

	timer.stop();
	solving_time = timer.getElapsedTime();
	logger().info(" took {}s", solving_time);
}

void State::compute_errors()
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (n_bases <= 0)
	{
		logger().error("Build the bases first!");
		return;
	}
	// if (stiffness.rows() <= 0) { logger().error("Assemble the stiffness matrix first!"); return; }
	if (rhs.size() <= 0)
	{
		logger().error("Assemble the rhs first!");
		return;
	}
	if (sol.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}

	int actual_dim = 1;
	if (!problem->is_scalar())
		actual_dim = mesh->dimension();

	igl::Timer timer;
	timer.start();
	logger().info("Computing errors...");
	using std::max;

	const int n_el = int(bases.size());

	MatrixXd v_exact, v_approx;
	MatrixXd v_exact_grad(0, 0), v_approx_grad;

	l2_err = 0;
	h1_err = 0;
	grad_max_err = 0;
	h1_semi_err = 0;
	linf_err = 0;
	lp_err = 0;
	// double pred_norm = 0;
	double div_l2 = 0;

	static const int p = 8;

	// Eigen::MatrixXd err_per_el(n_el, 5);
	ElementAssemblyValues vals;

	const double tend = args["tend"];

	for (int e = 0; e < n_el; ++e)
	{
		// const auto &vals    = values[e];
		// const auto &gvalues = iso_parametric() ? values[e] : geom_values[e];

		if (iso_parametric())
			vals.compute(e, mesh->is_volume(), bases[e], bases[e]);
		else
			vals.compute(e, mesh->is_volume(), bases[e], geom_bases[e]);

		if (problem->has_exact_sol())
		{
			problem->exact(vals.val, tend, v_exact);
			problem->exact_grad(vals.val, tend, v_exact_grad);
		}

		v_approx.resize(vals.val.rows(), actual_dim);
		v_approx.setZero();

		v_approx_grad.resize(vals.val.rows(), mesh->dimension() * actual_dim);
		v_approx_grad.setZero();

		const int n_loc_bases = int(vals.basis_values.size());

		for (int i = 0; i < n_loc_bases; ++i)
		{
			auto val = vals.basis_values[i];

			for (size_t ii = 0; ii < val.global.size(); ++ii)
			{
				for (int d = 0; d < actual_dim; ++d)
				{
					v_approx.col(d) += val.global[ii].val * sol(val.global[ii].index * actual_dim + d) * val.val;
					v_approx_grad.block(0, d * val.grad_t_m.cols(), v_approx_grad.rows(), val.grad_t_m.cols()) += val.global[ii].val * sol(val.global[ii].index * actual_dim + d) * val.grad_t_m;
				}
			}
		}

		const auto err = problem->has_exact_sol() ? (v_exact - v_approx).eval().rowwise().norm().eval() : (v_approx).eval().rowwise().norm().eval();
		const auto err_grad = problem->has_exact_sol() ? (v_exact_grad - v_approx_grad).eval().rowwise().norm().eval() : (v_approx_grad).eval().rowwise().norm().eval();
		const auto div = (v_approx_grad).eval().rowwise().sum().eval();
		// for(long i = 0; i < err.size(); ++i)
		// errors.push_back(err(i));

		linf_err = max(linf_err, err.maxCoeff());
		grad_max_err = max(linf_err, err_grad.maxCoeff());

		// {
		// 	const auto &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());
		// 	const auto v0 = mesh3d.point(mesh3d.cell_vertex(e, 0));
		// 	const auto v1 = mesh3d.point(mesh3d.cell_vertex(e, 1));
		// 	const auto v2 = mesh3d.point(mesh3d.cell_vertex(e, 2));
		// 	const auto v3 = mesh3d.point(mesh3d.cell_vertex(e, 3));

		// 	Eigen::Matrix<double, 6, 3> ee;
		// 	ee.row(0) = v0 - v1;
		// 	ee.row(1) = v1 - v2;
		// 	ee.row(2) = v2 - v0;

		// 	ee.row(3) = v0 - v3;
		// 	ee.row(4) = v1 - v3;
		// 	ee.row(5) = v2 - v3;

		// 	Eigen::Matrix<double, 6, 1> en = ee.rowwise().norm();

		// 	// Eigen::Matrix<double, 3*4, 1> alpha;
		// 	// alpha(0) = angle3(e.row(0), -e.row(1));	 	alpha(1) = angle3(e.row(1), -e.row(2));	 	alpha(2) = angle3(e.row(2), -e.row(0));
		// 	// alpha(3) = angle3(e.row(0), -e.row(4));	 	alpha(4) = angle3(e.row(4), e.row(3));	 	alpha(5) = angle3(-e.row(3), -e.row(0));
		// 	// alpha(6) = angle3(-e.row(4), -e.row(1));	alpha(7) = angle3(e.row(1), -e.row(5));	 	alpha(8) = angle3(e.row(5), e.row(4));
		// 	// alpha(9) = angle3(-e.row(2), -e.row(5));	alpha(10) = angle3(e.row(5), e.row(3));		alpha(11) = angle3(-e.row(3), e.row(2));

		// 	const double S = (ee.row(0).cross(ee.row(1)).norm() + ee.row(0).cross(ee.row(4)).norm() + ee.row(4).cross(ee.row(1)).norm() + ee.row(2).cross(ee.row(5)).norm()) / 2;
		// 	const double V = std::abs(ee.row(3).dot(ee.row(2).cross(-ee.row(0))))/6;
		// 	const double rho = 3 * V / S;
		// 	const double hp = en.maxCoeff();
		// 	const int pp = disc_orders(e);
		// 	const int p_ref = args["discr_order"];

		// 	err_per_el(e, 0) = err.mean();
		// 	err_per_el(e, 1) = err.maxCoeff();
		// 	err_per_el(e, 2) = std::pow(hp, pp+1)/(rho/hp); // /std::pow(average_edge_length, p_ref+1) * (sqrt(6)/12);
		// 	err_per_el(e, 3) = rho/hp;
		// 	err_per_el(e, 4) = (vals.det.array() * vals.quadrature.weights.array()).sum();

		// 	// pred_norm += (pow(std::pow(hp, pp+1)/(rho/hp),p) * vals.det.array() * vals.quadrature.weights.array()).sum();
		// }

		l2_err += (err.array() * err.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
		h1_err += (err_grad.array() * err_grad.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
		lp_err += (err.array().pow(p) * vals.det.array() * vals.quadrature.weights.array()).sum();
		div_l2 += (div.array().pow(2) * vals.det.array() * vals.quadrature.weights.array()).sum();
	}

	h1_semi_err = sqrt(fabs(h1_err));
	h1_err = sqrt(fabs(l2_err) + fabs(h1_err));
	l2_err = sqrt(fabs(l2_err));

	lp_err = pow(fabs(lp_err), 1. / p);
	div_l2 = pow(fabs(div_l2), 1. / 2);

	// pred_norm = pow(fabs(pred_norm), 1./p);

	timer.stop();
	computing_errors_time = timer.getElapsedTime();
	logger().info(" took {}s", computing_errors_time);

	logger().info("-- L2 error: {}", l2_err);
	logger().info("-- Lp error: {}", lp_err);
	logger().info("-- H1 error: {}", h1_err);
	logger().info("-- H1 semi error: {}", h1_semi_err);
	// logger().info("-- Perd norm: {}", pred_norm);

	logger().info("-- Linf error: {}", linf_err);
	logger().info("-- grad max error: {}", grad_max_err);
	logger().info("-- div l2 norm: {}", div_l2);

	logger().info("total time: {}s", (building_basis_time + assembling_stiffness_mat_time + solving_time));

	// {
	// 	std::ofstream out("errs.txt");
	// 	out<<err_per_el;
	// 	out.close();
	// }
}

void State::init(const json &args_in)
{
	this->args.merge_patch(args_in);

	if (args_in.find("stiffness_mat_save_path") != args_in.end() && !args_in["stiffness_mat_save_path"].empty())
	{
		logger().warn("use export: { stiffness_mat: 'path' } instead of stiffness_mat_save_path");
		this->args["export"]["stiffness_mat"] = args_in["stiffness_mat_save_path"];
	}

	if (args_in.find("solution") != args_in.end() && !args_in["solution"].empty())
	{
		logger().warn("use export: { solution: 'path' } instead of solution");
		this->args["export"]["solution"] = args_in["solution"];
	}

	problem = ProblemFactory::factory().get_problem(args["problem"]);
	//important for the BC
	problem->set_parameters(args["problem_params"]);

	if (args["use_spline"] && args["n_refs"] == 0)
	{
		logger().warn("n_refs > 0 with spline");
	}
}

void State::export_data()
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (n_bases <= 0)
	{
		logger().error("Build the bases first!");
		return;
	}
	// if (stiffness.rows() <= 0) { logger().error("Assemble the stiffness matrix first!"); return; }
	if (rhs.size() <= 0)
	{
		logger().error("Assemble the rhs first!");
		return;
	}
	if (sol.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}

	// Export vtu mesh of solution + wire mesh of deformed input
	// + mesh colored with the bases
	const std::string paraview_path = args["export"]["paraview"];
	const std::string old_path = args["export"]["vis_mesh"];
	const std::string vis_mesh_path = paraview_path.empty() ? old_path : paraview_path;
	const std::string wire_mesh_path = args["export"]["wire_mesh"];
	const std::string iso_mesh_path = args["export"]["iso_mesh"];
	const std::string nodes_path = args["export"]["nodes"];
	const std::string solution_path = args["export"]["solution"];
	const std::string solmat_path = args["export"]["solution_mat"];
	const std::string stress_path = args["export"]["stress_mat"];
	const std::string mises_path = args["export"]["mises"];

	if (!solution_path.empty())
	{
		std::ofstream out(solution_path);
		out.precision(100);
		out << std::scientific;
		out << sol << std::endl;
		out.close();
	}

	const double tend = args["tend"];

	if (!vis_mesh_path.empty())
	{
		save_vtu(vis_mesh_path, tend);
	}
	if (!wire_mesh_path.empty())
	{
		save_wire(wire_mesh_path);
	}
	if (!iso_mesh_path.empty())
	{
		save_wire(iso_mesh_path, true);
	}
	if (!nodes_path.empty())
	{
		MatrixXd nodes(n_bases, mesh->dimension());
		for (const ElementBases &eb : bases)
		{
			for (const Basis &b : eb.bases)
			{
				// for(const auto &lg : b.global())
				for (size_t ii = 0; ii < b.global().size(); ++ii)
				{
					const auto &lg = b.global()[ii];
					nodes.row(lg.index) = lg.node;
				}
			}
		}
		std::ofstream out(nodes_path);
		out.precision(100);
		out << nodes;
		out.close();
	}
	if (!solmat_path.empty())
	{
		Eigen::MatrixXd result;
		int problem_dim = (problem->is_scalar() ? 1 : mesh->dimension());
		compute_vertex_values(problem_dim, bases, sol, result);
		std::ofstream out(solmat_path);
		out.precision(20);
		out << result;
	}
	if (!stress_path.empty())
	{
		Eigen::MatrixXd result;
		Eigen::VectorXd mises;
		compute_stress_at_quadrature_points(sol, result, mises);
		std::ofstream out(stress_path);
		out.precision(20);
		out << result;
	}
	if (!mises_path.empty())
	{
		Eigen::MatrixXd result;
		Eigen::VectorXd mises;
		compute_stress_at_quadrature_points(sol, result, mises);
		std::ofstream out(mises_path);
		out.precision(20);
		out << mises;
	}
}

void State::build_vis_mesh(Eigen::MatrixXd &points, Eigen::MatrixXi &tets, Eigen::MatrixXi &el_id, Eigen::MatrixXd &discr)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (n_bases <= 0)
	{
		logger().error("Build the bases first!");
		return;
	}

	const auto &sampler = RefElementSampler::sampler();

	const auto &current_bases = iso_parametric() ? bases : geom_bases;
	int tet_total_size = 0;
	int pts_total_size = 0;

	const bool boundary_only = args["export"]["vis_boundary_only"];

	Eigen::MatrixXd vis_pts_poly;
	Eigen::MatrixXi vis_faces_poly;

	for (size_t i = 0; i < current_bases.size(); ++i)
	{
		const auto &bs = current_bases[i];

		if (boundary_only && mesh->is_volume() && !mesh->is_boundary_element(i))
			continue;

		if (mesh->is_simplex(i))
		{
			tet_total_size += sampler.simplex_volume().rows();
			pts_total_size += sampler.simplex_points().rows();
		}
		else if (mesh->is_cube(i))
		{
			tet_total_size += sampler.cube_volume().rows();
			pts_total_size += sampler.cube_points().rows();
		}
		else
		{
			if (mesh->is_volume())
			{
				sampler.sample_polyhedron(polys_3d[i].first, polys_3d[i].second, vis_pts_poly, vis_faces_poly);

				tet_total_size += vis_faces_poly.rows();
				pts_total_size += vis_pts_poly.rows();
			}
			else
			{
				sampler.sample_polygon(polys[i], vis_pts_poly, vis_faces_poly);

				tet_total_size += vis_faces_poly.rows();
				pts_total_size += vis_pts_poly.rows();
			}
		}
	}

	points.resize(pts_total_size, mesh->dimension());
	tets.resize(tet_total_size, mesh->is_volume() ? 4 : 3);

	el_id.resize(pts_total_size, 1);
	discr.resize(pts_total_size, 1);

	Eigen::MatrixXd mapped, tmp;
	int tet_index = 0, pts_index = 0;

	for (size_t i = 0; i < current_bases.size(); ++i)
	{
		const auto &bs = current_bases[i];

		if (boundary_only && mesh->is_volume() && !mesh->is_boundary_element(i))
			continue;

		if (mesh->is_simplex(i))
		{
			bs.eval_geom_mapping(sampler.simplex_points(), mapped);

			tets.block(tet_index, 0, sampler.simplex_volume().rows(), tets.cols()) = sampler.simplex_volume().array() + pts_index;
			tet_index += sampler.simplex_volume().rows();

			points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
			discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
			el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
			pts_index += mapped.rows();
		}
		else if (mesh->is_cube(i))
		{
			bs.eval_geom_mapping(sampler.cube_points(), mapped);

			tets.block(tet_index, 0, sampler.cube_volume().rows(), tets.cols()) = sampler.cube_volume().array() + pts_index;
			tet_index += sampler.cube_volume().rows();

			points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
			discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
			el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
			pts_index += mapped.rows();
		}
		else
		{
			if (mesh->is_volume())
			{
				sampler.sample_polyhedron(polys_3d[i].first, polys_3d[i].second, vis_pts_poly, vis_faces_poly);
				bs.eval_geom_mapping(vis_pts_poly, mapped);

				tets.block(tet_index, 0, vis_faces_poly.rows(), tets.cols()) = vis_faces_poly.array() + pts_index;
				tet_index += vis_faces_poly.rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(-1);
				el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
				pts_index += mapped.rows();
			}
			else
			{
				sampler.sample_polygon(polys[i], vis_pts_poly, vis_faces_poly);
				bs.eval_geom_mapping(vis_pts_poly, mapped);

				tets.block(tet_index, 0, vis_faces_poly.rows(), tets.cols()) = vis_faces_poly.array() + pts_index;
				tet_index += vis_faces_poly.rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(-1);
				el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
				pts_index += mapped.rows();
			}
		}
	}

	assert(pts_index == points.rows());
	assert(tet_index == tets.rows());
}

void State::save_vtu(const std::string &path, const double t)
{
	if (!mesh)
	{
		logger().error("Load the mesh first!");
		return;
	}
	if (n_bases <= 0)
	{
		logger().error("Build the bases first!");
		return;
	}
	// if (stiffness.rows() <= 0) { logger().error("Assemble the stiffness matrix first!"); return; }
	if (rhs.size() <= 0)
	{
		logger().error("Assemble the rhs first!");
		return;
	}
	if (sol.size() <= 0)
	{
		logger().error("Solve the problem first!");
		return;
	}

	const auto &assembler = AssemblerUtils::instance();

	Eigen::MatrixXd points;
	Eigen::MatrixXi tets;
	Eigen::MatrixXi el_id;
	Eigen::MatrixXd discr;

	build_vis_mesh(points, tets, el_id, discr);

	Eigen::MatrixXd fun, exact_fun, err;
	const bool boundary_only = args["export"]["vis_boundary_only"];
	const bool material_params = args["export"]["material_params"];

	interpolate_function(points.rows(), sol, fun, boundary_only);

	if (problem->has_exact_sol())
	{
		problem->exact(points, t, exact_fun);
		err = (fun - exact_fun).eval().rowwise().norm();
	}

	VTUWriter writer;

	if (solve_export_to_file && fun.cols() != 1 && !mesh->is_volume())
	{
		fun.conservativeResize(fun.rows(), 3);
		fun.col(2).setZero();

		exact_fun.conservativeResize(exact_fun.rows(), 3);
		exact_fun.col(2).setZero();
	}

	if (solve_export_to_file)
		writer.add_field("solution", fun);
	else
		solution_frames.back().solution = fun;

	// if(problem->is_mixed())
	if (assembler.is_mixed(formulation()))
	{
		Eigen::MatrixXd interp_p;
		interpolate_function(points.rows(), 1, pressure_bases, pressure, interp_p, boundary_only);
		if (solve_export_to_file)
			writer.add_field("pressure", interp_p);
		else
			solution_frames.back().pressure = fun;
	}

	if (solve_export_to_file)
		writer.add_field("discr", discr);
	if (problem->has_exact_sol())
	{
		if (solve_export_to_file)
		{
			writer.add_field("exact", exact_fun);
			writer.add_field("error", err);
		}
		else
		{
			solution_frames.back().exact = exact_fun;
			solution_frames.back().error = err;
		}
	}

	if (fun.cols() != 1)
	{
		Eigen::MatrixXd vals, tvals;
		compute_scalar_value(points.rows(), sol, vals, boundary_only);
		if (solve_export_to_file)
			writer.add_field("scalar_value", vals);
		else
			solution_frames.back().scalar_value = vals;

		if (solve_export_to_file)
		{
			compute_tensor_value(points.rows(), sol, tvals, boundary_only);
			for (int i = 0; i < tvals.cols(); ++i)
			{
				const int ii = (i / mesh->dimension()) + 1;
				const int jj = (i % mesh->dimension()) + 1;
				writer.add_field("tensor_value_" + std::to_string(ii) + std::to_string(jj), tvals.col(i));
			}
		}

		if (!args["use_spline"])
		{
			average_grad_based_function(points.rows(), sol, vals, tvals, boundary_only);
			if (solve_export_to_file)
				writer.add_field("scalar_value_avg", vals);
			else
				solution_frames.back().scalar_value_avg = vals;
			// for(int i = 0; i < tvals.cols(); ++i){
			// 	const int ii = (i / mesh->dimension()) + 1;
			// 	const int jj = (i % mesh->dimension()) + 1;
			// 	writer.add_field("tensor_value_avg_" + std::to_string(ii) + std::to_string(jj), tvals.col(i));
			// }
		}
	}

	if(material_params)
	{
		LameParameters params;
		params.init(build_json_params());

		Eigen::MatrixXd lambdas(points.rows(), 1);
		Eigen::MatrixXd mus(points.rows(), 1);

		for(int i = 0; i < points.rows(); ++i)
		{
			double lambda, mu;

			params.lambda_mu(points(i, 0), points(i, 1), points.cols() >= 3 ? points(i, 2) : 0, el_id(i), lambda, mu);
			lambdas(i) = lambda;
			mus(i) = mu;
		}

		writer.add_field("lambda", lambdas);
		writer.add_field("mu", mus);
	}

	// interpolate_function(pts_index, rhs, fun, boundary_only);
	// writer.add_field("rhs", fun);
	if (solve_export_to_file)
		writer.write_tet_mesh(path, points, tets);
	else
	{
		solution_frames.back().name = path;
		solution_frames.back().points = points;
		solution_frames.back().connectivity = tets;
	}
}

void State::save_wire(const std::string &name, bool isolines)
{
	if (!solve_export_to_file) //TODO?
		return;
	const auto &sampler = RefElementSampler::sampler();

	const auto &current_bases = iso_parametric() ? bases : geom_bases;
	int seg_total_size = 0;
	int pts_total_size = 0;
	int faces_total_size = 0;

	for (size_t i = 0; i < current_bases.size(); ++i)
	{
		const auto &bs = current_bases[i];

		if (mesh->is_simplex(i))
		{
			pts_total_size += sampler.simplex_points().rows();
			seg_total_size += sampler.simplex_edges().rows();
			faces_total_size += sampler.simplex_faces().rows();
		}
		else if (mesh->is_cube(i))
		{
			pts_total_size += sampler.cube_points().rows();
			seg_total_size += sampler.cube_edges().rows();
		}
		//TODO add edges for poly
	}

	Eigen::MatrixXd points(pts_total_size, mesh->dimension());
	Eigen::MatrixXi edges(seg_total_size, 2);
	Eigen::MatrixXi faces(faces_total_size, 3);
	points.setZero();

	MatrixXd mapped, tmp;
	int seg_index = 0, pts_index = 0, face_index = 0;
	for (size_t i = 0; i < current_bases.size(); ++i)
	{
		const auto &bs = current_bases[i];

		if (mesh->is_simplex(i))
		{
			bs.eval_geom_mapping(sampler.simplex_points(), mapped);
			edges.block(seg_index, 0, sampler.simplex_edges().rows(), edges.cols()) = sampler.simplex_edges().array() + pts_index;
			seg_index += sampler.simplex_edges().rows();

			faces.block(face_index, 0, sampler.simplex_faces().rows(), 3) = sampler.simplex_faces().array() + pts_index;
			face_index += sampler.simplex_faces().rows();

			points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
			pts_index += mapped.rows();
		}
		else if (mesh->is_cube(i))
		{
			bs.eval_geom_mapping(sampler.cube_points(), mapped);
			edges.block(seg_index, 0, sampler.cube_edges().rows(), edges.cols()) = sampler.cube_edges().array() + pts_index;
			seg_index += sampler.cube_edges().rows();

			points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
			pts_index += mapped.rows();
		}
	}

	assert(pts_index == points.rows());
	assert(face_index == faces.rows());

	if (mesh->is_volume())
	{
		//reverse all faces
		for (long i = 0; i < faces.rows(); ++i)
		{
			const int v0 = faces(i, 0);
			const int v1 = faces(i, 1);
			const int v2 = faces(i, 2);

			int tmpc = faces(i, 2);
			faces(i, 2) = faces(i, 1);
			faces(i, 1) = tmpc;
		}
	}
	else
	{
		Matrix2d mmat;
		for (long i = 0; i < faces.rows(); ++i)
		{
			const int v0 = faces(i, 0);
			const int v1 = faces(i, 1);
			const int v2 = faces(i, 2);

			mmat.row(0) = points.row(v2) - points.row(v0);
			mmat.row(1) = points.row(v1) - points.row(v0);

			if (mmat.determinant() > 0)
			{
				int tmpc = faces(i, 2);
				faces(i, 2) = faces(i, 1);
				faces(i, 1) = tmpc;
			}
		}
	}

	Eigen::MatrixXd fun;
	interpolate_function(pts_index, sol, fun);

	// Eigen::MatrixXd exact_fun, err;


	// if (problem->has_exact_sol())
	// {
	// 	problem->exact(points, exact_fun);
	// 	err = (fun - exact_fun).eval().rowwise().norm();
	// }

	if (fun.cols() != 1 && !mesh->is_volume())
	{
		fun.conservativeResize(fun.rows(), 3);
		fun.col(2).setZero();

	// 	exact_fun.conservativeResize(exact_fun.rows(), 3);
	// 	exact_fun.col(2).setZero();
	}

	if (!mesh->is_volume())
	{
		points.conservativeResize(points.rows(), 3);
		points.col(2).setZero();
	}

	// writer.add_field("solution", fun);
	// if (problem->has_exact_sol()) {
	// 	writer.add_field("exact", exact_fun);
	// 	writer.add_field("error", err);
	// }

	// if (fun.cols() != 1) {
	// 	Eigen::MatrixXd scalar_val;
	// 	compute_scalar_value(pts_index, sol, scalar_val);
	// 	writer.add_field("scalar_value", scalar_val);
	// }

	if (fun.cols() != 1)
	{
		assert(points.rows() == fun.rows());
		assert(points.cols() == fun.cols());
		points += fun;
	}
	else
	{
		if (isolines)
			points.col(2) += fun;
	}

	if (isolines)
	{
		Eigen::MatrixXd isoV;
		Eigen::MatrixXi isoE;
		igl::isolines(points, faces, Eigen::VectorXd(fun), 20, isoV, isoE);
		igl::write_triangle_mesh("foo.obj", points, faces);
		points = isoV;
		edges = isoE;
	}

	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::VectorXi I, J;
	igl::remove_unreferenced(points, edges, V, E, I);
	igl::remove_duplicate_vertices(V, E, 1e-14, points, I, J, edges);

	// Remove loops
	int last = edges.rows() - 1;
	int new_size = edges.rows();
	for (int i = 0; i <= last; ++i)
	{
		if (edges(i, 0) == edges(i, 1))
		{
			edges.row(i) = edges.row(last);
			--last;
			--i;
			--new_size;
		}
	}
	edges.conservativeResize(new_size, edges.cols());

	save_edges(name, points, edges);
}

// void State::compute_poly_basis_error(const std::string &path)
// {

// 	MatrixXd fun = MatrixXd::Zero(n_bases, 1);
// 	MatrixXd tmp, mapped;
// 	MatrixXd v_approx, v_exact;

// 	int poly_index = -1;

// 	for(size_t i = 0; i < bases.size(); ++i)
// 	{
// 		const ElementBases &basis = bases[i];
// 		if(!basis.has_parameterization){
// 			poly_index = i;
// 			continue;
// 		}

// 		for(std::size_t j = 0; j < basis.bases.size(); ++j)
// 		{
// 			for(std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
// 			{
// 				const Local2Global &l2g = basis.bases[j].global()[kk];
// 				const int g_index = l2g.index;

// 				const auto &node = l2g.node;
// 				problem->exact(node, tmp);

// 				fun(g_index) = tmp(0);
// 			}
// 		}
// 	}

// 	if(poly_index == -1)
// 		poly_index = 0;

// 	auto &poly_basis = bases[poly_index];
// 	ElementAssemblyValues vals;
// 	vals.compute(poly_index, true, poly_basis, poly_basis);

// 	// problem.exact(vals.val, v_exact);
// 	v_exact.resize(vals.val.rows(), vals.val.cols());
// 	dx(vals.val, tmp); v_exact.col(0) = tmp;
// 	dy(vals.val, tmp); v_exact.col(1) = tmp;
// 	dz(vals.val, tmp); v_exact.col(2) = tmp;

// 	v_approx = MatrixXd::Zero(v_exact.rows(), v_exact.cols());

// 	const int n_loc_bases=int(vals.basis_values.size());

// 	for(int i = 0; i < n_loc_bases; ++i)
// 	{
// 		auto &val=vals.basis_values[i];

// 		for(std::size_t ii = 0; ii < val.global.size(); ++ii)
// 		{
// 			// v_approx += val.global[ii].val * fun(val.global[ii].index) * val.val;
// 			v_approx += val.global[ii].val * fun(val.global[ii].index) * val.grad;
// 		}
// 	}

// 	const Eigen::MatrixXd err = (v_exact-v_approx).cwiseAbs();

// 	using json = nlohmann::json;
// 	json j;
// 	j["mesh_path"] = mesh_path;

// 	for(long c = 0; c < v_approx.cols();++c){
// 		double l2_err_interp = 0;
// 		double lp_err_interp = 0;

// 		l2_err_interp += (err.col(c).array() * err.col(c).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
// 		lp_err_interp += (err.col(c).array().pow(8.) * vals.det.array() * vals.quadrature.weights.array()).sum();

// 		l2_err_interp = sqrt(fabs(l2_err_interp));
// 		lp_err_interp = pow(fabs(lp_err_interp), 1./8.);

// 		j["err_l2_"+std::to_string(c)] = l2_err_interp;
// 		j["err_lp_"+std::to_string(c)] = lp_err_interp;
// 	}

// 	std::ofstream out(path);
// 	out << j.dump(4) << std::endl;
// 	out.close();
// }

} // namespace polyfem
