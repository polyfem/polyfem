#include <polyfem/State.hpp>

#include <polyfem/Units.hpp>

#include <polyfem/mesh/GeometryReader.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/utils/GeogramUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/par_for.hpp>

#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/varforms/VarFormFactory.hpp>

#include <jse/jse.h>
#include <polyfem/embedded_spec/polyfem.hpp>

#include <polysolve/linear/Solver.hpp>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <ipc/utils/logger.hpp>
#ifdef POLYFEM_WITH_ITR
#include <wmtk/utils/Logger.hpp>
#endif

#include <igl/Timer.h>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace spdlog::level
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		spdlog::level::level_enum,
		{{spdlog::level::level_enum::trace, "trace"},
		 {spdlog::level::level_enum::debug, "debug"},
		 {spdlog::level::level_enum::info, "info"},
		 {spdlog::level::level_enum::warn, "warning"},
		 {spdlog::level::level_enum::err, "error"},
		 {spdlog::level::level_enum::critical, "critical"},
		 {spdlog::level::level_enum::off, "off"},
		 {spdlog::level::level_enum::trace, 0},
		 {spdlog::level::level_enum::debug, 1},
		 {spdlog::level::level_enum::info, 2},
		 {spdlog::level::level_enum::warn, 3},
		 {spdlog::level::level_enum::err, 3},
		 {spdlog::level::level_enum::critical, 4},
		 {spdlog::level::level_enum::off, 5}})
}

namespace polyfem
{
	using namespace mesh;
	using namespace utils;

	namespace
	{
		std::string root_path(const json &args)
		{
			if (utils::is_param_valid(args, "root_path"))
				return args["root_path"].get<std::string>();
			return "";
		}

		std::string resolve_input_path(const json &args, const std::string &path, const bool only_if_exists = false)
		{
			return utils::resolve_path(path, root_path(args), only_if_exists);
		}

		std::string resolve_output_path(const std::string &output_dir, const std::string &path)
		{
			if (output_dir.empty() || path.empty() || std::filesystem::path(path).is_absolute())
				return path;
			return std::filesystem::weakly_canonical(std::filesystem::path(output_dir) / path).string();
		}

		bool contact_enabled(const json &args)
		{
			return args["contact"]["enabled"];
		}

		void init_time(json &args, Units &units)
		{
			if (!is_param_valid(args, "time"))
				return;

			const double t0 = Units::convert(args["time"]["t0"], units.time());
			double tend = 0;
			double dt = 0;
			int time_steps = 0;

			const int num_valid = is_param_valid(args["time"], "tend")
								  + is_param_valid(args["time"], "dt")
								  + is_param_valid(args["time"], "time_steps");
			if (num_valid < 2)
			{
				log_and_throw_error("Exactly two of (tend, dt, time_steps) must be specified");
			}
			else if (num_valid == 2)
			{
				if (is_param_valid(args["time"], "tend"))
				{
					tend = Units::convert(args["time"]["tend"], units.time());
					assert(tend > t0);
					if (is_param_valid(args["time"], "dt"))
					{
						dt = Units::convert(args["time"]["dt"], units.time());
						assert(dt > 0);
						time_steps = int(std::ceil((tend - t0) / dt));
						assert(time_steps > 0);
					}
					else if (is_param_valid(args["time"], "time_steps"))
					{
						time_steps = args["time"]["time_steps"];
						assert(time_steps > 0);
						dt = (tend - t0) / time_steps;
						assert(dt > 0);
					}
					else
					{
						throw std::runtime_error("This code should be unreachable!");
					}
				}
				else if (is_param_valid(args["time"], "dt"))
				{
					assert(is_param_valid(args["time"], "time_steps"));

					dt = Units::convert(args["time"]["dt"], units.time());
					assert(dt > 0);

					time_steps = args["time"]["time_steps"];
					assert(time_steps > 0);

					tend = t0 + time_steps * dt;
				}
				else
				{
					throw std::runtime_error("This code should be unreachable!");
				}
			}
			else if (num_valid == 3)
			{
				tend = Units::convert(args["time"]["tend"], units.time());
				dt = Units::convert(args["time"]["dt"], units.time());
				time_steps = args["time"]["time_steps"];

				if (std::abs(t0 + dt * time_steps - tend) > 1e-12)
					log_and_throw_error("Exactly two of (tend, dt, time_steps) must be specified");
			}

			args["time"]["tend"] = tend;
			args["time"]["dt"] = dt;
			args["time"]["time_steps"] = time_steps;

			units.characteristic_length() *= dt;

			logger().info("t0={}, dt={}, tend={}", t0, dt, tend);
		}
	} // namespace

	State::State()
	{
		GeogramUtils::instance().initialize();
	}

	void State::init_logger(
		const std::string &log_file,
		const spdlog::level::level_enum log_level,
		const spdlog::level::level_enum file_log_level,
		const bool is_quiet)
	{
		std::vector<spdlog::sink_ptr> sinks;

		if (!is_quiet)
		{
			console_sink_ = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
			sinks.emplace_back(console_sink_);
		}

		if (!log_file.empty())
		{
			file_sink_ = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, /*truncate=*/true);
			file_sink_->set_level(file_log_level);
			sinks.push_back(file_sink_);
		}

		init_logger(sinks, log_level);
		spdlog::flush_every(std::chrono::seconds(3));
	}

	void State::init_logger(std::ostream &os, const spdlog::level::level_enum log_level)
	{
		std::vector<spdlog::sink_ptr> sinks;
		sinks.emplace_back(std::make_shared<spdlog::sinks::ostream_sink_mt>(os, false));
		init_logger(sinks, log_level);
	}

	void State::init_logger(
		const std::vector<spdlog::sink_ptr> &sinks,
		const spdlog::level::level_enum log_level)
	{
		set_logger(std::make_shared<spdlog::logger>("polyfem", sinks.begin(), sinks.end()));
		GeogramUtils::instance().set_logger(logger());

		ipc::set_logger(std::make_shared<spdlog::logger>("ipctk", sinks.begin(), sinks.end()));

#ifdef POLYFEM_WITH_ITR
		wmtk::set_logger(std::make_shared<spdlog::logger>("wmtk", sinks.begin(), sinks.end()));
#endif

		logger().set_level(spdlog::level::trace);
		ipc::logger().set_level(spdlog::level::trace);
#ifdef POLYFEM_WITH_ITR
		wmtk::logger().set_level(spdlog::level::trace);
#endif

		set_log_level(log_level);
	}

	void State::set_log_level(const spdlog::level::level_enum log_level)
	{
		spdlog::set_level(log_level);
		if (console_sink_)
		{
			console_sink_->set_level(log_level);
		}
		else
		{
			logger().set_level(log_level);
			ipc::logger().set_level(log_level);
		}
	}

	void State::init(const json &p_args_in, const bool strict_validation)
	{
		json args_in = p_args_in;
		const bool contact_dhat_was_explicit = args_in.contains("/contact/dhat"_json_pointer);

		apply_common_params(args_in);

		json rules;
		jse::JSE jse;
		{
			jse.strict = strict_validation;
			rules = jse::embed::polyfem_spec::polyfem::spec();

			polysolve::linear::Solver::apply_default_solver(rules, "/solver/linear");
			polysolve::linear::Solver::apply_default_solver(rules, "/solver/adjoint_linear");
		}

		polysolve::linear::Solver::select_valid_solver(args_in["solver"]["linear"], logger());
		if (args_in["solver"]["adjoint_linear"].is_null())
			args_in["solver"]["adjoint_linear"] = args_in["solver"]["linear"];
		else
			polysolve::linear::Solver::select_valid_solver(args_in["solver"]["adjoint_linear"], logger());

		if (args_in.contains("/solver/nonlinear"_json_pointer))
		{
			if (args_in.contains("/solver/augmented_lagrangian/nonlinear"_json_pointer))
			{
				assert(args_in["solver"]["augmented_lagrangian"]["nonlinear"].is_object());
				json nonlinear = args_in["solver"]["nonlinear"];
				nonlinear.merge_patch(args_in["solver"]["augmented_lagrangian"]["nonlinear"]);
				args_in["solver"]["augmented_lagrangian"]["nonlinear"] = nonlinear;
			}
			else
			{
				args_in["solver"]["augmented_lagrangian"]["nonlinear"] = args_in["solver"]["nonlinear"];
			}
		}

		const bool valid_input = jse.verify_json(args_in, rules);
		if (!valid_input)
		{
			logger().error("invalid input json:\n{}", jse.log2str());
			throw std::runtime_error("Invalid input json file");
		}

		args = jse.inject_defaults(args_in, rules);

		Units units;
		units.init(args["units"]);

		if (!args_in.contains("/space/advanced/bc_method"_json_pointer) && args["space"]["basis_type"] != "Lagrange")
		{
			logger().warn("Setting bc method to lsq for non-Lagrange basis");
			args["space"]["advanced"]["bc_method"] = "lsq";
		}

		const std::string output_dir = resolve_input_path(args, args["output"]["directory"].get<std::string>());
		if (!output_dir.empty())
			std::filesystem::create_directories(output_dir);

		std::string out_path_log = args["output"]["log"]["path"];
		if (!out_path_log.empty())
			out_path_log = resolve_output_path(output_dir, out_path_log);

		for (auto &path : args["constraints"]["hard"])
			path = resolve_input_path(args, path.get<std::string>());

		for (auto &path : args["constraints"]["soft"])
			path["data"] = resolve_input_path(args, path["data"].get<std::string>());

		init_logger(
			out_path_log,
			args["output"]["log"]["level"],
			args["output"]["log"]["file_level"],
			args["output"]["log"]["quiet"]);

		logger().info("Saving output to {}", output_dir);

		set_max_threads(args["solver"]["max_threads"]);

		init_time(args, units);

		if (contact_enabled(args))
		{
			if (args["solver"]["contact"]["friction_iterations"] == 0)
			{
				logger().info("specified friction_iterations is 0; disabling friction");
				args["contact"]["friction_coefficient"] = 0.0;
			}
			else if (args["solver"]["contact"]["friction_iterations"] < 0)
			{
				args["solver"]["contact"]["friction_iterations"] = std::numeric_limits<int>::max();
			}
			if (args["contact"]["friction_coefficient"] == 0.0)
			{
				args["solver"]["contact"]["friction_iterations"] = 0;
			}
		}
		else
		{
			args["solver"]["contact"]["friction_iterations"] = 0;
			args["contact"]["friction_coefficient"] = 0;
			args["contact"]["periodic"] = false;
		}

		const std::string formulation = varform::formulation_from_args(args);
		if (formulation.empty())
		{
			logger().error("specify some 'materials'");
			throw std::runtime_error("invalid input");
		}

		variational_formulation = varform::VarFormFactory::create(formulation, args);
		if (!variational_formulation)
			throw std::runtime_error("polyfem::State is varform-only; use polyfem::legacy::State for " + formulation + ".");

		logger().info("Using variational formulation: {}", variational_formulation->name());
		args["contact"]["_dhat_was_explicit"] = contact_dhat_was_explicit;
		variational_formulation->init(formulation, units, args, output_dir);
		args["contact"].erase("_dhat_was_explicit");
	}

	void State::set_max_threads(const int max_threads)
	{
		NThread::get().set_num_threads(max_threads);
	}

	void State::load_mesh(
		GEO::Mesh &meshin,
		const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &boundary_marker,
		bool non_conforming,
		bool skip_boundary_sideset)
	{
		igl::Timer timer;
		timer.start();
		logger().info("Loading mesh...");

		std::unique_ptr<Mesh> mesh = Mesh::create(meshin, non_conforming);
		if (!mesh)
		{
			logger().error("Unable to load the mesh");
			return;
		}

		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		logger().info("mesh bb min [{}], max [{}]", min, max);

		if (!skip_boundary_sideset)
			mesh->compute_boundary_ids(boundary_marker);

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		assert(variational_formulation != nullptr);
		variational_formulation->set_mesh(std::move(mesh), timer.getElapsedTime());
	}

	void State::load_mesh(
		bool non_conforming,
		const std::vector<std::string> &names,
		const std::vector<Eigen::MatrixXi> &cells,
		const std::vector<Eigen::MatrixXd> &vertices)
	{
		assert(names.size() == cells.size());
		assert(vertices.size() == cells.size());

		igl::Timer timer;
		timer.start();

		logger().info("Loading mesh ...");
		assert(is_param_valid(args, "geometry"));
		Units units;
		units.init(args["units"]);
		std::unique_ptr<Mesh> mesh = mesh::read_fem_geometry(
			units,
			args["geometry"], args["root_path"],
			names, vertices, cells, non_conforming);

		if (mesh == nullptr)
			log_and_throw_error("unable to load the mesh!");

		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		logger().info("mesh bb min [{}], max [{}]", min, max);

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

#ifdef POLYFEM_WITH_MISO
		if (!mesh->is_simplicial())
#else
		if constexpr (true)
#endif
		{
			args["space"]["advanced"]["count_flipped_els_continuous"] = false;
			args["output"]["paraview"]["options"]["jacobian_validity"] = false;
			args["solver"]["advanced"]["check_inversion"] = "Discrete";
		}
		else if (args["solver"]["advanced"]["check_inversion"] != "Discrete")
		{
			args["space"]["advanced"]["use_corner_quadrature"] = true;
		}
		// FIXME: this is a temporary workaround to avoid incorrect Jacobian validity results for non-simplicial meshes when using discrete inversion checking. We should instead implement proper Jacobian validity checking for non-simplicial meshes.
		assert(variational_formulation != nullptr);
		variational_formulation->set_args(args);
		variational_formulation->set_mesh(std::move(mesh), timer.getElapsedTime());
	}

	void State::solve(Eigen::MatrixXd &sol)
	{
		assert(variational_formulation != nullptr);

		variational_formulation->set_time_callback(time_callback);
		variational_formulation->solve(sol);
	}

	void State::load_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, bool non_conforming)
	{
		assert(variational_formulation != nullptr);
		igl::Timer timer;
		timer.start();
		auto mesh = mesh::Mesh::create(V, F, non_conforming);
		timer.stop();
		variational_formulation->set_mesh(std::move(mesh), timer.getElapsedTime());
	}
} // namespace polyfem
