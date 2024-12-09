#include <polyfem/State.hpp>

#include <polyfem/problem/ProblemFactory.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/assembler/Mass.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/GeogramUtils.hpp>
#include <polyfem/problem/KernelProblem.hpp>
#include <polyfem/utils/par_for.hpp>

#include <polyfem/utils/JSONUtils.hpp>

#include <jse/jse.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>

#include <ipc/utils/logger.hpp>
#include <wmtk/utils/Logger.hpp>

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

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
	using namespace problem;
	using namespace utils;

	State::State()
	{
		using namespace polysolve;

		GeogramUtils::instance().initialize();

		problem = ProblemFactory::factory().get_problem("Linear");
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
			// Set the file sink separately from the console so it can save all messages
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

		wmtk::set_logger(std::make_shared<spdlog::logger>("wmtk", sinks.begin(), sinks.end()));

		// Set the logger at the lowest level, so all messages are passed to the sinks
		logger().set_level(spdlog::level::trace);
		ipc::logger().set_level(spdlog::level::trace);
		wmtk::logger().set_level(spdlog::level::trace);

		set_log_level(log_level);
	}

	void State::set_log_level(const spdlog::level::level_enum log_level)
	{
		spdlog::set_level(log_level);
		if (console_sink_)
		{
			// Set only the level of the console
			console_sink_->set_level(log_level); // Shared by all loggers
		}
		else
		{
			// Set the level of all sinks
			logger().set_level(log_level);
			ipc::logger().set_level(log_level);
		}
	}

	void State::init(const json &p_args_in, const bool strict_validation)
	{
		json args_in = p_args_in; // mutable copy

		apply_common_params(args_in);

		// CHECK validity json
		json rules;
		jse::JSE jse;
		{
			jse.strict = strict_validation;
			const std::string polyfem_input_spec = POLYFEM_INPUT_SPEC;
			std::ifstream file(polyfem_input_spec);

			if (file.is_open())
				file >> rules;
			else
			{
				logger().error("unable to open {} rules", polyfem_input_spec);
				throw std::runtime_error("Invald spec file");
			}

			jse.include_directories.push_back(POLYFEM_JSON_SPEC_DIR);
			jse.include_directories.push_back(POLYSOLVE_JSON_SPEC_DIR);
			rules = jse.inject_include(rules);

			polysolve::linear::Solver::apply_default_solver(rules, "/solver/linear");
			polysolve::linear::Solver::apply_default_solver(rules, "/solver/adjoint_linear");
		}

		polysolve::linear::Solver::select_valid_solver(args_in["solver"]["linear"], logger());
		if (args_in["solver"]["adjoint_linear"].is_null())
			args_in["solver"]["adjoint_linear"] = args_in["solver"]["linear"];
		else
			polysolve::linear::Solver::select_valid_solver(args_in["solver"]["adjoint_linear"], logger());

		// Use the /solver/nonlinear settings as the default for /solver/augmented_lagrangian/nonlinear
		if (args_in.contains("/solver/nonlinear"_json_pointer))
		{
			if (args_in.contains("/solver/augmented_lagrangian/nonlinear"_json_pointer))
			{
				assert(args_in["solver"]["augmented_lagrangian"]["nonlinear"].is_object());
				// Merge the augmented lagrangian settings into the nonlinear settings,
				// and then replace the augmented lagrangian settings with the merged settings.
				json nonlinear = args_in["solver"]["nonlinear"]; // copy
				nonlinear.merge_patch(args_in["solver"]["augmented_lagrangian"]["nonlinear"]);
				args_in["solver"]["augmented_lagrangian"]["nonlinear"] = nonlinear;
			}
			else
			{
				// Copy the nonlinear settings to the augmented_lagrangian settings
				args_in["solver"]["augmented_lagrangian"]["nonlinear"] = args_in["solver"]["nonlinear"];
			}
		}

		const bool valid_input = jse.verify_json(args_in, rules);

		if (!valid_input)
		{
			logger().error("invalid input json:\n{}", jse.log2str());
			throw std::runtime_error("Invald input json file");
		}
		// end of check

		this->args = jse.inject_defaults(args_in, rules);
		units.init(this->args["units"]);

		// Save output directory and resolve output paths dynamically
		const std::string output_dir = resolve_input_path(this->args["output"]["directory"]);
		if (!output_dir.empty())
		{
			std::filesystem::create_directories(output_dir);
		}
		this->output_dir = output_dir;

		std::string out_path_log = this->args["output"]["log"]["path"];
		if (!out_path_log.empty())
		{
			out_path_log = resolve_output_path(out_path_log);
		}

		init_logger(
			out_path_log,
			this->args["output"]["log"]["level"],
			this->args["output"]["log"]["file_level"],
			this->args["output"]["log"]["quiet"]);

		logger().info("Saving output to {}", output_dir);

		const unsigned int thread_in = this->args["solver"]["max_threads"];
		set_max_threads(thread_in);

		has_dhat = args_in["contact"].contains("dhat");

		init_time();

		if (is_contact_enabled())
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

		const std::string formulation = this->formulation();
		assembler = assembler::AssemblerUtils::make_assembler(formulation);
		assert(assembler->name() == formulation);
		mass_matrix_assembler = std::make_shared<assembler::Mass>();
		const auto other_name = assembler::AssemblerUtils::other_assembler_name(formulation);

		if (!other_name.empty())
		{
			mixed_assembler = assembler::AssemblerUtils::make_mixed_assembler(formulation);
			pressure_assembler = assembler::AssemblerUtils::make_assembler(other_name);
		}

		if (args["solver"]["advanced"]["check_inversion"] == "Conservative")
		{
			if (auto elastic_assembler = std::dynamic_pointer_cast<assembler::ElasticityAssembler>(assembler))
				elastic_assembler->set_use_robust_jacobian();
		}

		if (!args.contains("preset_problem"))
		{
			if (!assembler->is_tensor())
				problem = std::make_shared<assembler::GenericScalarProblem>("GenericScalar");
			else
				problem = std::make_shared<assembler::GenericTensorProblem>("GenericTensor");

			problem->clear();
			if (!args["time"].is_null())
			{
				const auto tmp = R"({"is_time_dependent": true})"_json;
				problem->set_parameters(tmp);
			}
			// important for the BC

			auto bc = args["boundary_conditions"];
			bc["root_path"] = root_path();
			problem->set_parameters(bc);
			problem->set_parameters(args["initial_conditions"]);

			problem->set_parameters(args["output"]);
		}
		else
		{
			if (args["preset_problem"]["type"] == "Kernel")
			{
				problem = std::make_shared<KernelProblem>("Kernel", *assembler);
				problem->clear();
				KernelProblem &kprob = *dynamic_cast<KernelProblem *>(problem.get());
			}
			else
			{
				problem = ProblemFactory::factory().get_problem(args["preset_problem"]["type"]);
				problem->clear();
			}
			// important for the BC
			problem->set_parameters(args["preset_problem"]);
		}

		problem->set_units(*assembler, units);

		if (optimization_enabled == solver::CacheLevel::Derivatives)
		{
			if (is_contact_enabled())
			{
				if (!args["contact"]["use_convergent_formulation"])
				{
					args["contact"]["use_convergent_formulation"] = true;
					logger().info("Use convergent formulation for differentiable contact...");
				}
				if (args["/solver/contact/barrier_stiffness"_json_pointer].is_string())
				{
					logger().error("Only constant barrier stiffness is supported in differentiable contact!");
				}
			}

			if (args.contains("boundary_conditions") && args["boundary_conditions"].contains("rhs"))
			{
				json rhs = args["boundary_conditions"]["rhs"];
				if ((rhs.is_array() && rhs.size() > 0 && rhs[0].is_string()) || rhs.is_string())
					logger().error("Only constant rhs over space is supported in differentiable code!");
			}
		}
	}

	void State::set_max_threads(const int max_threads)
	{
		NThread::get().set_num_threads(max_threads);
	}

	void State::init_time()
	{
		if (!is_param_valid(args, "time"))
			return;

		const double t0 = Units::convert(args["time"]["t0"], units.time());
		double tend, dt;
		int time_steps;

		// from "tend", "dt", "time_steps" only two can be used at a time
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
					time_steps = int(ceil((tend - t0) / dt));
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
				// tend is already confirmed to be invalid, so time_steps must be valid
				assert(is_param_valid(args["time"], "time_steps"));

				dt = Units::convert(args["time"]["dt"], units.time());
				assert(dt > 0);

				time_steps = args["time"]["time_steps"];
				assert(time_steps > 0);

				tend = t0 + time_steps * dt;
			}
			else
			{
				// tend and dt are already confirmed to be invalid
				throw std::runtime_error("This code should be unreachable!");
			}
		}
		else if (num_valid == 3)
		{
			tend = Units::convert(args["time"]["tend"], units.time());
			dt = Units::convert(args["time"]["dt"], units.time());
			time_steps = args["time"]["time_steps"];

			// Check that all parameters agree
			if (abs(t0 + dt * time_steps - tend) > 1e-12)
			{
				log_and_throw_error("Exactly two of (tend, dt, time_steps) must be specified");
			}
		}

		// Store these for use later
		args["time"]["tend"] = tend;
		args["time"]["dt"] = dt;
		args["time"]["time_steps"] = time_steps;

		units.characteristic_length() *= dt;

		logger().info("t0={}, dt={}, tend={}", t0, dt, tend);
	}

	void State::set_materials(std::vector<std::shared_ptr<assembler::Assembler>> &assemblers) const
	{
		const int size = (assembler->is_tensor() || assembler->is_fluid()) ? mesh->dimension() : 1;
		for (auto &a : assemblers)
			a->set_size(size);

		if (!utils::is_param_valid(args, "materials"))
			return;

		if (!args["materials"].is_array() && args["materials"]["type"] == "AMIPSAutodiff")
		{
			json transform_params = {};
			transform_params["canonical_transformation"] = json::array();
			if (!mesh->is_volume())
			{
				Eigen::MatrixXd regular_tri(3, 3);
				regular_tri << 0, 0, 1,
					1, 0, 1,
					1. / 2., std::sqrt(3) / 2., 1;
				regular_tri.transposeInPlace();
				Eigen::MatrixXd regular_tri_inv = regular_tri.inverse();

				const auto &mesh2d = *dynamic_cast<mesh::Mesh2D *>(mesh.get());
				for (int e = 0; e < mesh->n_elements(); e++)
				{
					Eigen::MatrixXd transform;
					mesh2d.compute_face_jacobian(e, regular_tri_inv, transform);
					transform_params["canonical_transformation"].push_back(json({
						{
							transform(0, 0),
							transform(0, 1),
						},
						{
							transform(1, 0),
							transform(1, 1),
						},
					}));
				}
			}
			else
			{
				Eigen::MatrixXd regular_tet(4, 4);
				regular_tet << 0, 0, 0, 1,
					1, 0, 0, 1,
					1. / 2., std::sqrt(3) / 2., 0, 1,
					1. / 2., 1. / 2. / std::sqrt(3), std::sqrt(3) / 2., 1;
				regular_tet.transposeInPlace();
				Eigen::MatrixXd regular_tet_inv = regular_tet.inverse();

				const auto &mesh3d = *dynamic_cast<mesh::Mesh3D *>(mesh.get());
				for (int e = 0; e < mesh->n_elements(); e++)
				{
					Eigen::MatrixXd transform;
					mesh3d.compute_cell_jacobian(e, regular_tet_inv, transform);
					transform_params["canonical_transformation"].push_back(json({
						{
							transform(0, 0),
							transform(0, 1),
							transform(0, 2),
						},
						{
							transform(1, 0),
							transform(1, 1),
							transform(1, 2),
						},
						{
							transform(2, 0),
							transform(2, 1),
							transform(2, 2),
						},
					}));
				}
			}
			transform_params["solve_displacement"] = true;
			assembler->set_materials({}, transform_params, units);

			return;
		}

		std::vector<int> body_ids(mesh->n_elements());
		for (int i = 0; i < mesh->n_elements(); ++i)
			body_ids[i] = mesh->get_body_id(i);

		for (auto &a : assemblers)
			a->set_materials(body_ids, args["materials"], units);
	}

	void State::set_materials(assembler::Assembler &assembler) const
	{
		const int size = (this->assembler->is_tensor() || this->assembler->is_fluid()) ? this->mesh->dimension() : 1;
		assembler.set_size(size);

		if (!utils::is_param_valid(args, "materials"))
			return;

		std::vector<int> body_ids(mesh->n_elements());
		for (int i = 0; i < mesh->n_elements(); ++i)
			body_ids[i] = mesh->get_body_id(i);

		assembler.set_materials(body_ids, args["materials"], units);
	}

} // namespace polyfem
