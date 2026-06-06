#include "ElasticVarForm.hpp"

#include <polyfem/assembler/ElementAssemblyValues.hpp>

#include <polyfem/assembler/MultiModel.hpp>

#include <polyfem/basis/Basis.hpp>

#include <polyfem/io/MshWriter.hpp>
#include <polyfem/io/MatrixIO.hpp>

#include <polyfem/problem/KernelProblem.hpp>
#include <polyfem/problem/ProblemFactory.hpp>

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <ostream>

#include <spdlog/fmt/fmt.h>

namespace polyfem::varform
{
	void ElasticVarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		VarForm::init(formulation, units, args, out_path);
		const bool is_time_dependent = args.contains("time") && !args["time"].is_null();

		assembler = assembler::AssemblerUtils::make_assembler(formulation);
		assert(assembler->name() == formulation);
		mass_matrix_assembler = std::make_shared<assembler::Mass>();
		pure_mass_matrix_assembler = std::make_shared<assembler::HRZMass>();

		if (!args.contains("preset_problem"))
		{
			problem = std::make_shared<assembler::GenericTensorProblem>("GenericTensor");

			problem->clear();
			json tmp;
			tmp["is_time_dependent"] = is_time_dependent;
			problem->set_parameters(tmp, root_path);

			// important for the BC

			auto bc = args["boundary_conditions"];
			bc["root_path"] = root_path;
			problem->set_parameters(bc, root_path);
			problem->set_parameters(args["initial_conditions"], root_path);
			problem->set_parameters(args["output"], root_path);
		}
		else
		{
			if (args["preset_problem"]["type"] == "Kernel")
			{
				problem = std::make_shared<problem::KernelProblem>("Kernel", *assembler);
				problem->clear();
				problem::KernelProblem &kprob = *dynamic_cast<problem::KernelProblem *>(problem.get());
			}
			else
			{
				problem = problem::ProblemFactory::factory().get_problem(args["preset_problem"]["type"]);
				problem->clear();
			}
			// important for the BC
			problem->set_parameters(args["preset_problem"], root_path);
		}

		problem->set_units(*assembler, units);

		t0 = is_time_dependent ? args["time"]["t0"].get<double>() : 0.0;
		time_steps = is_time_dependent ? args["time"]["time_steps"].get<int>() : 0;
		dt = is_time_dependent ? args["time"]["dt"].get<double>() : 0.0;
	}

	void ElasticVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		VarForm::load_mesh(mesh, args);

		if (assembler::MultiModel *mm = dynamic_cast<assembler::MultiModel *>(assembler.get()))
		{
			assert(args["materials"].is_array());

			std::vector<std::string> materials(mesh.n_elements());

			std::map<int, std::string> mats;

			for (const auto &m : args["materials"])
				mats[m["id"].get<int>()] = m["type"];

			for (int i = 0; i < materials.size(); ++i)
				materials[i] = mats.at(mesh.get_body_id(i));

			mm->init_multimodels(materials);
		}
	}

	void ElasticVarForm::initial_velocity(Eigen::MatrixXd &velocity) const
	{
		assert(solve_data.rhs_assembler != nullptr);

		const bool was_velocity_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "v",
			args["input"]["data"]["reorder"], in_node_to_node,
			mesh_->dimension(), velocity);

		if (!was_velocity_loaded)
			solve_data.rhs_assembler->initial_velocity(velocity);
	}

	void ElasticVarForm::initial_acceleration(Eigen::MatrixXd &acceleration) const
	{
		assert(solve_data.rhs_assembler != nullptr);

		const bool was_acceleration_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "a",
			args["input"]["data"]["reorder"], in_node_to_node,
			mesh_->dimension(), acceleration);

		if (!was_acceleration_loaded)
			solve_data.rhs_assembler->initial_acceleration(acceleration);
	}

	io::OutStatsData ElasticVarForm::compute_errors(const Eigen::MatrixXd &solution)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return stats;

		double tend = 0;
		if (!args["time"].is_null())
			tend = args["time"]["tend"];

		stats.compute_errors(n_bases, bases, geom_bases(), *mesh_, *problem, tend, solution);
		return stats;
	}

	void ElasticVarForm::save_json(const Eigen::MatrixXd &solution, std::ostream &out) const
	{
		if (!mesh_)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (solution.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		logger().info("Saving json...");
		const int actual_dim = problem_dimension();
		const int primary_size = n_bases * actual_dim;
		const Eigen::MatrixXd stats_solution =
			solution.rows() >= primary_size
				? solution.topRows(primary_size).eval()
				: solution;

		nlohmann::json j;
		stats.save_json(
			args, n_bases, 0,
			stats_solution, *mesh_, disc_orders, disc_ordersq, *problem,
			timings, assembler ? assembler->name() : name(), iso_parametric,
			args["output"]["advanced"]["sol_at_node"], j);
		out << j.dump(4) << std::endl;
	}

	void ElasticVarForm::build_mesh_matrices(Eigen::MatrixXd &V, Eigen::MatrixXi &F) const
	{
		assert(mesh_);
		assert(bases.size() == mesh_->n_elements());
		const size_t n_vertices = n_bases - n_obstacle_vertices();
		const int dim = mesh_->dimension();

		V.resize(n_vertices, dim);
		F.resize(bases.size(), dim + 1);

		for (int i = 0; i < bases.size(); i++)
		{
			const basis::ElementBases &element = bases[i];
			for (int j = 0; j < element.bases.size(); j++)
			{
				const basis::Basis &basis = element.bases[j];
				assert(basis.global().size() == 1);
				V.row(basis.global()[0].index) = basis.global()[0].node;
				if (j < F.cols())
					F(i, j) = basis.global()[0].index;
			}
		}
	}

	void ElasticVarForm::save_step_state(const double t0, const double dt, const int t, const Eigen::MatrixXd &sol) const
	{
		if (!mesh_)
			return;

		const std::string rest_mesh_path = args["output"]["data"]["rest_mesh"].get<std::string>();
		if (!rest_mesh_path.empty())
		{
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			build_mesh_matrices(V, F);
			io::MshWriter::write(
				resolve_output_path(fmt::format(args["output"]["data"]["rest_mesh"], t)),
				V, F, mesh_->get_body_ids(), mesh_->is_volume(), /*binary=*/true);
		}

		const std::string state_path = resolve_output_path(fmt::format(args["output"]["data"]["state"], t));
		if (!state_path.empty() && solve_data.time_integrator)
			solve_data.time_integrator->save_state(state_path);

		save_restart_json(t0, dt, t);
	}

} // namespace polyfem::varform
