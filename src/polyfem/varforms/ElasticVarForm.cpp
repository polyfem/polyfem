#include "ElasticVarForm.hpp"

#include <polyfem/assembler/ElementAssemblyValues.hpp>

#include <polyfem/assembler/MultiModel.hpp>

#include <polyfem/basis/Basis.hpp>

#include <polyfem/io/MshWriter.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <polyfem/mesh/Obstacle.hpp>

#include <polyfem/problem/KernelProblem.hpp>
#include <polyfem/problem/ProblemFactory.hpp>

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

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
		assert(rhs_assembler != nullptr);

		const bool was_velocity_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "v",
			args["input"]["data"]["reorder"], in_node_to_node,
			mesh_->dimension(), velocity);

		if (!was_velocity_loaded)
			rhs_assembler->initial_velocity(velocity);
	}

	void ElasticVarForm::initial_acceleration(Eigen::MatrixXd &acceleration) const
	{
		assert(rhs_assembler != nullptr);

		const bool was_acceleration_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "a",
			args["input"]["data"]["reorder"], in_node_to_node,
			mesh_->dimension(), acceleration);

		if (!was_acceleration_loaded)
			rhs_assembler->initial_acceleration(acceleration);
	}

	std::vector<io::OutputField> ElasticVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		std::vector<io::OutputField> fields = common_output_fields(sample, solution, options);
		append_primary_output_fields(fields, sample, solution, options);
		return fields;
	}

	void ElasticVarForm::append_primary_output_fields(
		std::vector<io::OutputField> &fields,
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options,
		const mesh::Obstacle *obstacle) const
	{
		if (!mesh_ || solution.size() <= 0)
			return;

		const int dim = mesh_->dimension();
		const bool has_element_samples =
			sample.local_points.rows() > 0
			&& sample.local_points.rows() == sample.element_ids.size();
		const bool export_solution_gradient =
			!options.fields.empty() && options.export_field("solution_gradient");

		Eigen::MatrixXd values, gradients;
		if (has_element_samples)
		{
			values.resize(sample.local_points.rows(), dim);
			if (export_solution_gradient)
				gradients.resize(sample.local_points.rows(), dim * mesh_->dimension());
			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id < 0)
				{
					values.row(i).setZero();
					if (gradients.rows() > 0)
						gradients.row(i).setZero();
					continue;
				}

				Eigen::MatrixXd local_value, local_gradient;
				io::Evaluator::interpolate_at_local_vals(
					*mesh_, dim, bases, geom_bases(),
					element_id, sample.local_points.row(i), solution,
					local_value, local_gradient);
				values.row(i) = local_value;
				if (gradients.rows() > 0)
					gradients.row(i) = local_gradient;
			}

			if (obstacle && obstacle->n_vertices() > 0
				&& sample.points.rows() == values.rows() + obstacle->n_vertices()
				&& sample.points.cols() == obstacle->v().cols()
				&& sample.points.bottomRows(obstacle->n_vertices()).isApprox(obstacle->v()))
			{
				values.conservativeResize(values.rows() + obstacle->n_vertices(), Eigen::NoChange);
				if (solution.rows() >= obstacle->ndof())
					values.bottomRows(obstacle->n_vertices()) =
						utils::unflatten(solution.bottomRows(obstacle->ndof()), dim);
				else
					values.bottomRows(obstacle->n_vertices()).setZero();
				if (gradients.rows() > 0)
				{
					gradients.conservativeResize(values.rows(), Eigen::NoChange);
					gradients.bottomRows(obstacle->n_vertices()).setZero();
				}
			}
		}
		else if (sample.node_ids.size() > 0)
		{
			values.resize(sample.node_ids.size(), dim);
			for (int i = 0; i < sample.node_ids.size(); ++i)
			{
				for (int d = 0; d < dim; ++d)
				{
					const int dof = sample.node_ids(i) * dim + d;
					if (dof < 0 || dof >= solution.rows())
						return;
					values(i, d) = solution(dof);
				}
			}
		}
		else
		{
			return;
		}

		if (sample.points.rows() > 0 && values.rows() != sample.points.rows())
			return;
		if (options.export_field("displacement"))
			fields.push_back({"displacement", values, io::OutputField::Association::Point});
		if (options.export_field("solution"))
			fields.push_back({"solution", values, io::OutputField::Association::Point});
		if (export_solution_gradient && gradients.rows() == values.rows())
			fields.push_back({"solution_gradient", gradients, io::OutputField::Association::Point});

		if (options.export_field("displaced_normals"))
		{
			Eigen::MatrixXd normals = displaced_output_normals(sample, solution);
			if (normals.rows() == values.rows())
				fields.push_back({"displaced_normals", normals, io::OutputField::Association::Point});
		}
	}

	Eigen::MatrixXd ElasticVarForm::displaced_output_normals(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution) const
	{
		if (!mesh_
			|| sample.normals.rows() == 0
			|| sample.normals.rows() != sample.local_points.rows()
			|| sample.local_points.rows() != sample.element_ids.size())
			return {};

		const int dim = mesh_->dimension();
		Eigen::MatrixXd displaced_normals = sample.normals;
		for (int i = 0; i < sample.local_points.rows(); ++i)
		{
			const int element_id = sample.element_ids(i);
			if (element_id < 0)
				continue;

			Eigen::MatrixXd local_value, local_gradient;
			io::Evaluator::interpolate_at_local_vals(
				*mesh_, dim, bases, geom_bases(),
				element_id, sample.local_points.row(i), solution,
				local_value, local_gradient);

			Eigen::MatrixXd deformation = Eigen::MatrixXd::Identity(dim, dim);
			for (int d = 0; d < dim; ++d)
				deformation.row(d) += local_gradient.block(0, d * dim, 1, dim);
			displaced_normals.row(i) = sample.normals.row(i) * deformation.inverse();
			displaced_normals.row(i).normalize();
		}
		return displaced_normals;
	}

	void ElasticVarForm::save_json(const Eigen::MatrixXd &solution, std::ostream &out) const
	{
		save_json_stats(solution, 0, out);
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

} // namespace polyfem::varform
