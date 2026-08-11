#include "NavierStokesFSIVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/assembler/NavierStokesFSI.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FSIInterfaceForm.hpp>
#include <polyfem/solver/forms/NavierStokesForm.hpp>
#include <polyfem/solver/forms/NavierStokesFSIForm.hpp>
#include <polyfem/solver/forms/StackedForm.hpp>
#include <polyfem/solver/forms/lagrangian/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/lagrangian/StackedAugmentedLagrangianForm.hpp>
#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/varforms/NonlinearElasticVarForm.hpp>

#include <igl/Timer.h>
#include <polysolve/linear/FEMSolver.hpp>
#include <polysolve/nonlinear/Solver.hpp>
#include <paraviewo/VTMWriter.hpp>
#include <spdlog/fmt/fmt.h>

#include <optional>
#include <map>
#include <set>

namespace polyfem::varform
{
	namespace
	{
		json first_material(const json &materials)
		{
			return materials.is_array() ? materials.at(0) : materials;
		}

		json mesh_material(const json &material)
		{
			json result = material.at("mesh_material");
			if (material.contains("id"))
				result["id"] = material["id"];
			return result;
		}

		json filter_fe_space_entries(const json &entries, const int fe_space_id)
		{
			if (!entries.is_array())
				return entries;

			json result = json::array();
			for (const json &entry : entries)
			{
				if (!entry.is_object())
				{
					result.push_back(entry);
					continue;
				}
				if (entry.contains("fe_space") && entry["fe_space"].get<int>() != fe_space_id)
					continue;
				json filtered = entry;
				filtered.erase("fe_space");
				result.push_back(std::move(filtered));
			}
			return result;
		}

		json residual_solver_params(const json &input)
		{
			json params = input;
			params["solver"] = "Newton";
			params["line_search"]["method"] = "ResidualBacktracking";
			if (!params.contains("Newton") || params["Newton"].is_null())
				params["Newton"] = json::object();
			params["Newton"]["force_psd_projection"] = false;
			params["Newton"]["use_psd_projection"] = true;
			return params;
		}

		StiffnessMatrix residual_mass(
			const StiffnessMatrix &velocity_mass,
			const int pressure_size,
			const StiffnessMatrix &mesh_mass,
			const StiffnessMatrix &solid_mass,
			const StiffnessMatrix &fluid_interface_mass,
			const StiffnessMatrix &mesh_interface_mass,
			const bool add_average)
		{
			const int mesh_offset = velocity_mass.rows() + pressure_size;
			const int solid_offset = mesh_offset + mesh_mass.rows();
			const int fluid_interface_offset = solid_offset + solid_mass.rows();
			const int mesh_interface_offset = fluid_interface_offset + fluid_interface_mass.rows();
			const int total = mesh_interface_offset + mesh_interface_mass.rows() + (add_average ? 1 : 0);
			std::vector<Eigen::Triplet<double>> entries;
			entries.reserve(velocity_mass.nonZeros() + pressure_size + mesh_mass.nonZeros()
							+ solid_mass.nonZeros() + fluid_interface_mass.nonZeros()
							+ mesh_interface_mass.nonZeros() + (add_average ? 1 : 0));
			for (int k = 0; k < velocity_mass.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(velocity_mass, k); it; ++it)
					entries.emplace_back(it.row(), it.col(), it.value());
			for (int i = 0; i < pressure_size; ++i)
				entries.emplace_back(velocity_mass.rows() + i, velocity_mass.rows() + i, 1);
			for (int k = 0; k < mesh_mass.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(mesh_mass, k); it; ++it)
					entries.emplace_back(mesh_offset + it.row(), mesh_offset + it.col(), it.value());
			for (int k = 0; k < solid_mass.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(solid_mass, k); it; ++it)
					entries.emplace_back(solid_offset + it.row(), solid_offset + it.col(), it.value());
			for (int k = 0; k < fluid_interface_mass.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(fluid_interface_mass, k); it; ++it)
					entries.emplace_back(fluid_interface_offset + it.row(), fluid_interface_offset + it.col(), it.value());
			for (int k = 0; k < mesh_interface_mass.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(mesh_interface_mass, k); it; ++it)
					entries.emplace_back(mesh_interface_offset + it.row(), mesh_interface_offset + it.col(), it.value());
			if (add_average)
				entries.emplace_back(total - 1, total - 1, 1);
			StiffnessMatrix result(total, total);
			result.setFromTriplets(entries.begin(), entries.end());
			result.makeCompressed();
			return result;
		}

		bool same_point(const RowVectorNd &a, const RowVectorNd &b)
		{
			return a.size() == b.size() && (a - b).norm() <= 1e-10;
		}

		int local_edge(const mesh::Mesh2D &mesh, const mesh::Navigation::Index &index)
		{
			for (int le = 0; le < mesh.n_face_vertices(index.face); ++le)
				if (mesh.get_index_from_face(index.face, le).edge == index.edge)
					return le;
			log_and_throw_error("Unable to locate interface edge {} in element {}.", index.edge, index.face);
		}

		std::map<int, Eigen::VectorXd> global_basis_values(
			const basis::ElementBases &bases, const Eigen::MatrixXd &points)
		{
			std::vector<assembler::AssemblyValues> values;
			bases.evaluate_bases(points, values);
			std::map<int, Eigen::VectorXd> result;
			for (int i = 0; i < int(values.size()); ++i)
				for (const auto &global : bases.bases[i].global())
				{
					auto it = result.try_emplace(
										global.index, Eigen::VectorXd::Zero(points.rows()))
								  .first;
					it->second += global.val * values[i].val.col(0);
				}
			return result;
		}

		struct ScalarTraceOperators
		{
			StiffnessMatrix source;
			StiffnessMatrix solid;
			std::vector<int> multiplier_source_ids;
		};

		struct VectorTraceOperators
		{
			StiffnessMatrix source;
			StiffnessMatrix solid;
		};

		ScalarTraceOperators assemble_2d_trace_operators(
			const mesh::Mesh2D &fluid_mesh,
			const mesh::Mesh2D &solid_mesh,
			const std::vector<std::pair<mesh::Navigation::Index, mesh::Navigation::Index>> &interface_pairs,
			const std::vector<basis::ElementBases> &source_bases,
			const int source_n_bases,
			const std::vector<basis::ElementBases> &solid_bases,
			const int solid_n_bases,
			const int quadrature_order)
		{
			ScalarTraceOperators result;
			std::map<int, int> multiplier_index;
			std::vector<Eigen::Triplet<double>> source_entries, solid_entries;
			for (const auto &[fluid_index, solid_index] : interface_pairs)
			{
				const int fluid_edge = local_edge(fluid_mesh, fluid_index);
				const int solid_edge = local_edge(solid_mesh, solid_index);
				const int fluid_vertices = fluid_mesh.n_face_vertices(fluid_index.face);
				const int solid_vertices = solid_mesh.n_face_vertices(solid_index.face);
				if ((fluid_vertices != 3 && fluid_vertices != 4)
					|| (solid_vertices != 3 && solid_vertices != 4))
					log_and_throw_error("FSI interface coupling currently supports triangular and quadrilateral 2D elements.");

				const RowVectorNd fluid_from = fluid_mesh.point(fluid_mesh.face_vertex(fluid_index.face, fluid_edge));
				const RowVectorNd fluid_to = fluid_mesh.point(fluid_mesh.face_vertex(fluid_index.face, (fluid_edge + 1) % fluid_vertices));
				const RowVectorNd solid_from = solid_mesh.point(solid_mesh.face_vertex(solid_index.face, solid_edge));
				const RowVectorNd solid_to = solid_mesh.point(solid_mesh.face_vertex(solid_index.face, (solid_edge + 1) % solid_vertices));
				const bool same_orientation = same_point(fluid_from, solid_from) && same_point(fluid_to, solid_to);
				const bool opposite_orientation = same_point(fluid_from, solid_to) && same_point(fluid_to, solid_from);
				if (!same_orientation && !opposite_orientation)
					log_and_throw_error(
						"FSI interface edge pair ({}, {}) is only partially overlapping; conforming facets are required.",
						fluid_index.edge, solid_index.edge);

				Eigen::MatrixXd uv, fluid_points;
				Eigen::VectorXd weights;
				if (fluid_vertices == 3)
					utils::BoundarySampler::quadrature_for_tri_edge(
						fluid_edge, quadrature_order, fluid_index.edge, fluid_mesh, uv, fluid_points, weights);
				else
					utils::BoundarySampler::quadrature_for_quad_edge(
						fluid_edge, quadrature_order, fluid_index.edge, fluid_mesh, uv, fluid_points, weights);

				const Eigen::Matrix2d solid_endpoints = solid_vertices == 3
															? utils::BoundarySampler::tri_local_node_coordinates_from_edge(solid_edge)
															: utils::BoundarySampler::quad_local_node_coordinates_from_edge(solid_edge);
				Eigen::MatrixXd solid_points(fluid_points.rows(), 2);
				for (int q = 0; q < solid_points.rows(); ++q)
				{
					const double t = uv(q, 1);
					if (same_orientation)
						solid_points.row(q) = (1 - t) * solid_endpoints.row(0) + t * solid_endpoints.row(1);
					else
						solid_points.row(q) = (1 - t) * solid_endpoints.row(1) + t * solid_endpoints.row(0);
				}

				const auto source_values = global_basis_values(source_bases.at(fluid_index.face), fluid_points);
				const auto solid_values = global_basis_values(solid_bases.at(solid_index.face), solid_points);
				for (const auto &[source_id, multiplier_values] : source_values)
				{
					if (multiplier_values.cwiseAbs().maxCoeff() < 1e-12)
						continue;
					const auto [it, inserted] = multiplier_index.try_emplace(source_id, multiplier_index.size());
					const int row = it->second;
					if (inserted)
						result.multiplier_source_ids.push_back(source_id);
					for (const auto &[trial_id, trial_values] : source_values)
					{
						const double value = (weights.array() * multiplier_values.array() * trial_values.array()).sum();
						if (std::abs(value) > 1e-14)
							source_entries.emplace_back(row, trial_id, value);
					}
					for (const auto &[trial_id, trial_values] : solid_values)
					{
						const double value = (weights.array() * multiplier_values.array() * trial_values.array()).sum();
						if (std::abs(value) > 1e-14)
							solid_entries.emplace_back(row, trial_id, value);
					}
				}
			}
			result.source.resize(multiplier_index.size(), source_n_bases);
			result.source.setFromTriplets(source_entries.begin(), source_entries.end());
			result.solid.resize(multiplier_index.size(), solid_n_bases);
			result.solid.setFromTriplets(solid_entries.begin(), solid_entries.end());
			return result;
		}

		VectorTraceOperators vector_trace(
			const ScalarTraceOperators &scalar,
			const int dim,
			const std::vector<int> &source_dirichlet_dofs)
		{
			assert(scalar.source.rows() == scalar.solid.rows());
			assert(scalar.source.rows() == int(scalar.multiplier_source_ids.size()));

			std::vector<bool> is_dirichlet(scalar.source.cols() * dim, false);
			for (const int dof : source_dirichlet_dofs)
				if (dof >= 0 && dof < int(is_dirichlet.size()))
					is_dirichlet[dof] = true;

			std::vector<int> vector_rows(scalar.source.rows() * dim, -1);
			int n_rows = 0;
			for (int row = 0; row < scalar.source.rows(); ++row)
				for (int d = 0; d < dim; ++d)
					if (!is_dirichlet[scalar.multiplier_source_ids[row] * dim + d])
						vector_rows[row * dim + d] = n_rows++;

			const auto expand = [&](const StiffnessMatrix &matrix) {
				std::vector<Eigen::Triplet<double>> entries;
				entries.reserve(matrix.nonZeros() * dim);
				for (int k = 0; k < matrix.outerSize(); ++k)
					for (StiffnessMatrix::InnerIterator it(matrix, k); it; ++it)
						for (int d = 0; d < dim; ++d)
						{
							const int row = vector_rows[it.row() * dim + d];
							if (row >= 0)
								entries.emplace_back(row, it.col() * dim + d, it.value());
						}
				StiffnessMatrix result(n_rows, matrix.cols() * dim);
				result.setFromTriplets(entries.begin(), entries.end());
				return result;
			};

			VectorTraceOperators result;
			result.source = expand(scalar.source);
			result.solid = expand(scalar.solid);
			return result;
		}
	} // namespace

	void NavierStokesFSIVarForm::reset()
	{
		FluidVarForm::reset();
		mesh_displacement_space_id_ = -1;
		displacement_space_id_ = -1;
		fluid_geometry_id_ = -1;
		solid_geometry_id_ = -1;
		has_solid_ = false;
		mesh_elastic_formulation_ = "NeoHookean";
		solid_elastic_formulation_ = "NeoHookean";
		solid_args_ = json();
		solid_varform_ = nullptr;
		interface_2d_.clear();
		interface_3d_.clear();
		mesh_displacement_space_.reset();
		mesh_displacement_boundary_.reset();
		mesh_displacement_problem_ = nullptr;
		mesh_displacement_ass_vals_cache_.init_empty();
		mesh_displacement_mass_ass_vals_cache_.init_empty(true);
		mesh_displacement_pure_mass_ass_vals_cache_.init_empty(true);
		mesh_elastic_assembler_ = nullptr;
		mesh_mass_assembler_ = nullptr;
		mesh_pure_mass_assembler_ = nullptr;
		mesh_rhs_assembler_ = nullptr;
		mesh_rhs_.resize(0, 0);
		fluid_zero_rhs_.resize(0, 0);
		mesh_pure_mass_.resize(0, 0);
		interface_velocity_trace_.resize(0, 0);
		interface_solid_velocity_trace_.resize(0, 0);
		interface_mesh_trace_.resize(0, 0);
		interface_solid_mesh_trace_.resize(0, 0);
		ale_assemblers_.clear();
		mesh_displacement_time_integrator_ = nullptr;
		fsi_forms_.clear();
		fsi_al_forms_.clear();
		fsi_problem_ = nullptr;
		ale_form_ = nullptr;
		interface_form_ = nullptr;
		auxiliary_form_ = nullptr;
		mesh_elastic_form_ = nullptr;
		fluid_neumann_form_ = nullptr;
		mesh_body_form_ = nullptr;
		average_pressure_form_ = nullptr;
	}

	void NavierStokesFSIVarForm::init(
		const std::string &formulation,
		const Units &units,
		const json &args,
		const std::string &out_path)
	{
		if (!args.contains("time") || args["time"].is_null())
			log_and_throw_error("NavierStokesFSI is only available for time-dependent problems.");
		FluidVarForm::init(formulation, units, args, out_path);

		const json &materials = args.at("materials");
		const json material = first_material(materials);
		mesh_displacement_space_id_ = material.at("mesh_displacement_space_id").get<int>();
		if (mesh_displacement_space_id_ == velocity_space_id_
			|| mesh_displacement_space_id_ == pressure_space_id_)
			log_and_throw_error("NavierStokesFSI requires distinct velocity, pressure, and mesh-displacement FE spaces.");
		mesh_elastic_formulation_ = material.at("mesh_material").at("type").get<std::string>();
		if (!assembler::AssemblerUtils::is_elastic_material(mesh_elastic_formulation_))
			log_and_throw_error("NavierStokesFSI mesh_material must be an elastic material, got {}.", mesh_elastic_formulation_);

		const std::array<std::string, 4> solid_fields{{"fluid_geometry_id", "solid_geometry_id", "displacement_space_id", "solid_material"}};
		int present_solid_fields = 0;
		for (const std::string &field : solid_fields)
			present_solid_fields += material.contains(field);
		if (present_solid_fields != 0 && present_solid_fields != int(solid_fields.size()))
			log_and_throw_error("Two-mesh NavierStokesFSI requires fluid_geometry_id, solid_geometry_id, displacement_space_id, and solid_material together.");
		has_solid_ = present_solid_fields == int(solid_fields.size());
		if (has_solid_)
		{
			fluid_geometry_id_ = material.at("fluid_geometry_id").get<int>();
			solid_geometry_id_ = material.at("solid_geometry_id").get<int>();
			displacement_space_id_ = material.at("displacement_space_id").get<int>();
			solid_elastic_formulation_ = material.at("solid_material").at("type").get<std::string>();
			if (!assembler::AssemblerUtils::is_elastic_material(solid_elastic_formulation_))
				log_and_throw_error("NavierStokesFSI solid_material must be an elastic material, got {}.", solid_elastic_formulation_);
			const std::set<int> ids{
				velocity_space_id_, pressure_space_id_, mesh_displacement_space_id_, displacement_space_id_};
			if (ids.size() != 4)
				log_and_throw_error("Two-mesh NavierStokesFSI requires four distinct FE-space IDs.");
			if (fluid_geometry_id_ == solid_geometry_id_)
				log_and_throw_error("Two-mesh NavierStokesFSI requires distinct fluid and solid geometry IDs.");
		}

		if (materials.is_array())
			for (const json &entry : materials)
			{
				if (entry.at("mesh_displacement_space_id").get<int>() != mesh_displacement_space_id_)
					log_and_throw_error("All NavierStokesFSI materials must use the same mesh-displacement FE space.");
				if (entry.at("mesh_material").at("type").get<std::string>() != mesh_elastic_formulation_)
					log_and_throw_error("All NavierStokesFSI regions must use the same mesh elastic formulation.");
				for (const std::string &field : solid_fields)
					if (entry.contains(field) != has_solid_)
						log_and_throw_error("All NavierStokesFSI materials must consistently enable the two-mesh solid fields.");
				if (has_solid_ && (entry.at("fluid_geometry_id") != fluid_geometry_id_ || entry.at("solid_geometry_id") != solid_geometry_id_ || entry.at("displacement_space_id") != displacement_space_id_ || entry.at("solid_material").at("type") != solid_elastic_formulation_))
					log_and_throw_error("All NavierStokesFSI materials must use the same geometry IDs, solid FE space, and solid formulation.");
			}

		if (args.at("space").at("discr_order").is_array())
		{
			bool found = false;
			for (const json &entry : args.at("space").at("discr_order"))
				found |= entry.at("fe_space").get<int>() == mesh_displacement_space_id_;
			if (!found)
				log_and_throw_error("NavierStokesFSI discretization orders must name the mesh-displacement FE space.");
		}

		mesh_elastic_assembler_ = assembler::AssemblerUtils::make_assembler(mesh_elastic_formulation_);
		mesh_mass_assembler_ = std::make_shared<assembler::Mass>();
		mesh_pure_mass_assembler_ = std::make_shared<assembler::HRZMass>();
		ale_assemblers_ = {
			std::make_shared<assembler::NavierStokesFSIVelocity>(),
			std::make_shared<assembler::NavierStokesFSIMixed>(),
			std::make_shared<assembler::NavierStokesFSIPressure>(),
			std::make_shared<assembler::NavierStokesFSIInertia>()};

		mesh_displacement_problem_ = std::make_shared<assembler::GenericTensorProblem>("NavierStokesFSIMeshDisplacement");
		mesh_displacement_problem_->clear();
		mesh_displacement_problem_->set_parameters({{"is_time_dependent", true}}, root_path);
		auto boundary_conditions = args["boundary_conditions"];
		boundary_conditions["root_path"] = root_path;
		mesh_displacement_problem_->set_parameters(boundary_conditions, root_path);
		mesh_displacement_problem_->set_parameters(args["initial_conditions"], root_path);
		mesh_displacement_problem_->set_parameters(args["output"], root_path);
		mesh_displacement_problem_->set_units(*mesh_elastic_assembler_, units);

		if (has_solid_)
		{
			solid_args_ = solid_varform_args();
			solid_varform_ = std::make_shared<NonlinearElasticTransientVarForm>();
			solid_varform_->init(solid_elastic_formulation_, units, solid_args_, out_path);
		}
	}

	json NavierStokesFSIVarForm::mesh_material_args() const
	{
		if (args["materials"].is_array())
		{
			json result = json::array();
			for (const json &material : args["materials"])
				result.push_back(mesh_material(material));
			return result;
		}
		return mesh_material(args["materials"]);
	}

	json NavierStokesFSIVarForm::solid_varform_args() const
	{
		json result = args;
		const json material = first_material(args.at("materials"));
		result["materials"] = material.at("solid_material");
		result.erase("preset_problem");

		if (result["space"]["discr_order"].is_array())
			result["space"]["discr_order"] = filter_fe_space_entries(
				result["space"]["discr_order"], displacement_space_id_);

		for (const char *key : {
				 "rhs", "dirichlet_boundary", "neumann_boundary",
				 "nodal_neumann_boundary", "normal_aligned_neumann_boundary"})
		{
			if (result["boundary_conditions"].contains(key))
				result["boundary_conditions"][key] = filter_fe_space_entries(
					result["boundary_conditions"][key], displacement_space_id_);
		}
		result["boundary_conditions"]["pressure_boundary"] = json::array();
		result["boundary_conditions"]["pressure_cavity"] = json::array();

		for (const char *key : {"solution", "velocity", "acceleration"})
			if (result["initial_conditions"].contains(key))
				result["initial_conditions"][key] = filter_fe_space_entries(
					result["initial_conditions"][key], displacement_space_id_);

		result["time"]["integrator"] = time_integrator_args(displacement_space_id_);
		result["constraints"]["hard"] = json::array();
		result["constraints"]["soft"] = json::array();
		result["space"]["remesh"]["enabled"] = false;

		result["output"]["advanced"]["timestep_prefix"] =
			"solid_" + result["output"]["advanced"]["timestep_prefix"].get<std::string>();
		return result;
	}

	json NavierStokesFSIVarForm::time_integrator_args(const int fe_space_id) const
	{
		const json &integrators = args["time"]["integrator"];
		if (!integrators.is_array())
			return integrators;
		for (const json &integrator : integrators)
			if (integrator.value("fe_space", -1) == fe_space_id)
			{
				json result = integrator;
				result.erase("fe_space");
				return result;
			}
		log_and_throw_error("Missing time integrator for FE space {}.", fe_space_id);
	}

	void NavierStokesFSIVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		if (has_solid_)
		{
			auto pieces = mesh.split();
			if (pieces.size() != 2)
				log_and_throw_error("Two-mesh NavierStokesFSI expected exactly two geometry partitions, got {}.", pieces.size());

			std::unique_ptr<mesh::Mesh> fluid_mesh, solid_mesh;
			for (auto &piece : pieces)
			{
				if (piece.id == fluid_geometry_id_)
					fluid_mesh = std::move(piece.mesh);
				else if (piece.id == solid_geometry_id_)
					solid_mesh = std::move(piece.mesh);
				else
					log_and_throw_error("Unexpected geometry ID {} in two-mesh NavierStokesFSI.", piece.id);
			}
			if (!fluid_mesh || !solid_mesh)
				log_and_throw_error(
					"Unable to find configured fluid/solid geometry IDs {}/{}.",
					fluid_geometry_id_, solid_geometry_id_);

			if (fluid_mesh->dimension() == 2)
			{
				interface_2d_ = mesh::compute_mesh_interface(
					dynamic_cast<const mesh::Mesh2D &>(*fluid_mesh),
					dynamic_cast<const mesh::Mesh2D &>(*solid_mesh));
				if (interface_2d_.empty())
					log_and_throw_error("Configured 2D fluid and solid geometries do not share an interface.");
			}
			else
			{
				interface_3d_ = mesh::compute_mesh_interface(
					dynamic_cast<const mesh::Mesh3D &>(*fluid_mesh),
					dynamic_cast<const mesh::Mesh3D &>(*solid_mesh));
				if (interface_3d_.empty())
					log_and_throw_error("Configured 3D fluid and solid geometries do not share an interface.");
			}

			mesh_ = std::move(fluid_mesh);
			solid_varform_->set_mesh(std::move(solid_mesh));
		}

		FluidVarForm::load_mesh(*mesh_, args);
		std::vector<int> body_ids(mesh_->n_elements());
		for (int e = 0; e < mesh_->n_elements(); ++e)
			body_ids[e] = mesh_->get_body_id(e);
		for (const auto &assembler : ale_assemblers_)
		{
			assembler->set_size(mesh_->dimension());
			assembler->set_materials(body_ids, this->args["materials"], units, root_path);
		}
		const json mesh_materials = mesh_material_args();
		mesh_elastic_assembler_->set_size(mesh_->dimension());
		mesh_elastic_assembler_->set_materials(body_ids, mesh_materials, units, root_path);
		mesh_mass_assembler_->set_size(mesh_->dimension());
		mesh_mass_assembler_->set_materials(body_ids, mesh_materials, units, root_path);
		mesh_pure_mass_assembler_->set_size(mesh_->dimension());
		mesh_displacement_problem_->init(*mesh_);
	}

	void NavierStokesFSIVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		FluidVarForm::build_basis(mesh, iso_parametric, args);
		// The interface traction multiplier can exchange a constant normal
		// traction with the fluid pressure. Keep an explicit physical-domain
		// pressure reference in the coupled problem even when an outer boundary
		// is marked Neumann.
		if (has_solid_)
			use_avg_pressure = true;
		Eigen::VectorXi orders, ordersq;
		assign_discr_orders(args["space"], mesh_displacement_space_id_, mesh, orders, ordersq);
		build_fe_space(
			mesh, iso_parametric, orders, ordersq,
			args["space"]["basis_type"], args["space"]["poly_basis_type"],
			*mesh_elastic_assembler_, mesh.dimension(),
			args["space"]["advanced"]["quadrature_order"],
			args["space"]["advanced"]["mass_quadrature_order"],
			args["space"]["advanced"]["use_corner_quadrature"],
			args["space"]["advanced"]["n_harmonic_samples"],
			args["space"]["advanced"]["integral_constraints"],
			mesh_displacement_space_, mesh_displacement_boundary_, space_.geometry);
		build_mesh_displacement_boundary(mesh);

		if (std::max({space_.n_bases, pressure_space_.n_bases, mesh_displacement_space_.n_bases})
			<= args["solver"]["advanced"]["cache_size"])
		{
			mesh_displacement_ass_vals_cache_.init(mesh.is_volume(), mesh_displacement_space_.basis_list(), space_.geometry_basis_list());
			mesh_displacement_mass_ass_vals_cache_.init(mesh.is_volume(), mesh_displacement_space_.basis_list(), space_.geometry_basis_list(), true);
			mesh_displacement_pure_mass_ass_vals_cache_.init(mesh.is_volume(), mesh_displacement_space_.basis_list(), space_.geometry_basis_list(), true);
		}
		else
		{
			mesh_displacement_ass_vals_cache_.init_empty();
			mesh_displacement_mass_ass_vals_cache_.init_empty(true);
			mesh_displacement_pure_mass_ass_vals_cache_.init_empty(true);
		}
		build_rhs_assembler();
		logger().info("n mesh displacement bases: {}", mesh_displacement_space_.n_bases);
		if (has_solid_)
		{
			solid_varform_->prepare_for_embedding();
			build_interface_operators();
			logger().info("n solid displacement dofs: {}", solid_varform_->embedding_ndof());
			logger().info(
				"n FSI interface multiplier dofs: physical={}, mesh={}",
				fluid_interface_multiplier_ndof(), mesh_interface_multiplier_ndof());
		}
	}

	void NavierStokesFSIVarForm::build_interface_operators()
	{
		assert(has_solid_ && solid_varform_ && mesh_);
		if (mesh_->dimension() != 2)
			log_and_throw_error("Two-mesh FSI interface coupling currently supports 2D meshes.");
		const io::OutputSpace solid_output = solid_varform_->output_space();
		assert(solid_output.mesh);
		const auto &fluid_mesh = dynamic_cast<const mesh::Mesh2D &>(*mesh_);
		const auto &solid_mesh = dynamic_cast<const mesh::Mesh2D &>(*solid_output.mesh);
		const FESpace &solid_space = solid_varform_->embedding_space();
		const int order = 2 * std::max({space_.disc_orders.maxCoeff(), mesh_displacement_space_.disc_orders.maxCoeff(), solid_space.disc_orders.maxCoeff()}) + 2;

		const ScalarTraceOperators physical = assemble_2d_trace_operators(
			fluid_mesh, solid_mesh, interface_2d_,
			space_.basis_list(), space_.n_bases,
			solid_space.basis_list(), solid_space.n_bases, order);
		const ScalarTraceOperators computational = assemble_2d_trace_operators(
			fluid_mesh, solid_mesh, interface_2d_,
			mesh_displacement_space_.basis_list(), mesh_displacement_space_.n_bases,
			solid_space.basis_list(), solid_space.n_bases, order);
		const VectorTraceOperators physical_vector =
			vector_trace(physical, mesh_->dimension(), boundary_.boundary_nodes);
		const VectorTraceOperators computational_vector =
			vector_trace(computational, mesh_->dimension(), mesh_displacement_boundary_.boundary_nodes);
		interface_velocity_trace_ = physical_vector.source;
		interface_solid_velocity_trace_ = physical_vector.solid;
		interface_mesh_trace_ = computational_vector.source;
		interface_solid_mesh_trace_ = computational_vector.solid;
		if (interface_velocity_trace_.rows() == 0 || interface_mesh_trace_.rows() == 0)
			log_and_throw_error("The fluid-solid interface has no active FE trace degrees of freedom.");
	}

	void NavierStokesFSIVarForm::build_mesh_displacement_boundary(mesh::Mesh &mesh)
	{
		mesh_displacement_boundary_.clear_boundary_conditions();
		mesh_displacement_problem_->update_nodes(mesh_displacement_space_.space_in_node_to_node);
		mesh_displacement_problem_->setup_bc(
			mesh, assembler::BoundaryKind::Dirichlet, mesh_displacement_space_id_,
			mesh_displacement_space_.basis_list(), mesh_displacement_boundary_.total_local_boundary,
			mesh_displacement_boundary_.local_boundary, mesh_displacement_boundary_.boundary_nodes,
			mesh.dimension());
		std::vector<int> unused;
		mesh_displacement_problem_->setup_bc(
			mesh, assembler::BoundaryKind::Neumann, mesh_displacement_space_id_,
			mesh_displacement_space_.basis_list(), mesh_displacement_boundary_.total_local_boundary,
			mesh_displacement_boundary_.local_neumann_boundary, unused, mesh.dimension());
		mesh_displacement_problem_->setup_nodal_bc(mesh, assembler::BoundaryKind::Dirichlet, mesh_displacement_space_id_, mesh_displacement_space_.n_bases, mesh_displacement_boundary_.dirichlet_nodes);
		mesh_displacement_problem_->setup_nodal_bc(mesh, assembler::BoundaryKind::Neumann, mesh_displacement_space_id_, mesh_displacement_space_.n_bases, mesh_displacement_boundary_.neumann_nodes);
		for (const int node : mesh_displacement_boundary_.dirichlet_nodes)
		{
			const int tag = mesh.get_node_id(node);
			for (int d = 0; d < mesh.dimension(); ++d)
				if (mesh_displacement_problem_->is_nodal_dimension_dirichlet(node, tag, d, mesh_displacement_space_id_))
					mesh_displacement_boundary_.boundary_nodes.push_back(node * mesh.dimension() + d);
		}
		mesh_displacement_boundary_.normalize_boundary_nodes();
		rebuild_node_positions(mesh_displacement_space_.basis_list(), mesh_displacement_boundary_.dirichlet_nodes, mesh_displacement_boundary_.dirichlet_nodes_position);
		rebuild_node_positions(mesh_displacement_space_.basis_list(), mesh_displacement_boundary_.neumann_nodes, mesh_displacement_boundary_.neumann_nodes_position);
	}

	void NavierStokesFSIVarForm::build_rhs_assembler()
	{
		FluidVarForm::build_rhs_assembler();
		if (mesh_displacement_space_.n_bases <= 0 || !mesh_)
			return;
		json solver_params = args["solver"]["linear"];
		if (!solver_params.contains("Pardiso"))
			solver_params["Pardiso"] = {};
		solver_params["Pardiso"]["mtype"] = -2;
		mesh_rhs_assembler_ = std::make_shared<assembler::RhsAssembler>(
			*mesh_elastic_assembler_, *mesh_, nullptr,
			mesh_displacement_boundary_.dirichlet_nodes, mesh_displacement_boundary_.neumann_nodes,
			mesh_displacement_boundary_.dirichlet_nodes_position, mesh_displacement_boundary_.neumann_nodes_position,
			mesh_displacement_space_.n_bases, mesh_->dimension(),
			mesh_displacement_space_.basis_list(), space_.geometry_basis_list(),
			mesh_displacement_mass_ass_vals_cache_, *mesh_displacement_problem_,
			args["space"]["advanced"]["bc_method"], solver_params,
			mesh_displacement_space_id_);
	}

	void NavierStokesFSIVarForm::assemble_rhs(const mesh::Mesh &mesh)
	{
		FluidVarForm::assemble_rhs(mesh);
		assert(mesh_rhs_assembler_);
		mesh_rhs_assembler_->assemble(mesh_mass_assembler_->density(), mesh_rhs_);
		mesh_rhs_ *= -1;
		const Eigen::MatrixXd velocity_rhs = rhs_.topRows(primary_ndof());
		rhs_.setZero(total_ndof(), 1);
		rhs_.topRows(primary_ndof()) = velocity_rhs;
		rhs_.middleRows(mesh_displacement_offset(), mesh_displacement_ndof()) = mesh_rhs_;
	}

	void NavierStokesFSIVarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
	{
		FluidVarForm::assemble_mass_mat(mesh, args);
		mesh_pure_mass_assembler_->assemble(
			mesh.is_volume(), mesh_displacement_space_.n_bases,
			mesh_displacement_space_.basis_list(), space_.geometry_basis_list(),
			mesh_displacement_pure_mass_ass_vals_cache_, 0, mesh_pure_mass_, true);
	}

	int NavierStokesFSIVarForm::mesh_displacement_ndof() const
	{
		return mesh_ ? mesh_displacement_space_.n_bases * mesh_->dimension() : 0;
	}

	int NavierStokesFSIVarForm::solid_displacement_ndof() const
	{
		return has_solid_ && solid_varform_ ? solid_varform_->embedding_ndof() : 0;
	}

	int NavierStokesFSIVarForm::fluid_interface_multiplier_ndof() const
	{
		return has_solid_ ? interface_velocity_trace_.rows() : 0;
	}

	int NavierStokesFSIVarForm::mesh_interface_multiplier_ndof() const
	{
		return has_solid_ ? interface_mesh_trace_.rows() : 0;
	}

	int NavierStokesFSIVarForm::total_ndof() const
	{
		return primary_ndof() + pressure_space_.n_bases + mesh_displacement_ndof()
			   + solid_displacement_ndof() + fluid_interface_multiplier_ndof()
			   + mesh_interface_multiplier_ndof() + (use_avg_pressure ? 1 : 0);
	}

	void NavierStokesFSIVarForm::prepare_fsi_initial_solution(Eigen::MatrixXd &sol) const
	{
		if (sol.size() == 0)
		{
			Eigen::MatrixXd velocity, mesh_displacement, solid_displacement;
			const std::string state_path = resolve_input_path(args["input"]["data"]["state"]);
			const bool loaded_velocity = read_initial_x_from_file(
				state_path, "u", args["input"]["data"]["reorder"],
				space_.space_in_node_to_node, mesh_->dimension(), velocity);
			const bool loaded_mesh_displacement = read_initial_x_from_file(
				state_path, "mesh_u", args["input"]["data"]["reorder"],
				mesh_displacement_space_.space_in_node_to_node, mesh_->dimension(), mesh_displacement);
			if (!loaded_velocity)
				rhs_assembler_->initial_solution(velocity);
			if (!loaded_mesh_displacement)
				mesh_rhs_assembler_->initial_solution(mesh_displacement);
			if (has_solid_)
				solid_varform_->initial_solution_for_embedding(solid_displacement, "solid_");
			sol.setZero(total_ndof(), 1);
			sol.topRows(primary_ndof()) = velocity.topRows(primary_ndof()).leftCols(1);
			sol.middleRows(mesh_displacement_offset(), mesh_displacement_ndof()) =
				mesh_displacement.topRows(mesh_displacement_ndof()).leftCols(1);
			if (has_solid_)
				sol.middleRows(solid_displacement_offset(), solid_displacement_ndof()) =
					solid_displacement.topRows(solid_displacement_ndof()).leftCols(1);
		}
		else
		{
			if (sol.cols() > 1)
				sol.conservativeResize(Eigen::NoChange, 1);
			if (sol.rows() != total_ndof())
			{
				const Eigen::MatrixXd input = sol;
				sol.setZero(total_ndof(), 1);
				const int rows = std::min<int>(input.rows(), total_ndof());
				if (rows > 0)
					sol.topRows(rows) = input.topRows(rows);
			}
		}
	}

	void NavierStokesFSIVarForm::build_forms(Eigen::MatrixXd &sol, const double t)
	{
		const int dim = mesh_->dimension();
		const Eigen::VectorXd velocity = sol.topRows(primary_ndof());
		const Eigen::VectorXd mesh_displacement = sol.middleRows(mesh_displacement_offset(), mesh_displacement_ndof());
		Eigen::MatrixXd solid_displacement;
		if (has_solid_)
		{
			solid_displacement = sol.middleRows(solid_displacement_offset(), solid_displacement_ndof());
			solid_varform_->init_forms_for_embedding(solid_displacement, t, "solid_");
		}

		auto velocity_bdf = time_integrator::ImplicitTimeIntegrator::construct_bdf_integrator(
			time_integrator_args(velocity_space_id_), time_integrator::ImplicitTimeIntegrator::DynamicOrder::First);
		auto mesh_bdf = time_integrator::ImplicitTimeIntegrator::construct_bdf_integrator(
			time_integrator_args(mesh_displacement_space_id_), time_integrator::ImplicitTimeIntegrator::DynamicOrder::First);
		Eigen::MatrixXd velocity_initial_velocity, mesh_initial_velocity;
		rhs_assembler_->initial_velocity(velocity_initial_velocity);
		mesh_rhs_assembler_->initial_velocity(mesh_initial_velocity);
		Eigen::MatrixXd velocity_history = velocity;
		Eigen::MatrixXd velocity_history_velocity = velocity_initial_velocity;
		Eigen::MatrixXd velocity_history_acceleration = Eigen::MatrixXd::Zero(primary_ndof(), 1);
		Eigen::MatrixXd mesh_history = mesh_displacement;
		Eigen::MatrixXd mesh_history_velocity = mesh_initial_velocity;
		Eigen::MatrixXd mesh_history_acceleration = Eigen::MatrixXd::Zero(mesh_displacement_ndof(), 1);
		const std::string state_path = resolve_input_path(args["input"]["data"]["state"]);
		if (read_initial_x_from_file(
				state_path, "u", args["input"]["data"]["reorder"],
				space_.space_in_node_to_node, dim, velocity_history))
		{
			if (!read_initial_x_from_file(
					state_path, "v", args["input"]["data"]["reorder"],
					space_.space_in_node_to_node, dim, velocity_history_velocity))
				velocity_history_velocity.setZero(velocity_history.rows(), velocity_history.cols());
			if (!read_initial_x_from_file(
					state_path, "a", args["input"]["data"]["reorder"],
					space_.space_in_node_to_node, dim, velocity_history_acceleration))
				velocity_history_acceleration.setZero(velocity_history.rows(), velocity_history.cols());
		}
		if (read_initial_x_from_file(
				state_path, "mesh_u", args["input"]["data"]["reorder"],
				mesh_displacement_space_.space_in_node_to_node, dim, mesh_history))
		{
			if (!read_initial_x_from_file(
					state_path, "mesh_v", args["input"]["data"]["reorder"],
					mesh_displacement_space_.space_in_node_to_node, dim, mesh_history_velocity))
				mesh_history_velocity.setZero(mesh_history.rows(), mesh_history.cols());
			if (!read_initial_x_from_file(
					state_path, "mesh_a", args["input"]["data"]["reorder"],
					mesh_displacement_space_.space_in_node_to_node, dim, mesh_history_acceleration))
				mesh_history_acceleration.setZero(mesh_history.rows(), mesh_history.cols());
		}
		velocity_bdf->init(velocity_history, velocity_history_velocity, velocity_history_acceleration, dt);
		mesh_bdf->init(mesh_history, mesh_history_velocity, mesh_history_acceleration, dt);
		time_integrator = velocity_bdf;
		mesh_displacement_time_integrator_ = mesh_bdf;

		ale_form_ = std::make_shared<solver::NavierStokesFSIForm>(
			total_ndof(), space_.n_bases, pressure_space_.n_bases, mesh_displacement_space_.n_bases,
			space_.basis_list(), pressure_space_.basis_list(), mesh_displacement_space_.basis_list(),
			space_.geometry_basis_list(), ass_vals_cache_, pressure_ass_vals_cache_, mesh_displacement_ass_vals_cache_,
			ale_assemblers_, time_integrator.get(), mesh_displacement_time_integrator_.get(),
			t, dt, mesh_->is_volume(),
			[this](const int element, const Eigen::MatrixXd &points, const double time, Eigen::MatrixXd &value) {
				problem->rhs(*primary_assembler_, *mesh_, element, points, time, value, velocity_space_id_);
			});
		const int gorder = mesh_->orders().size() == 0 ? 1 : mesh_->orders().maxCoeff();
		const QuadratureOrders velocity_samples = n_boundary_samples(
			space_.disc_orders.maxCoeff(), space_.disc_ordersq.maxCoeff(), gorder);
		ale_form_->set_velocity_tilde_updater(
			[this, velocity_samples](const double time, const Eigen::VectorXd &, Eigen::VectorXd &target) {
				Eigen::MatrixXd projected = target;
				const std::vector<mesh::LocalBoundary> empty_neumann;
				rhs_assembler_->set_bc(boundary_.local_boundary, boundary_.boundary_nodes, velocity_samples, empty_neumann, projected, Eigen::MatrixXd(), time);
				target = projected.col(0);
			});

		auxiliary_form_ = std::make_shared<solver::StackedForm>();
		const auto velocity_block = auxiliary_form_->add_block(primary_ndof());
		auxiliary_form_->add_block(pressure_space_.n_bases);
		const auto mesh_block = auxiliary_form_->add_block(mesh_displacement_ndof());
		std::optional<solver::StackedForm::Block> solid_block;
		if (has_solid_)
		{
			solid_block = auxiliary_form_->add_block(solid_displacement_ndof());
			for (const auto &form : solid_varform_->embedding_forms())
				auxiliary_form_->add(*solid_block, form);
			auxiliary_form_->add_block(fluid_interface_multiplier_ndof());
			auxiliary_form_->add_block(mesh_interface_multiplier_ndof());
		}

		const solver::ElementInversionCheck check = args["solver"]["advanced"]["check_inversion"];
		mesh_elastic_form_ = std::make_shared<solver::ElasticForm>(
			mesh_displacement_space_.n_bases, *mesh_displacement_space_.bases, space_.geometry_basis_list(),
			*mesh_elastic_assembler_, mesh_displacement_ass_vals_cache_, t, dt, mesh_->is_volume(),
			args["solver"]["advanced"]["jacobian_threshold"], check);
		auxiliary_form_->add(mesh_block, mesh_elastic_form_);

		fluid_zero_rhs_ = Eigen::MatrixXd::Zero(primary_ndof(), 1);
		fluid_neumann_form_ = std::make_shared<solver::BodyForm>(
			primary_ndof(), 0, boundary_.boundary_nodes, boundary_.local_boundary,
			boundary_.local_neumann_boundary, velocity_samples, fluid_zero_rhs_, *rhs_assembler_,
			mass_assembler_->density(), false, true);
		fluid_neumann_form_->update_quantities(t, velocity);
		auxiliary_form_->add(velocity_block, fluid_neumann_form_);

		const QuadratureOrders mesh_samples = n_boundary_samples(
			mesh_displacement_space_.disc_orders.maxCoeff(),
			mesh_displacement_space_.disc_ordersq.maxCoeff(), gorder);
		mesh_body_form_ = std::make_shared<solver::BodyForm>(
			mesh_displacement_ndof(), 0,
			mesh_displacement_boundary_.boundary_nodes, mesh_displacement_boundary_.local_boundary,
			mesh_displacement_boundary_.local_neumann_boundary, mesh_samples,
			mesh_rhs_, *mesh_rhs_assembler_, mesh_mass_assembler_->density(), false, true);
		mesh_body_form_->update_quantities(t, mesh_displacement);
		auxiliary_form_->add(mesh_block, mesh_body_form_);

		if (use_avg_pressure)
		{
			auxiliary_form_->add_block(1);
			average_pressure_form_ = std::make_shared<solver::NavierStokesFSIAveragePressureForm>(
				total_ndof(), space_.n_bases, pressure_space_.n_bases,
				mesh_displacement_space_.n_bases, average_pressure_offset(), dim,
				pressure_space_.basis_list(), mesh_displacement_space_.basis_list(),
				space_.geometry_basis_list(), pressure_ass_vals_cache_,
				mesh_displacement_ass_vals_cache_, mesh_->is_volume());
		}
		else
			average_pressure_form_ = nullptr;

		if (has_solid_)
		{
			interface_form_ = std::make_shared<solver::FSIInterfaceForm>(
				total_ndof(), 0, mesh_displacement_offset(), solid_displacement_offset(),
				fluid_interface_multiplier_offset(), mesh_interface_multiplier_offset(),
				interface_velocity_trace_, interface_solid_velocity_trace_,
				interface_mesh_trace_, interface_solid_mesh_trace_,
				*time_integrator, *solid_varform_->embedding_time_integrator());
		}
		else
			interface_form_ = nullptr;

		fsi_forms_ = {ale_form_, auxiliary_form_};
		if (interface_form_)
			fsi_forms_.push_back(interface_form_);
		if (average_pressure_form_)
			fsi_forms_.push_back(average_pressure_form_);
		for (const auto &form : fsi_forms_)
			form->set_output_dir(output_path);
		fsi_al_forms_.clear();
		if (!boundary_.boundary_nodes.empty() || !mesh_displacement_boundary_.boundary_nodes.empty()
			|| (has_solid_ && !solid_varform_->embedding_al_forms().empty()))
		{
			auto stacked_al = std::make_shared<solver::StackedAugmentedLagrangianForm>();
			const auto velocity_al = stacked_al->add_block(primary_ndof());
			stacked_al->add_block(pressure_space_.n_bases);
			const auto mesh_al = stacked_al->add_block(mesh_displacement_ndof());
			std::optional<solver::StackedAugmentedLagrangianForm::Block> solid_al;
			if (has_solid_)
			{
				solid_al = stacked_al->add_block(solid_displacement_ndof());
				stacked_al->add_block(fluid_interface_multiplier_ndof());
				stacked_al->add_block(mesh_interface_multiplier_ndof());
			}
			if (use_avg_pressure)
				stacked_al->add_block(1);
			if (!boundary_.boundary_nodes.empty())
				stacked_al->add(velocity_al, std::make_shared<solver::BCLagrangianForm>(
												 primary_ndof(), boundary_.boundary_nodes, boundary_.local_boundary, boundary_.local_neumann_boundary,
												 velocity_samples, pure_mass_, *rhs_assembler_, 0, true, t));
			if (!mesh_displacement_boundary_.boundary_nodes.empty())
				stacked_al->add(mesh_al, std::make_shared<solver::BCLagrangianForm>(
											 mesh_displacement_ndof(), mesh_displacement_boundary_.boundary_nodes,
											 mesh_displacement_boundary_.local_boundary, mesh_displacement_boundary_.local_neumann_boundary,
											 mesh_samples, mesh_pure_mass_, *mesh_rhs_assembler_, 0, true, t));
			if (has_solid_)
				for (const auto &form : solid_varform_->embedding_al_forms())
					stacked_al->add(*solid_al, form);
			fsi_al_forms_.push_back(stacked_al);
		}

		fsi_problem_ = std::make_shared<solver::NLProblem>(
			total_ndof(), t, fsi_forms_, fsi_al_forms_,
			polysolve::linear::Solver::create(args["solver"]["linear"], logger()),
			units.characteristic_length(), 1,
			residual_mass(
				pure_mass_, pressure_space_.n_bases, mesh_pure_mass_,
				has_solid_ ? solid_varform_->embedding_norm_matrix() : StiffnessMatrix(),
				has_solid_ ? interface_form_->fluid_multiplier_mass() : StiffnessMatrix(),
				has_solid_ ? interface_form_->mesh_multiplier_mass() : StiffnessMatrix(),
				use_avg_pressure),
			dim, true);
		fsi_problem_->init(sol);
		fsi_problem_->update_quantities(t, sol);
		update_transient_form_weights();
		stats.solver_info = json::array();
	}

	void NavierStokesFSIVarForm::update_transient_form_weights()
	{
		const double scale = time_integrator->acceleration_scaling();
		if (fluid_neumann_form_)
			fluid_neumann_form_->set_weight(scale);
		if (average_pressure_form_)
			average_pressure_form_->set_weight(scale);
	}

	void NavierStokesFSIVarForm::solve_nonlinear_step(const int step, Eigen::MatrixXd &sol)
	{
		const json nonlinear_params = residual_solver_params(args["solver"]["nonlinear"]);
		const json al_params = residual_solver_params(args["solver"]["augmented_lagrangian"]["nonlinear"]);
		std::shared_ptr<polysolve::nonlinear::Solver> nonlinear_solver = polysolve::nonlinear::Solver::create(
			nonlinear_params, args["solver"]["linear"], units.characteristic_length(), logger());
		solver::ALSolver al_solver(
			fsi_al_forms_, args["solver"]["augmented_lagrangian"]["initial_weight"],
			args["solver"]["augmented_lagrangian"]["scaling"],
			args["solver"]["augmented_lagrangian"]["max_weight"],
			args["solver"]["augmented_lagrangian"]["eta"],
			[this](const Eigen::VectorXd &x) {
				if (has_solid_)
					solid_varform_->update_barrier_stiffness_for_embedding(
						x.segment(solid_displacement_offset(), solid_displacement_ndof()));
			});
		al_solver.post_subsolve = [&](const double weight) {
			stats.solver_info.push_back({{"type", weight > 0 ? "al" : "rc"}, {"t", step}, {"info", nonlinear_solver->info()}});
			if (weight > 0)
				stats.solver_info.back()["weight"] = weight;
			save_subsolve(stats.solver_info.size(), step, sol);
		};
		if (!fsi_al_forms_.empty())
			al_solver.solve_al(*fsi_problem_, sol, al_params, args["solver"]["linear"], units.characteristic_length(), nonlinear_solver);
		al_solver.solve_reduced(*fsi_problem_, sol, nonlinear_params, args["solver"]["linear"], units.characteristic_length(), nonlinear_solver);
	}

	void NavierStokesFSIVarForm::solve_problem(
		Eigen::MatrixXd &sol,
		const InitialConditionOverride *initial_condition_override,
		const ForwardStepCallback &post_step)
	{
		assert(!initial_condition_override && "Navier-Stokes FSI does not support initial-condition overrides");
		assert(!post_step && "Navier-Stokes FSI does not support post-step callbacks");

		igl::Timer timer;
		timer.start();
		prepare_fsi_initial_solution(sol);
		build_forms(sol, t0 + dt);
		save_fsi_timestep(t0, 0, sol);
		for (int step = 1; step <= time_steps; ++step)
		{
			const double time = t0 + step * dt;
			logger().info("{}/{} steps, dt={}s t={}s", step, time_steps, dt, time);
			solve_nonlinear_step(step, sol);
			time_integrator->update_quantities(sol.topRows(primary_ndof()));
			mesh_displacement_time_integrator_->update_quantities(
				sol.middleRows(mesh_displacement_offset(), mesh_displacement_ndof()));
			if (has_solid_)
				solid_varform_->advance_for_embedding(
					sol.middleRows(solid_displacement_offset(), solid_displacement_ndof()));
			update_transient_form_weights();
			fsi_problem_->update_quantities(t0 + (step + 1) * dt, sol);
			save_fsi_timestep(time, step, sol);
			save_step_state(t0, dt, step, time_integrator.get());
			save_mesh_integrator_state(step);
			if (has_solid_)
				save_solid_integrator_state(step);
			notify_time_step(step, time_steps, t0, dt);
		}
		timer.stop();
		timings.solving_time = timer.getElapsedTime();
	}

	void NavierStokesFSIVarForm::save_fsi_timestep(
		const double time, const int step, const Eigen::MatrixXd &solution) const
	{
		if (!has_solid_)
		{
			save_timestep(time, step, t0, dt, solution);
			return;
		}

		paraviewo::VTMWriter vtm(time);
		const bool fluid_saved = save_timestep_to_vtm(time, step, dt, solution, vtm, "Fluid");
		const bool solid_saved = solid_varform_->save_timestep_for_embedding(
			time, step, dt,
			solution.middleRows(solid_displacement_offset(), solid_displacement_ndof()),
			vtm, "Solid");
		if (!fluid_saved && !solid_saved)
			return;

		const int global_t = output_file_index(step);
		const std::string step_name = args["output"]["advanced"]["timestep_prefix"];
		vtm.save(resolve_output_path(fmt::format(step_name + "{:d}.vtm", global_t)));
		output_geometry_.save_pvd(
			resolve_output_path(args["output"]["paraview"]["file_name"]),
			[step_name](int i) { return fmt::format(step_name + "{:d}.vtm", i); },
			global_t, t0, dt, args["output"]["paraview"]["skip_frame"].get<int>());
	}

	void NavierStokesFSIVarForm::save_mesh_integrator_state(const int step) const
	{
		assert(mesh_displacement_time_integrator_);
		const std::string state_path = resolve_output_path(
			fmt::format(args["output"]["data"]["state"].get<std::string>(), output_file_index(step)));
		if (state_path.empty())
			return;

		const auto save_history = [&](const std::string &name, const std::deque<Eigen::VectorXd> &history) {
			Eigen::MatrixXd values(history.front().size(), history.size());
			for (int i = 0; i < int(history.size()); ++i)
				values.col(i) = history[i];
			io::write_matrix(state_path, name, values, /*replace=*/false);
		};
		save_history("mesh_u", mesh_displacement_time_integrator_->x_prevs());
		save_history("mesh_v", mesh_displacement_time_integrator_->v_prevs());
		save_history("mesh_a", mesh_displacement_time_integrator_->a_prevs());
	}

	void NavierStokesFSIVarForm::save_solid_integrator_state(const int step) const
	{
		assert(has_solid_ && solid_varform_->embedding_time_integrator());
		const std::string state_path = resolve_output_path(
			fmt::format(args["output"]["data"]["state"].get<std::string>(), output_file_index(step)));
		if (state_path.empty())
			return;

		const auto save_history = [&](const std::string &name, const std::deque<Eigen::VectorXd> &history) {
			Eigen::MatrixXd values(history.front().size(), history.size());
			for (int i = 0; i < int(history.size()); ++i)
				values.col(i) = history[i];
			io::write_matrix(state_path, name, values, /*replace=*/false);
		};
		const auto &integrator = solid_varform_->embedding_time_integrator();
		save_history("solid_u", integrator->x_prevs());
		save_history("solid_v", integrator->v_prevs());
		save_history("solid_a", integrator->a_prevs());
	}

	std::vector<io::OutputField> NavierStokesFSIVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		std::vector<io::OutputField> fields = FluidVarForm::output_fields(sample, solution, options);
		if (!mesh_ || solution.rows() < mesh_displacement_offset() + mesh_displacement_ndof()
			|| !options.export_field("mesh_displacement"))
			return fields;

		const int dim = mesh_->dimension();
		const Eigen::MatrixXd mesh_displacement =
			solution.middleRows(mesh_displacement_offset(), mesh_displacement_ndof());
		const bool has_element_samples =
			sample.local_points.rows() > 0 && sample.local_points.rows() == sample.element_ids.size();
		const int output_rows = sample.points.rows() > 0
									? sample.points.rows()
									: std::max<int>(sample.local_points.rows(), sample.node_ids.size());
		Eigen::MatrixXd values;

		if (has_element_samples)
		{
			values.setZero(output_rows, dim);
			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element = sample.element_ids(i);
				if (element < 0)
					continue;
				Eigen::MatrixXd local_value, local_gradient;
				io::Evaluator::interpolate_at_local_vals(
					*mesh_, dim,
					mesh_displacement_space_.basis_list(), space_.geometry_basis_list(),
					element, sample.local_points.row(i), mesh_displacement,
					local_value, local_gradient);
				for (int d = 0; d < dim; ++d)
					values(i, d) = local_value(d);
			}
		}
		else if (sample.node_ids.size() > 0)
		{
			values.resize(sample.node_ids.size(), dim);
			for (int i = 0; i < sample.node_ids.size(); ++i)
			{
				const int node = sample.node_ids(i);
				if (node < 0 || node * dim + dim > mesh_displacement.rows())
					return fields;
				values.row(i) = mesh_displacement.block(node * dim, 0, dim, 1).transpose();
			}
		}
		else
		{
			return fields;
		}

		fields.push_back({"mesh_displacement", values, io::OutputField::Association::Point});
		return fields;
	}
} // namespace polyfem::varform
