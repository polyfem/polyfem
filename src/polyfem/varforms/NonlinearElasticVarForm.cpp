#include "NonlinearElasticVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/MacroStrain.hpp>

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/mesh/collision_proxy/CollisionProxy.hpp>
#include <polyfem/mesh/GeometryReader.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Jacobian.hpp>

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>
#include <polyfem/autogen/prism_bases.hpp>

#include <polyfem/utils/BoundarySampler.hpp>

#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <igl/Timer.h>
#include <igl/edges.h>

#include <ipc/ipc.hpp>

#include <polysolve/linear/Solver.hpp>
#include <polysolve/nonlinear/Solver.hpp>

#include <algorithm>

namespace polyfem::varform
{
	using namespace solver;
	using namespace time_integrator;

	namespace
	{
		void copy_local_boundaries(
			const std::vector<mesh::LocalBoundary> &from,
			std::vector<mesh::LocalBoundary> &to)
		{
			to.clear();
			to.reserve(from.size());
			for (const auto &lb : from)
				to.emplace_back(lb);
		}

		void copy_local_boundary_map(
			const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &from,
			std::unordered_map<int, std::vector<mesh::LocalBoundary>> &to)
		{
			to.clear();
			to.reserve(from.size());
			for (const auto &[id, boundaries] : from)
			{
				auto &dst = to[id];
				copy_local_boundaries(boundaries, dst);
			}
		}

	} // namespace

	void NonlinearElasticVarForm::reset()
	{
		ElasticVarForm::reset();
		collision_mesh = ipc::CollisionMesh();
		elasticity_pressure_assembler = nullptr;
		damping_assembler = nullptr;
		damping_prev_assembler = nullptr;
	}

	void NonlinearElasticVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		ElasticVarForm::load_mesh(mesh, args);

		logger().info("Loading obstacles...");
		obstacle = mesh::read_obstacle_geometry(
			units,
			args["geometry"],
			utils::json_as_array(args["boundary_conditions"]["obstacle_displacements"]),
			utils::json_as_array(args["boundary_conditions"]["dirichlet_boundary"]),
			root_path, mesh.dimension());
	}

	io::OutputSpace NonlinearElasticVarForm::output_space() const
	{
		auto space = ElasticVarForm::output_space();
		space.collision_mesh = &collision_mesh;
		space.obstacle = &obstacle;
		return space;
	}

	VarFormDebugData NonlinearElasticVarForm::debug_data() const
	{
		return {
			mesh_.get(),
			assembler.get(),
			&bases,
			&geom_bases(),
			&total_local_boundary,
			n_bases,
			n_obstacle_vertices(),
			root_path};
	}

	std::vector<io::OutputField> NonlinearElasticVarForm::output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		std::vector<io::OutputField> fields;
		if (!mesh_ || !problem || solution.size() <= 0)
			return fields;

		const bool export_displacement = options.export_field("displacement");
		const bool export_solution = options.export_field("solution");
		const bool has_element_samples = sample.local_points.rows() > 0 && sample.local_points.rows() == sample.element_ids.size();
		const int output_rows = sample.points.rows() > 0 ? sample.points.rows() : std::max<int>(sample.local_points.rows(), sample.node_ids.size());

		const int actual_dim = problem->is_scalar() ? 1 : mesh_->dimension();
		const auto &paraview_options = args["output"]["paraview"]["options"];
		const bool material_params = paraview_options["material"];
		const bool body_ids = paraview_options["body_ids"];
		const bool velocity = paraview_options["velocity"];
		const bool acceleration = paraview_options["acceleration"];
		const bool forces = paraview_options["forces"] && !problem->is_scalar();
		const bool tensor_values = paraview_options["tensor_values"] && !problem->is_scalar();
		const bool scalar_values = paraview_options["scalar_values"];
		const bool use_spline = args["space"]["basis_type"] == "Spline";

		const auto append_obstacle_values = [&](Eigen::MatrixXd &sampled_values, const Eigen::MatrixXd &dof_values) -> bool {
			if (obstacle.n_vertices() <= 0)
				return sample.points.rows() == 0 || sample.points.rows() == sampled_values.rows();

			const bool has_obstacle_rows =
				sample.points.rows() == sampled_values.rows() + obstacle.n_vertices()
				&& sample.points.cols() == obstacle.v().cols()
				&& sample.points.bottomRows(obstacle.n_vertices()).isApprox(obstacle.v());

			if (!has_obstacle_rows)
				return sample.points.rows() == 0 || sample.points.rows() == sampled_values.rows();

			sampled_values.conservativeResize(sampled_values.rows() + obstacle.n_vertices(), sampled_values.cols());
			if (dof_values.rows() >= obstacle.ndof())
				sampled_values.bottomRows(obstacle.n_vertices()) =
					utils::unflatten(dof_values.bottomRows(obstacle.ndof()), sampled_values.cols());
			else
				sampled_values.bottomRows(obstacle.n_vertices()).setZero();
			return true;
		};

		const auto resize_to_output_rows = [&](Eigen::MatrixXd &values) {
			if (output_rows <= values.rows())
				return;

			const int previous_rows = values.rows();
			values.conservativeResize(output_rows, values.cols());
			values.bottomRows(output_rows - previous_rows).setZero();
		};

		const auto sample_dof_field = [&](const Eigen::MatrixXd &dof_values, const int field_dim, Eigen::MatrixXd &values) -> bool {
			if (dof_values.size() <= 0 || field_dim <= 0)
				return false;

			if (has_element_samples)
			{
				values.resize(sample.local_points.rows(), field_dim);
				for (int i = 0; i < sample.local_points.rows(); ++i)
				{
					const int element_id = sample.element_ids(i);
					if (element_id < 0)
					{
						values.row(i).setZero();
						continue;
					}

					Eigen::MatrixXd local_sol, local_grad;
					io::Evaluator::interpolate_at_local_vals(
						*mesh_, field_dim, bases, geom_bases(),
						element_id, sample.local_points.row(i), dof_values, local_sol, local_grad);

					for (int d = 0; d < field_dim; ++d)
						values(i, d) = local_sol(d);
				}

				return append_obstacle_values(values, dof_values);
			}

			if (sample.node_ids.size() > 0)
			{
				values.resize(sample.node_ids.size(), field_dim);
				for (int i = 0; i < sample.node_ids.size(); ++i)
				{
					const int node_id = sample.node_ids(i);
					for (int d = 0; d < field_dim; ++d)
					{
						const int dof = node_id * field_dim + d;
						if (dof < 0 || dof >= dof_values.rows())
							return false;
						values(i, d) = dof_values(dof);
					}
				}

				return sample.points.rows() == 0 || sample.points.rows() == values.rows();
			}

			return false;
		};

		const auto append_sampled_dof_field = [&](const std::string &name, const Eigen::MatrixXd &dof_values, const int field_dim) {
			Eigen::MatrixXd values;
			if (sample_dof_field(dof_values, field_dim, values))
				fields.push_back({name, values, io::OutputField::Association::Point});
		};

		const auto append_scalar_values = [&]() {
			if (!scalar_values || problem->is_scalar() || !has_element_samples)
				return;

			const bool wants_scalar = options.fields.empty()
									  || options.export_field("von_mises")
									  || options.export_field("von_mises_avg");
			if (!wants_scalar)
				return;

			std::vector<assembler::Assembler::NamedMatrix> point_values;
			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id < 0)
					continue;

				std::vector<assembler::Assembler::NamedMatrix> local_values;
				assembler->compute_scalar_value(
					assembler::OutputData(sample.time, element_id, bases[element_id], geom_bases()[element_id], sample.local_points.row(i), solution),
					local_values);

				if (point_values.empty())
				{
					point_values.resize(local_values.size());
					for (int k = 0; k < local_values.size(); ++k)
					{
						point_values[k].first = local_values[k].first;
						point_values[k].second.setZero(output_rows, local_values[k].second.cols());
					}
				}

				for (int k = 0; k < local_values.size(); ++k)
					point_values[k].second.row(i) = local_values[k].second;
			}

			for (const auto &[name, values] : point_values)
			{
				if (options.export_field(name))
					fields.push_back({name, values, io::OutputField::Association::Point});
			}
		};

		const auto append_tensor_values = [&]() {
			if (!tensor_values || problem->is_scalar() || !has_element_samples)
				return;

			const bool wants_tensor = options.fields.empty()
									  || options.export_field("cauchy_stess")
									  || options.export_field("pk1_stess")
									  || options.export_field("pk2_stess")
									  || options.export_field("F");
			if (!wants_tensor)
				return;

			std::vector<assembler::Assembler::NamedMatrix> point_values;
			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id < 0)
					continue;

				std::vector<assembler::Assembler::NamedMatrix> local_values;
				assembler->compute_tensor_value(
					assembler::OutputData(sample.time, element_id, bases[element_id], geom_bases()[element_id], sample.local_points.row(i), solution),
					local_values);

				if (point_values.empty())
				{
					point_values.resize(local_values.size());
					for (int k = 0; k < local_values.size(); ++k)
					{
						point_values[k].first = local_values[k].first;
						point_values[k].second.setZero(output_rows, local_values[k].second.cols());
					}
				}

				for (int k = 0; k < local_values.size(); ++k)
					point_values[k].second.row(i) = local_values[k].second;
			}

			for (const auto &[name, values] : point_values)
			{
				if (!options.export_field(name))
					continue;

				const int stride = mesh_->dimension();
				assert(values.cols() % stride == 0);
				for (int i = 0; i < values.cols(); i += stride)
				{
					const int ii = (i / stride) + 1;
					fields.push_back({fmt::format("{:s}_{:d}", name, ii), values.middleCols(i, stride), io::OutputField::Association::Point});
				}
			}
		};

		const auto append_averaged_values = [&]() {
			if (use_spline || problem->is_scalar() || !has_element_samples || (!scalar_values && !tensor_values))
				return;

			const bool wants_avg = options.fields.empty()
								   || options.export_field("von_mises_avg")
								   || options.export_field("cauchy_stess_avg")
								   || options.export_field("pk1_stess_avg")
								   || options.export_field("pk2_stess_avg")
								   || options.export_field("F_avg");
			if (!wants_avg)
				return;

			Eigen::MatrixXd areas(n_bases, 1);
			areas.setZero();
			std::vector<assembler::Assembler::NamedMatrix> tmp_s, tmp_t;
			std::vector<Eigen::MatrixXd> avg_scalar, avg_tensor;

			for (int e = 0; e < int(bases.size()); ++e)
			{
				Eigen::MatrixXd local_pts;
				if (mesh_->is_simplex(e))
				{
					if (mesh_->dimension() == 3)
						autogen::p_nodes_3d(disc_orders(e), local_pts);
					else
						autogen::p_nodes_2d(disc_orders(e), local_pts);
				}
				else if (mesh_->is_cube(e))
				{
					if (mesh_->dimension() == 3)
						autogen::q_nodes_3d(disc_orders(e), local_pts);
					else
						autogen::q_nodes_2d(disc_orders(e), local_pts);
				}
				else if (mesh_->is_prism(e))
				{
					autogen::prism_nodes_3d(disc_orders(e), disc_ordersq(e), local_pts);
				}
				else
				{
					continue;
				}

				const basis::ElementBases &bs = bases[e];
				const basis::ElementBases &gbs = geom_bases()[e];

				assembler::ElementAssemblyValues vals;
				vals.compute(e, mesh_->is_volume(), bs, gbs);
				const double area = (vals.det.array() * vals.quadrature.weights.array()).sum();

				if (scalar_values)
					assembler->compute_scalar_value(assembler::OutputData(sample.time, e, bs, gbs, local_pts, solution), tmp_s);
				if (tensor_values)
					assembler->compute_tensor_value(assembler::OutputData(sample.time, e, bs, gbs, local_pts, solution), tmp_t);

				if (avg_scalar.empty() && !tmp_s.empty())
				{
					avg_scalar.resize(tmp_s.size());
					for (auto &m : avg_scalar)
						m.setZero(n_bases, 1);
				}
				if (avg_tensor.empty() && !tmp_t.empty())
				{
					avg_tensor.resize(tmp_t.size());
					for (auto &m : avg_tensor)
						m.setZero(n_bases, actual_dim * actual_dim);
				}

				for (size_t j = 0; j < bs.bases.size(); ++j)
				{
					const basis::Basis &b = bs.bases[j];
					if (b.global().size() > 1)
						continue;

					const int index = b.global().front().index;
					areas(index) += area;
					for (int k = 0; k < tmp_s.size(); ++k)
						avg_scalar[k](index) += tmp_s[k].second(j) * area;
					for (int k = 0; k < tmp_t.size(); ++k)
						avg_tensor[k].row(index) += tmp_t[k].second.row(j) * area;
				}
			}

			for (auto &m : avg_scalar)
				for (int i = 0; i < m.rows(); ++i)
					if (areas(i) > 0)
						m(i) /= areas(i);
			for (auto &m : avg_tensor)
				for (int i = 0; i < m.rows(); ++i)
					if (areas(i) > 0)
						m.row(i) /= areas(i);

			for (int k = 0; k < tmp_s.size(); ++k)
			{
				const std::string name = fmt::format("{:s}_avg", tmp_s[k].first);
				if (!options.export_field(name))
					continue;

				Eigen::MatrixXd sampled;
				if (sample_dof_field(avg_scalar[k], 1, sampled))
					fields.push_back({name, sampled, io::OutputField::Association::Point});
			}

			for (int k = 0; k < tmp_t.size(); ++k)
			{
				const std::string base_name = fmt::format("{:s}_avg", tmp_t[k].first);
				if (!options.export_field(base_name))
					continue;

				Eigen::MatrixXd sampled;
				if (!sample_dof_field(utils::flatten(avg_tensor[k]), actual_dim * actual_dim, sampled))
					continue;

				const int stride = mesh_->dimension();
				for (int i = 0; i < sampled.cols(); i += stride)
				{
					const int ii = (i / stride) + 1;
					fields.push_back({fmt::format("{:s}_{:d}", base_name, ii), sampled.middleCols(i, stride), io::OutputField::Association::Point});
				}
			}
		};

		const auto append_material_fields = [&]() {
			if (!material_params || !has_element_samples)
				return;

			const auto &params = assembler->parameters();
			std::map<std::string, Eigen::MatrixXd> param_values;
			for (const auto &[p, _] : params)
				param_values[p].setZero(output_rows, 1);
			Eigen::MatrixXd rhos = Eigen::MatrixXd::Zero(output_rows, 1);

			const auto &density = mass_matrix_assembler->density();
			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id < 0)
					continue;

				for (const auto &[p, func] : params)
					param_values.at(p)(i) = func(sample.local_points.row(i), sample.points.row(i), sample.time, element_id);
				rhos(i) = density(sample.local_points.row(i), sample.points.row(i), sample.time, element_id);
			}

			for (const auto &[name, values] : param_values)
				if (options.export_field(name))
					fields.push_back({name, values, io::OutputField::Association::Point});
			if (options.export_field("rho"))
				fields.push_back({"rho", rhos, io::OutputField::Association::Point});
		};

		const auto append_body_ids = [&]() {
			if (!(body_ids || options.export_field("body_ids")) || !has_element_samples)
				return;

			Eigen::MatrixXd ids = Eigen::MatrixXd::Zero(output_rows, 1);
			for (int i = 0; i < sample.element_ids.size(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id >= 0)
					ids(i) = mesh_->get_body_id(element_id);
			}
			fields.push_back({"body_ids", ids, io::OutputField::Association::Point});
		};

		const auto compute_traction_forces = [&]() {
			Eigen::MatrixXd traction_forces;
			traction_forces.setZero(n_bases * actual_dim, 1);

			Eigen::MatrixXd uv, points, normals;
			Eigen::VectorXd weights;
			Eigen::VectorXi global_primitive_ids;
			assembler::ElementAssemblyValues vals;

			for (const auto &lb : total_local_boundary)
			{
				const int e = lb.element_id();
				const bool has_samples = utils::BoundarySampler::boundary_quadrature(
					lb, n_boundary_samples(), *mesh_, false, uv, points, normals, weights, global_primitive_ids);
				if (!has_samples)
					continue;

				const basis::ElementBases &bs = bases[e];
				const basis::ElementBases &gbs = geom_bases()[e];
				vals.compute(e, mesh_->is_volume(), points, bs, gbs);

				for (int n = 0; n < normals.rows(); ++n)
				{
					Eigen::MatrixXd deform_mat = Eigen::MatrixXd::Zero(actual_dim, actual_dim);
					for (const auto &b : vals.basis_values)
					{
						for (const auto &g : b.global)
						{
							for (int d = 0; d < actual_dim; ++d)
								deform_mat.row(d) += solution(g.index * actual_dim + d) * b.grad.row(n);
						}
					}

					Eigen::MatrixXd trafo = vals.jac_it[n].inverse() + deform_mat;
					normals.row(n) = normals.row(n) * trafo.inverse();
					normals.row(n).normalize();
				}

				std::vector<assembler::Assembler::NamedMatrix> tensor_flat;
				assembler->compute_tensor_value(assembler::OutputData(sample.time, e, bs, gbs, points, solution), tensor_flat);

				for (long n = 0; n < vals.basis_values.size(); ++n)
				{
					const assembler::AssemblyValues &v = vals.basis_values[n];
					const int g_index = v.global[0].index * actual_dim;
					for (int q = 0; q < points.rows(); ++q)
					{
						assert(tensor_flat[0].first == "cauchy_stess");
						Eigen::MatrixXd stress_tensor = utils::unflatten(tensor_flat[0].second.row(q), actual_dim);
						traction_forces.block(g_index, 0, actual_dim, 1) += stress_tensor * normals.row(q).transpose() * v.val(q) * weights(q);
					}
				}
			}

			return traction_forces;
		};

		const auto append_traction_force = [&]() {
			if (problem->is_scalar() || !options.export_field("traction_force"))
				return;

			if (has_element_samples && sample.normals.rows() == sample.local_points.rows() && sample.primitive_ids.size() == sample.local_points.rows())
			{
				Eigen::MatrixXd values = Eigen::MatrixXd::Zero(output_rows, actual_dim);
				for (int i = 0; i < sample.local_points.rows(); ++i)
				{
					const int element_id = sample.element_ids(i);
					if (element_id < 0)
						continue;

					std::vector<assembler::Assembler::NamedMatrix> tensor_flat;
					assembler->compute_tensor_value(
						assembler::OutputData(sample.time, element_id, bases[element_id], geom_bases()[element_id], sample.local_points.row(i), solution),
						tensor_flat);

					assert(tensor_flat[0].first == "cauchy_stess");
					Eigen::Map<Eigen::MatrixXd> tensor(tensor_flat[0].second.data(), actual_dim, actual_dim);
					values.row(i) = sample.normals.row(i) * tensor;

					double area = 0;
					const int primitive_id = sample.primitive_ids(i);
					if (mesh_->is_volume())
					{
						if (mesh_->is_simplex(element_id))
							area = mesh_->tri_area(primitive_id);
						else if (mesh_->is_cube(element_id))
							area = mesh_->quad_area(primitive_id);
						else if (mesh_->is_prism(element_id))
							area = mesh_->n_face_vertices(primitive_id) == 4 ? mesh_->quad_area(primitive_id) : mesh_->tri_area(primitive_id);
					}
					else
					{
						area = mesh_->edge_length(primitive_id);
					}
					values.row(i) *= area;
				}
				fields.push_back({"traction_force", values, io::OutputField::Association::Point});
				return;
			}

			append_sampled_dof_field("traction_force", compute_traction_forces(), actual_dim);
		};

		append_scalar_values();
		append_tensor_values();
		append_averaged_values();
		append_material_fields();
		append_body_ids();

		if (problem->is_time_dependent())
		{
			if (velocity || options.export_field("velocity"))
				append_sampled_dof_field(
					"velocity",
					solve_data.time_integrator ? solve_data.time_integrator->v_prev() : Eigen::VectorXd::Zero(solution.size()),
					actual_dim);
			if (acceleration || options.export_field("acceleration"))
				append_sampled_dof_field(
					"acceleration",
					solve_data.time_integrator ? solve_data.time_integrator->a_prev() : Eigen::VectorXd::Zero(solution.size()),
					actual_dim);
		}

		if (forces)
		{
			const double s = solve_data.time_integrator ? solve_data.time_integrator->acceleration_scaling() : 1;
			for (const auto &[name, form] : solve_data.named_forms())
			{
				const std::string field_name = name + "_forces";
				if (!options.export_field(field_name))
					continue;

				Eigen::VectorXd force;
				if (form && form->enabled())
				{
					form->first_derivative(solution, force);
					force *= -1.0 / s;
				}
				else
				{
					force.setZero(solution.size());
				}
				append_sampled_dof_field(field_name, force, actual_dim);
			}
		}

		append_traction_force();

		if (options.export_field("gradient_of_elastic_potential") && solve_data.elastic_form)
		{
			Eigen::VectorXd potential_grad;
			solve_data.elastic_form->first_derivative(solution, potential_grad);
			append_sampled_dof_field("gradient_of_elastic_potential", potential_grad, actual_dim);
		}

		if (options.export_field("gradient_of_contact_potential") && solve_data.contact_form && solve_data.contact_form->weight() > 0)
		{
			Eigen::VectorXd potential_grad;
			solve_data.contact_form->first_derivative(solution, potential_grad);
			potential_grad *= -solve_data.contact_form->barrier_stiffness() / solve_data.contact_form->weight();
			append_sampled_dof_field("gradient_of_contact_potential", potential_grad, actual_dim);
		}

		if (export_displacement)
			append_sampled_dof_field("displacement", solution, actual_dim);
		if (export_solution)
			append_sampled_dof_field("solution", solution, actual_dim);

		const auto has_field = [&](const std::string &name) {
			return std::any_of(fields.begin(), fields.end(), [&](const io::OutputField &field) {
				return field.association == io::OutputField::Association::Point && field.name == name;
			});
		};

		const auto append_collision_dof_field = [&](const std::string &name, const Eigen::MatrixXd &dof_values) {
			if (has_field(name) || dof_values.size() <= 0)
				return;

			Eigen::MatrixXd values = collision_mesh.map_displacements(utils::unflatten(dof_values, actual_dim));
			if (values.rows() == sample.points.rows())
				fields.push_back({name, values, io::OutputField::Association::Point});
		};

		if (paraview_options["forces"] && !problem->is_scalar())
		{
			const double s = solve_data.time_integrator ? solve_data.time_integrator->acceleration_scaling() : 1;
			for (const auto &[name, form] : solve_data.named_forms())
			{
				const std::string field_name = name + "_forces";
				if (!options.export_field(field_name))
					continue;

				Eigen::VectorXd force;
				if (form && form->enabled())
				{
					form->first_derivative(solution, force);
					force *= -1.0 / s;
				}
				else
				{
					force.setZero(solution.size());
				}
				append_collision_dof_field(field_name, force);
			}
		}

		if (options.export_field("gradient_of_elastic_potential") && solve_data.elastic_form)
		{
			Eigen::VectorXd potential_grad;
			solve_data.elastic_form->first_derivative(solution, potential_grad);
			append_collision_dof_field("gradient_of_elastic_potential", potential_grad);
		}

		if (options.export_field("gradient_of_contact_potential") && solve_data.contact_form && solve_data.contact_form->weight() > 0)
		{
			Eigen::VectorXd potential_grad;
			solve_data.contact_form->first_derivative(solution, potential_grad);
			potential_grad *= -solve_data.contact_form->barrier_stiffness() / solve_data.contact_form->weight();
			append_collision_dof_field("gradient_of_contact_potential", potential_grad);
		}

		return fields;
	}

	void NonlinearElasticVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		ElasticVarForm::build_basis(mesh, iso_parametric, args);

		logger().info("Building collision mesh...");
		build_collision_mesh(mesh, args);
		// FIXME!! handle periodic collision mesh
		//  if (periodic_bc && args["contact"]["periodic"])
		//  	build_periodic_collision_mesh();
		logger().info("Done!");
	}

	std::shared_ptr<assembler::RhsAssembler> NonlinearElasticVarForm::build_rhs_assembler(
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const assembler::AssemblyValsCache &ass_vals_cache) const
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		const int size = problem->is_scalar() ? 1 : mesh_->dimension();

		return std::make_shared<assembler::RhsAssembler>(
			*assembler, *mesh_, obstacle,
			dirichlet_nodes, neumann_nodes,
			dirichlet_nodes_position, neumann_nodes_position,
			n_bases, size, bases, geom_bases(), ass_vals_cache, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params);
	}

	void NonlinearElasticVarForm::build_collision_mesh(
		const mesh::Mesh &mesh,
		const json &args)
	{
		build_collision_mesh(
			mesh, n_bases, bases, geom_bases(), total_local_boundary, obstacle,
			args, [this](const std::string &p) { return utils::resolve_path(p, root_path, false); },
			in_node_to_node, collision_mesh);
	}

	void NonlinearElasticVarForm::build_collision_mesh(
		const mesh::Mesh &mesh,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		const mesh::Obstacle &obstacle,
		const json &args,
		const std::function<std::string(const std::string &)> &resolve_input_path,
		const Eigen::VectorXi &in_node_to_node,
		ipc::CollisionMesh &collision_mesh)
	{
		Eigen::MatrixXd collision_vertices;
		Eigen::VectorXi collision_codim_vids;
		Eigen::MatrixXi collision_edges, collision_triangles;
		std::vector<Eigen::Triplet<double>> displacement_map_entries;

		if (args.contains("/contact/collision_mesh"_json_pointer)
			&& args.at("/contact/collision_mesh/enabled"_json_pointer).get<bool>())
		{
			const json collision_mesh_args = args.at("/contact/collision_mesh"_json_pointer);
			if (collision_mesh_args.contains("linear_map"))
			{
				assert(displacement_map_entries.empty());
				assert(collision_mesh_args.contains("mesh"));
				const std::string root_path = utils::json_value<std::string>(args, "root_path", "");
				// TODO: handle transformation per geometry
				const json transformation = utils::json_as_array(args["geometry"])[0]["transformation"];
				mesh::load_collision_proxy(
					utils::resolve_path(collision_mesh_args["mesh"], root_path),
					utils::resolve_path(collision_mesh_args["linear_map"], root_path),
					in_node_to_node, transformation, collision_vertices, collision_codim_vids,
					collision_edges, collision_triangles, displacement_map_entries);
			}
			else if (collision_mesh_args.contains("max_edge_length"))
			{
				logger().debug(
					"Building collision proxy with max edge length={} ...",
					collision_mesh_args["max_edge_length"].get<double>());
				igl::Timer timer;
				timer.start();
				build_collision_proxy(
					bases, geom_bases, total_local_boundary, n_bases, mesh.dimension(),
					collision_mesh_args["max_edge_length"], collision_vertices,
					collision_triangles, displacement_map_entries,
					collision_mesh_args["tessellation_type"]);
				if (collision_triangles.size())
					igl::edges(collision_triangles, collision_edges);
				timer.stop();
				logger().debug(fmt::format(
					std::locale("en_US.UTF-8"),
					"Done (took {:g}s, {:L} vertices, {:L} triangles)",
					timer.getElapsedTime(),
					collision_vertices.rows(), collision_triangles.rows()));
			}
			else
			{
				io::OutGeometryData::extract_boundary_mesh(
					mesh, n_bases - obstacle.n_vertices(), bases, total_local_boundary,
					collision_vertices, collision_edges, collision_triangles, displacement_map_entries);
			}
		}
		else
		{
			io::OutGeometryData::extract_boundary_mesh(
				mesh, n_bases - obstacle.n_vertices(), bases, total_local_boundary,
				collision_vertices, collision_edges, collision_triangles, displacement_map_entries);
		}

		std::vector<bool> is_orientable_vertex(collision_vertices.rows(), true);

		// n_bases already contains the obstacle vertices
		const int num_fe_nodes = n_bases - obstacle.n_vertices();
		const int num_fe_collision_vertices = collision_vertices.rows();
		assert(collision_edges.size() == 0 || collision_edges.maxCoeff() < num_fe_collision_vertices);
		assert(collision_triangles.size() == 0 || collision_triangles.maxCoeff() < num_fe_collision_vertices);

		// Append the obstacles to the collision mesh
		if (obstacle.n_vertices() > 0)
		{
			utils::append_rows(collision_vertices, obstacle.v());
			utils::append_rows(collision_codim_vids, obstacle.codim_v().array() + num_fe_collision_vertices);
			utils::append_rows(collision_edges, obstacle.e().array() + num_fe_collision_vertices);
			utils::append_rows(collision_triangles, obstacle.f().array() + num_fe_collision_vertices);

			for (int i = 0; i < obstacle.n_vertices(); i++)
			{
				is_orientable_vertex.push_back(false);
			}

			if (!displacement_map_entries.empty())
			{
				displacement_map_entries.reserve(displacement_map_entries.size() + obstacle.n_vertices());
				for (int i = 0; i < obstacle.n_vertices(); i++)
				{
					displacement_map_entries.emplace_back(num_fe_collision_vertices + i, num_fe_nodes + i, 1.0);
				}
			}
		}

		std::vector<bool> is_on_surface = ipc::CollisionMesh::construct_is_on_surface(
			collision_vertices.rows(), collision_edges);
		for (const int vid : collision_codim_vids)
		{
			is_on_surface[vid] = true;
		}

		Eigen::SparseMatrix<double> displacement_map;
		if (!displacement_map_entries.empty())
		{
			displacement_map.resize(collision_vertices.rows(), n_bases);
			displacement_map.setFromTriplets(displacement_map_entries.begin(), displacement_map_entries.end());
		}

		collision_mesh = ipc::CollisionMesh(
			is_on_surface, is_orientable_vertex, collision_vertices, collision_edges, collision_triangles,
			displacement_map);

		collision_mesh.can_collide = [&collision_mesh, num_fe_collision_vertices](size_t vi, size_t vj) {
			// obstacles do not collide with other obstacles
			return collision_mesh.to_full_vertex_id(vi) < num_fe_collision_vertices
				   || collision_mesh.to_full_vertex_id(vj) < num_fe_collision_vertices;
		};

		collision_mesh.init_area_jacobians();
	}

	std::shared_ptr<assembler::PressureAssembler> NonlinearElasticVarForm::build_pressure_assembler() const
	{
		const int size = problem->is_scalar() ? 1 : mesh_->dimension();

		return std::make_shared<assembler::PressureAssembler>(
			*assembler, *mesh_, obstacle,
			local_pressure_boundary,
			local_pressure_cavity,
			boundary_nodes,
			primitive_to_node(), node_to_primitive(),
			n_bases, size, bases, geom_bases(), *problem);
	}

	void NonlinearElasticStaticVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", assembler->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			// FIXME
			//  read_initial_x_from_file(
			//  resolve_input_path(args["input"]["data"]["state"]), "u",
			//  args["input"]["data"]["reorder"], in_node_to_node,
			//  mesh->dimension(), solution);

			if (sol.size() <= 0)
				initial_solution(sol);

			if (sol.cols() > 1) // ignore previous solutions
				sol.conservativeResize(Eigen::NoChange, 1);
		}
		init_solve(sol, 1.0);

		solve_tensor_nonlinear(0, sol, true);

		const std::string state_path = resolve_output_path(args["output"]["data"]["state"]);
		if (!state_path.empty())
			io::write_matrix(state_path, "u", sol);

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

	void NonlinearElasticTransientVarForm::solve_problem(Eigen::MatrixXd &sol)
	{
		const bool save_stats = args["output"]["stats"];
		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", assembler->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			// FIXME
			//  read_initial_x_from_file(
			//  resolve_input_path(args["input"]["data"]["state"]), "u",
			//  args["input"]["data"]["reorder"], in_node_to_node,
			//  mesh->dimension(), solution);

			if (sol.size() <= 0)
				initial_solution(sol);

			if (sol.cols() > 1) // ignore previous solutions
				sol.conservativeResize(Eigen::NoChange, 1);
		}
		init_solve(sol, t0 + dt);

		const int t_offset = args["output"]["data"]["file_index_offset"].get<int>();

		// Write the total energy to a CSV file
		int save_i = 0;

		std::unique_ptr<io::EnergyCSVWriter> energy_csv = nullptr;
		std::unique_ptr<io::RuntimeStatsCSVWriter> stats_csv = nullptr;

		if (save_stats)
		{
			logger().debug("Saving nl stats to {} and {}", resolve_output_path("energy.csv"), resolve_output_path("stats.csv"));
			energy_csv = std::make_unique<io::EnergyCSVWriter>(resolve_output_path("energy.csv"), solve_data);
			const io::OutputSpace space = output_space();
			stats_csv = std::make_unique<io::RuntimeStatsCSVWriter>(
				resolve_output_path("stats.csv"),
				n_bases,
				space.mesh ? space.mesh->n_elements() : 0,
				t0, dt);
		}

		// Save the initial solution
		if (energy_csv)
			energy_csv->write(save_i, sol);
		save_timestep(t0, t_offset, t0, dt, sol);

		save_i++;

		for (int t = 1; t <= time_steps; ++t)
		{
			double forward_solve_time = 0, remeshing_time = 0, global_relaxation_time = 0;

			{
				POLYFEM_SCOPED_TIMER(forward_solve_time);
				solve_tensor_nonlinear(t, sol, true);
			}

			// Always save the solution for consistency
			if (energy_csv)
				energy_csv->write(save_i, sol);
			save_timestep(t0 + dt * t, t + t_offset, t0, dt, sol);
			save_i++;

			{
				POLYFEM_SCOPED_TIMER("Update quantities");

				solve_data.time_integrator->update_quantities(sol);

				solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);

				solve_data.update_dt();
				solve_data.update_barrier_stiffness(sol);
			}

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);

			save_step_state(t0, dt, t + t_offset, sol);
			if (stats_csv)
				stats_csv->write(t, forward_solve_time, remeshing_time, global_relaxation_time, sol);
		}

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

	void NonlinearElasticVarForm::init_forms(const json &args, const int dim, Eigen::MatrixXd &sol, const double t)
	{
		damping_assembler = std::make_shared<assembler::ViscousDamping>();
		set_materials(*damping_assembler);

		elasticity_pressure_assembler = build_pressure_assembler();

		// for backward solve
		damping_prev_assembler = std::make_shared<assembler::ViscousDampingPrev>();
		set_materials(*damping_prev_assembler);

		const ElementInversionCheck check_inversion = args["solver"]["advanced"]["check_inversion"];

		forms = solve_data.init_forms(
			// General
			units,
			dim, t, in_node_to_node,
			// Elastic form
			n_bases, bases, geom_bases(), *assembler, ass_vals_cache, mass_ass_vals_cache, args["solver"]["advanced"]["jacobian_threshold"], check_inversion,
			// Body form
			0, boundary_nodes, local_boundary,
			local_neumann_boundary,
			n_boundary_samples(), rhs, sol, mass_matrix_assembler->density(),
			// Pressure form
			local_pressure_boundary, local_pressure_cavity, elasticity_pressure_assembler,
			// Inertia form
			args.value("/time/quasistatic"_json_pointer, true), mass,
			damping_assembler->is_valid() ? damping_assembler : nullptr,
			// Lagged regularization form
			args["solver"]["advanced"]["lagged_regularization_weight"],
			args["solver"]["advanced"]["lagged_regularization_iterations"],
			// Augmented lagrangian form
			obstacle.ndof(), args["constraints"]["hard"], args["constraints"]["soft"],
			// Contact form
			args["contact"]["enabled"], collision_mesh, args["contact"]["dhat"],
			avg_mass, args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_area_weighting"]) : false,
			args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_improved_max_operator"]) : false,
			args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_physical_barrier"]) : false,
			args["solver"]["contact"]["barrier_stiffness"],
			args["solver"]["contact"]["initial_barrier_stiffness"],
			args["solver"]["contact"]["CCD"]["broad_phase"],
			args["solver"]["contact"]["CCD"]["tolerance"],
			args["solver"]["contact"]["CCD"]["max_iterations"],
			false,
			// Smooth Contact Form
			args["contact"]["use_gcp_formulation"],
			args["contact"]["alpha_t"],
			args["contact"]["alpha_n"],
			args["contact"]["use_adaptive_dhat"],
			args["contact"]["min_distance_ratio"],
			// Normal Adhesion Form
			args["contact"]["adhesion"]["adhesion_enabled"],
			args["contact"]["adhesion"]["dhat_p"],
			args["contact"]["adhesion"]["dhat_a"],
			args["contact"]["adhesion"]["adhesion_strength"],
			// Tangential Adhesion Form
			args["contact"]["adhesion"]["tangential_adhesion_coefficient"],
			args["contact"]["adhesion"]["epsa"],
			args["solver"]["contact"]["tangential_adhesion_iterations"],
			// Homogenization
			assembler::MacroStrainValue(),
			// Periodic contact
			false, Eigen::VectorXi(), nullptr,
			// Friction form
			args["contact"]["friction_coefficient"],
			args["contact"]["epsv"],
			args["solver"]["contact"]["friction_iterations"],
			// Rayleigh damping form
			args["solver"]["rayleigh_damping"]);

		for (const auto &form : forms)
			form->set_output_dir(output_path);

		if (solve_data.contact_form != nullptr)
			solve_data.contact_form->save_ccd_debug_meshes = args["output"]["advanced"]["save_ccd_debug_meshes"];
	}

	void NonlinearElasticVarForm::init_solve(Eigen::MatrixXd &sol, const double t)
	{
		assert(sol.cols() == 1);
		assert(!problem->is_scalar()); // tensor

		// FIXME
		//  if (optimization_enabled != solver::CacheLevel::None)
		//  {
		//  	if (initial_sol_update.size() == ndof())
		//  		sol = initial_sol_update;
		//  	else
		//  		initial_sol_update = sol;
		//  }

		// --------------------------------------------------------------------
		// Check for initial intersections
		if (args["contact"]["enabled"])
		{
			POLYFEM_SCOPED_TIMER("Check for initial intersections");

			const Eigen::MatrixXd displaced = collision_mesh.displace_vertices(
				utils::unflatten(sol, mesh_->dimension()));

			if (ipc::has_intersections(collision_mesh, displaced, ipc::create_broad_phase(args["solver"]["contact"]["CCD"]["broad_phase"]).get()))
			{
				io::OBJWriter::write(
					resolve_output_path("intersection.obj"), displaced,
					collision_mesh.edges(), collision_mesh.faces());
				log_and_throw_error("Unable to solve, initial solution has intersections!");
			}
		}

		// --------------------------------------------------------------------

		if (problem->is_time_dependent())
		{
			POLYFEM_SCOPED_TIMER("Initialize time integrator");
			solve_data.time_integrator = ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);

			Eigen::MatrixXd solution, velocity, acceleration;
			initial_solution(solution); // Reload this because we need all previous solutions
			solution.col(0) = sol;      // Make sure the current solution is the same as `sol`
			assert(solution.rows() == sol.size());
			initial_velocity(velocity);
			assert(velocity.rows() == sol.size());
			initial_acceleration(acceleration);
			assert(acceleration.rows() == sol.size());

			solve_data.time_integrator->init(solution, velocity, acceleration, dt);
			assert(solve_data.time_integrator != nullptr);
		}
		else
		{
			solve_data.time_integrator = nullptr;
		}

		// --------------------------------------------------------------------
		// Initialize forms

		// --------------------------------------------------------------------
		// Initialize nonlinear problems

		init_forms(args, mesh_->dimension(), sol, t);

		double characteristic_length = 0;
		if (args["solver"]["advanced"]["characteristic_length"] > 0)
		{
			characteristic_length = args["solver"]["advanced"]["characteristic_length"];
		}
		else
		{
			RowVectorNd min, max;
			mesh_->bounding_box(min, max);
			characteristic_length = (max - min).norm();
		}

		double characteristic_force_density = 0;
		if (args["solver"]["advanced"]["characteristic_force_density"] <= 0)
		{
			logger().warn("No user-specified force density was provided, defaulting to 10000.");
			characteristic_force_density = 10000;
		}
		else
		{
			characteristic_force_density = args["solver"]["advanced"]["characteristic_force_density"];
		}

		if (pure_mass.size() == 0)
			pure_mass_matrix_assembler->assemble(mesh_->is_volume(), n_bases, bases, geom_bases(), pure_mass_ass_vals_cache, 0, pure_mass, true);

		const int ndof = n_bases * mesh_->dimension();
		solve_data.nl_problem = std::make_shared<solver::NLProblem>(
			ndof, nullptr, t, forms, solve_data.al_form,
			polysolve::linear::Solver::create(args["solver"]["linear"], logger()),
			characteristic_length, characteristic_force_density, pure_mass, mesh_->dimension());
		solve_data.nl_problem->init(sol);
		solve_data.nl_problem->update_quantities(t, sol);
		// --------------------------------------------------------------------

		stats.solver_info = json::array();
	}

	void NonlinearElasticVarForm::solve_tensor_nonlinear(int step, Eigen::MatrixXd &sol, const bool init_lagging)
	{
		assert(solve_data.nl_problem != nullptr);
		solver::NLProblem &nl_problem = *(solve_data.nl_problem);

		assert(sol.size() == rhs.size());

		if (nl_problem.uses_lagging())
		{
			if (init_lagging)
			{
				POLYFEM_SCOPED_TIMER("Initializing lagging");
				nl_problem.init_lagging(sol);
			}
			logger().info("Lagging iteration 1:");
		}

		save_subsolve(0, step, sol);

		std::shared_ptr<polysolve::nonlinear::Solver> nl_solver =
			polysolve::nonlinear::Solver::create(args["solver"]["augmented_lagrangian"]["nonlinear"], args["solver"]["linear"], units.characteristic_length(), logger());

		ALSolver al_solver(
			solve_data.al_form,
			args["solver"]["augmented_lagrangian"]["initial_weight"],
			args["solver"]["augmented_lagrangian"]["scaling"],
			args["solver"]["augmented_lagrangian"]["max_weight"],
			args["solver"]["augmented_lagrangian"]["eta"],
			[&](const Eigen::VectorXd &x) {
				this->solve_data.update_barrier_stiffness(sol);
			});

		al_solver.post_subsolve = [&](const double al_weight) {
			stats.solver_info.push_back(
				{{"type", al_weight > 0 ? "al" : "rc"},
				 {"t", step},
				 {"info", nl_solver->info()}});
			if (al_weight > 0)
				stats.solver_info.back()["weight"] = al_weight;
			save_subsolve(stats.solver_info.size(), step, sol);
		};

		Eigen::MatrixXd prev_sol = sol;
		al_solver.solve_al(nl_problem, sol,
						   args["solver"]["augmented_lagrangian"]["nonlinear"], args["solver"]["linear"], units.characteristic_length());

		al_solver.solve_reduced(nl_problem, sol,
								args["solver"]["nonlinear"], args["solver"]["linear"], units.characteristic_length());

		if (args["space"]["advanced"]["count_flipped_els_continuous"])
		{
			const auto invalidList = utils::count_invalid(mesh_->dimension(), bases, geom_bases(), sol);
			logger().debug("Flipped elements (cnt {}) : {}", invalidList.size(), invalidList);
		}

		const double lagging_tol = args["solver"]["contact"].value("friction_convergence_tol", 1e-2) * units.characteristic_length();

		bool lagging_converged = !nl_problem.uses_lagging();
		for (int lag_i = 1; !lagging_converged; lag_i++)
		{
			Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);

			nl_problem.update_lagging(tmp_sol, lag_i);

			Eigen::VectorXd grad;
			nl_problem.gradient(tmp_sol, grad);
			const double delta_x_norm = (prev_sol - sol).lpNorm<Eigen::Infinity>();
			logger().debug("Lagging convergence grad_norm={:g} tol={:g} (||Δx||={:g})", grad.norm(), lagging_tol, delta_x_norm);
			if (grad.norm() <= lagging_tol)
			{
				logger().info(
					"Lagging converged in {:d} iteration(s) (grad_norm={:g} tol={:g})",
					lag_i, grad.norm(), lagging_tol);
				lagging_converged = true;
				break;
			}

			if (delta_x_norm <= 1e-12)
			{
				logger().warn(
					"Lagging produced tiny update between iterations {:d} and {:d} (grad_norm={:g} grad_tol={:g} ||Δx||={:g} Δx_tol={:g}); stopping early",
					lag_i - 1, lag_i, grad.norm(), lagging_tol, delta_x_norm, 1e-6);
				lagging_converged = false;
				break;
			}

			if (lag_i >= nl_problem.max_lagging_iterations())
			{
				logger().warn(
					"Lagging failed to converge with {:d} iteration(s) (grad_norm={:g} tol={:g})",
					lag_i, grad.norm(), lagging_tol);
				lagging_converged = false;
				break;
			}

			logger().info("Lagging iteration {:d}:", lag_i + 1);
			nl_problem.init(sol);
			solve_data.update_barrier_stiffness(sol);
			nl_problem.normalize_forms();
			nl_solver->minimize(nl_problem, tmp_sol);
			nl_problem.finish();
			prev_sol = sol;
			sol = nl_problem.reduced_to_full(tmp_sol);

			stats.solver_info.push_back(
				{{"type", "rc"},
				 {"t", step},
				 {"lag_i", lag_i},
				 {"info", nl_solver->info()}});
			save_subsolve(stats.solver_info.size(), step, sol);
		}
	}
} // namespace polyfem::varform
