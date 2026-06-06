#include "ElasticVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>

#include <polyfem/basis/Basis.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>
#include <polyfem/autogen/prism_bases.hpp>

#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/MshWriter.hpp>

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/utils/MatrixUtils.hpp>


#include <ostream>

#include <spdlog/fmt/fmt.h>

namespace polyfem::varform
{
	QuadratureOrders ElasticVarForm::n_boundary_samples() const
	{
		using assembler::AssemblerUtils;
		const int n_b_samples_j = args["space"]["advanced"]["n_boundary_samples"];
		const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
		const int discr_order = std::max(disc_orders.maxCoeff(), gdiscr_order);

		const int n_b_samples = std::max(n_b_samples_j, AssemblerUtils::quadrature_order("Mass", discr_order, AssemblerUtils::BasisType::POLY, mesh_->dimension()));
		return {{n_b_samples, n_b_samples}};
	}

	std::vector<io::OutputField> ElasticVarForm::output_fields(
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

		const auto sample_dof_field = [&](const Eigen::MatrixXd &dof_values, const int field_dim, const bool use_obstacle_tail, Eigen::MatrixXd &values) -> bool {
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

				if (use_obstacle_tail)
					return append_obstacle_values(values, dof_values);

				resize_to_output_rows(values);
				return true;
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

		const auto append_sampled_dof_field = [&](const std::string &name, const Eigen::MatrixXd &dof_values, const int field_dim, const bool use_obstacle_tail) {
			Eigen::MatrixXd values;
			if (sample_dof_field(dof_values, field_dim, use_obstacle_tail, values))
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
				if (sample_dof_field(avg_scalar[k], 1, false, sampled))
					fields.push_back({name, sampled, io::OutputField::Association::Point});
			}

			for (int k = 0; k < tmp_t.size(); ++k)
			{
				const std::string base_name = fmt::format("{:s}_avg", tmp_t[k].first);
				if (!options.export_field(base_name))
					continue;

				Eigen::MatrixXd sampled;
				if (!sample_dof_field(utils::flatten(avg_tensor[k]), actual_dim * actual_dim, false, sampled))
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

			append_sampled_dof_field("traction_force", compute_traction_forces(), actual_dim, false);
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
					actual_dim, true);
			if (acceleration || options.export_field("acceleration"))
				append_sampled_dof_field(
					"acceleration",
					solve_data.time_integrator ? solve_data.time_integrator->a_prev() : Eigen::VectorXd::Zero(solution.size()),
					actual_dim, true);
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
				append_sampled_dof_field(field_name, force, actual_dim, true);
			}
		}

		append_traction_force();

		if (options.export_field("gradient_of_elastic_potential") && solve_data.elastic_form)
		{
			Eigen::VectorXd potential_grad;
			solve_data.elastic_form->first_derivative(solution, potential_grad);
			append_sampled_dof_field("gradient_of_elastic_potential", potential_grad, actual_dim, true);
		}

		if (options.export_field("gradient_of_contact_potential") && solve_data.contact_form && solve_data.contact_form->weight() > 0)
		{
			Eigen::VectorXd potential_grad;
			solve_data.contact_form->first_derivative(solution, potential_grad);
			potential_grad *= -solve_data.contact_form->barrier_stiffness() / solve_data.contact_form->weight();
			append_sampled_dof_field("gradient_of_contact_potential", potential_grad, actual_dim, true);
		}

		if (export_displacement)
			append_sampled_dof_field("displacement", solution, actual_dim, true);
		if (export_solution)
			append_sampled_dof_field("solution", solution, actual_dim, true);
		return fields;
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

	VarFormDebugData ElasticVarForm::debug_data() const
	{
		return {
			mesh_.get(),
			assembler.get(),
			&bases,
			&geom_bases(),
			&total_local_boundary,
			n_bases,
			obstacle.n_vertices(),
			root_path};
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
		const size_t n_vertices = n_bases - obstacle.n_vertices();
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

	io::OutputSpace ElasticVarForm::output_space() const
	{
		Eigen::VectorXi output_orders = disc_orders;
		if (mesh_ && disc_ordersq.size() == disc_orders.size())
		{
			for (int e = 0; e < output_orders.size(); ++e)
			{
				if (mesh_->is_prism(e))
					output_orders(e) = std::max(disc_orders(e), disc_ordersq(e));
			}
		}

		return {
			mesh_.get(),
			&geom_bases(),
			output_orders,
			&polys,
			&polys_3d,
			&total_local_boundary,
			&obstacle,
			nullptr,
			&dirichlet_nodes,
			&dirichlet_nodes_position};
	}
} // namespace polyfem::varform
