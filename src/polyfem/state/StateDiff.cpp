#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polysolve/FEMSolver.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>

#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/utils/local_to_global.hpp>

#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <deque>
#include <map>
#include <algorithm>

#include <fstream>

using namespace polyfem::basis;

namespace polyfem
{
	namespace
	{
		typedef std::vector<std::function<void(int, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::VectorXd &)>> VolumeIntegrandTerms;
		typedef std::vector<std::function<void(int, const Eigen::VectorXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::VectorXd &)>> SurfaceIntegrandTerms;

		class LocalThreadVecStorage
		{
		public:
			Eigen::MatrixXd vec;
			assembler::ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadVecStorage(const int size)
			{
				vec.resize(size, 1);
				vec.setZero();
			}
		};

		class LocalThreadScalarStorage
		{
		public:
			double val;
			assembler::ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadScalarStorage()
			{
				val = 0;
			}
		};

		double dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }

		void vector2matrix(const Eigen::VectorXd &vec, Eigen::MatrixXd &mat)
		{
			int size = sqrt(vec.size());
			assert(size * size == vec.size());

			mat.resize(size, size);
			for (int i = 0; i < size; i++)
				for (int j = 0; j < size; j++)
					mat(i, j) = vec(i * size + j);
		}

		void solve_zero_dirichlet(const json &args, StiffnessMatrix &A, Eigen::VectorXd &b, const std::vector<int> &indices, Eigen::MatrixXd &adjoint_solution)
		{
			auto solver = polysolve::LinearSolver::create(args["solver"], args["precond"]);
			solver->setParameters(args);
			const int precond_num = A.rows();

			for (int i : indices)
				b(i) = 0;

			Eigen::Vector4d adjoint_spectrum;
			Eigen::VectorXd x;
			adjoint_spectrum = dirichlet_solve(*solver, A, b, indices, x, precond_num, "", "", false, false);
			adjoint_solution = x;
		}

		void volume_integral(
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const VolumeIntegrandTerms &integrand_functions,
			const mesh::Mesh &mesh,
			double &integral)
		{
			integral = 0;

			const int n_elements = int(bases.size());

			auto storage = utils::create_thread_storage(LocalThreadScalarStorage());

			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					vals.compute(e, mesh.is_volume(), bases[e], gbases[e]);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					std::vector<Eigen::VectorXd> integrands;
					for (const auto &integrand_function : integrand_functions)
					{
						Eigen::VectorXd vec_term;
						integrand_function(e, quadrature.points, vals.val, vec_term);
						integrands.push_back(vec_term);
					}

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						double integrand_product = 1;
						for (const auto &integrand : integrands)
						{
							integrand_product *= integrand(q);
						}
						local_storage.val += integrand_product * local_storage.da(q);
					}
				}
			});

			for (const LocalThreadScalarStorage &local_storage : storage)
				integral += local_storage.val;
		}

		void surface_integral(
			const std::vector<mesh::LocalBoundary> &local_boundary,
			const int resolution,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const SurfaceIntegrandTerms &integrand_functions,
			const mesh::Mesh &mesh,
			double &integral)
		{
			integral = 0;

			auto storage = utils::create_thread_storage(LocalThreadScalarStorage());

			utils::maybe_parallel_for(local_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::MatrixXd uv, samples, gtmp;
				Eigen::VectorXi global_primitive_ids;
				Eigen::MatrixXd points, normals;
				Eigen::VectorXd weights;

				for (int lb_id = start; lb_id < end; ++lb_id)
				{
					const auto &lb = local_boundary[lb_id];
					const int e = lb.element_id();

					assembler::ElementAssemblyValues vals;
					bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, resolution, mesh, false, uv, points, normals, weights, global_primitive_ids);

					if (!has_samples)
						continue;

					const ElementBases &gbs = gbases[e];
					const ElementBases &bs = bases[e];

					vals.compute(e, mesh.is_volume(), points, bs, gbs);

					const Eigen::VectorXd da = weights.array();
					// const Eigen::VectorXd da = vals.det.array() * weights.array();

					for (int n = 0; n < vals.jac_it.size(); ++n)
					{
						normals.row(n) = normals.row(n) * vals.jac_it[n];
						normals.row(n).normalize();
					}

					std::vector<Eigen::VectorXd> integrands;
					for (const auto &integrand_function : integrand_functions)
					{
						Eigen::VectorXd vec_term;
						integrand_function(e, global_primitive_ids, points, vals.val, normals, vec_term);
						integrands.push_back(vec_term);
					}

					for (int q = 0; q < da.size(); ++q)
					{
						double integrand_product = 1;
						for (const auto &integrand : integrands)
						{
							integrand_product *= integrand(q);
						}
						local_storage.val += integrand_product * da(q);
					}
				}
			});

			for (const LocalThreadScalarStorage &local_storage : storage)
				integral += local_storage.val;
		}

		void replace_rows_by_identity(StiffnessMatrix &reduced_mat, const StiffnessMatrix &mat, const std::vector<int> &rows)
		{
			reduced_mat.resize(mat.rows(), mat.cols());

			std::vector<bool> mask(mat.rows(), false);
			for (int i : rows)
				mask[i] = true;

			std::vector<Eigen::Triplet<double>> coeffs;
			for (int k = 0; k < mat.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(mat, k); it; ++it)
				{
					if (mask[it.row()])
					{
						if (it.row() == it.col())
							coeffs.emplace_back(it.row(), it.col(), 1.0);
					}
					else
						coeffs.emplace_back(it.row(), it.col(), it.value());
				}
			}
			reduced_mat.setFromTriplets(coeffs.begin(), coeffs.end());
		}

		void replace_rows_by_zero(StiffnessMatrix &reduced_mat, const StiffnessMatrix &mat, const std::vector<int> &rows)
		{
			reduced_mat.resize(mat.rows(), mat.cols());

			std::vector<bool> mask(mat.rows(), false);
			for (int i : rows)
				mask[i] = true;

			std::vector<Eigen::Triplet<double>> coeffs;
			for (int k = 0; k < mat.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(mat, k); it; ++it)
				{
					if (mask[it.row()])
					{
						continue;
					}
					else
						coeffs.emplace_back(it.row(), it.col(), it.value());
				}
			}
			reduced_mat.setFromTriplets(coeffs.begin(), coeffs.end());
		}

		void get_bdf_parts(
			const int bdf_order,
			const int index,
			const std::vector<Eigen::MatrixXd> &adjoint_p,
			const std::vector<Eigen::MatrixXd> &adjoint_nu,
			Eigen::MatrixXd &sum_adjoint_p,
			Eigen::MatrixXd &sum_adjoint_nu,
			double &beta)
		{
			sum_adjoint_p.setZero(adjoint_p[0].size(), 1);
			sum_adjoint_nu.setZero(adjoint_nu[0].size(), 1);

			int num = std::min(bdf_order, int(adjoint_p.size()) - 2 - index);
			for (int j = 1; j <= num; ++j)
			{
				int order = std::min(bdf_order, index + j);
				sum_adjoint_p += -time_integrator::BDF::alphas(order - 1)[j - 1] * adjoint_p[index + j];
				sum_adjoint_nu += -time_integrator::BDF::alphas(order - 1)[j - 1] * adjoint_nu[index + j];
			}
			int order = std::min(bdf_order, index);
			if (order >= 1)
				beta = time_integrator::BDF::betas(order - 1);
			else
				beta = std::nan("");
		}
	} // namespace

	void State::perturb_mesh(const Eigen::MatrixXd &perturbation)
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;

		get_vf(V, F);
		V.conservativeResize(V.rows(), mesh->dimension());

		V += utils::unflatten(perturbation, V.cols());

		set_v(V);
	}
	void State::perturb_material(const Eigen::MatrixXd &perturbation)
	{
		const auto &cur_lambdas = assembler.lame_params().lambda_mat_;
		const auto &cur_mus = assembler.lame_params().mu_mat_;

		Eigen::MatrixXd lambda_update(cur_lambdas.size(), 1), mu_update(cur_mus.size(), 1);
		for (int i = 0; i < lambda_update.size(); i++)
		{
			lambda_update(i) = perturbation(i);
			mu_update(i) = perturbation(i + lambda_update.size());
		}

		assembler.update_lame_params(cur_lambdas + lambda_update, cur_mus + mu_update);
	}

	void State::get_vf(Eigen::MatrixXd &vertices, Eigen::MatrixXi &faces, const bool geometric) const
	{
		const auto &cur_bases = geometric ? geom_bases() : bases;
		const int n_elements = int(cur_bases.size());
		vertices = Eigen::MatrixXd::Zero(geometric ? n_geom_bases : n_bases, 3);
		faces = Eigen::MatrixXi::Zero(cur_bases.size(), cur_bases[0].bases.size());
		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < n_elements; ++e)
		{
			const int n_loc_bases = cur_bases[e].bases.size();
			for (int i = 0; i < n_loc_bases; ++i)
			{
				const auto &v = cur_bases[e].bases[i];
				assert(v.global().size() == 1);

				vertices.block(v.global()[0].index, 0, 1, mesh->dimension()) = v.global()[0].node;
				faces(e, i) = v.global()[0].index;
			}
		}
	}

	void State::set_v(const Eigen::MatrixXd &vertices)
	{
		auto &gbases = geom_bases();
		const int n_elements = int(gbases.size());
		for (int e = 0; e < n_elements; ++e)
		{
			int n_loc_bases = gbases[e].bases.size();
			for (int i = 0; i < n_loc_bases; ++i)
			{
				auto &v = gbases[e].bases[i];
				assert(v.global().size() == 1);

				v.global()[0].node = vertices.block(v.global()[0].index, 0, 1, mesh->dimension());
			}

			if (!iso_parametric())
			{
				n_loc_bases = bases[e].bases.size();
				Eigen::MatrixXd local_pts, pts;
				if (!mesh->is_volume())
					autogen::p_nodes_2d(bases[e].bases.front().order(), local_pts);
				else
					autogen::p_nodes_3d(bases[e].bases.front().order(), local_pts);
				gbases[e].eval_geom_mapping(local_pts, pts);
				for (int i = 0; i < n_loc_bases; ++i)
				{
					auto &v = bases[e].bases[i];
					assert(v.global().size() == 1);

					v.global()[0].node = pts.block(i, 0, 1, mesh->dimension());
				}
			}
		}

		{
			const auto &primitive_to_node = iso_parametric() ? primitive_to_bases_node : primitive_to_geom_bases_node;
			for (int v = 0; v < mesh->n_vertices(); v++)
				if (primitive_to_node[v] >= 0 && primitive_to_node[v] < vertices.rows())
					mesh->set_point(v, vertices.block(primitive_to_node[v], 0, 1, mesh->dimension()));
		}

		// update assembly cache
		ass_vals_cache.clear();
		pressure_ass_vals_cache.clear();
		mass_ass_vals_cache.clear();
		if (n_bases <= args["cache_size"])
		{
			ass_vals_cache.init(mesh->is_volume(), bases, gbases);
			mass_ass_vals_cache.init(mesh->is_volume(), bases, bases, true);
			if (assembler.is_mixed(formulation()))
				pressure_ass_vals_cache.init(mesh->is_volume(), pressure_bases, gbases);
		}

		build_collision_mesh(boundary_nodes_pos, collision_mesh, n_bases, bases);
	}

	double State::J_transient_step(const IntegrableFunctional &j, const int step)
	{
		const auto &gbases = geom_bases();
		const double dt = args["time"]["dt"];

		double result = 0;
		if (j.is_volume_integral())
		{
			VolumeIntegrandTerms integrand_functions = {
				[&](int e, const Eigen::MatrixXd &reference_points, const Eigen::MatrixXd &global_points, Eigen::VectorXd &vec_term) {
					Eigen::MatrixXd u, grad_u;
					io::Evaluator::interpolate_at_local_vals(*mesh, problem->is_scalar(), bases, gbases, e, reference_points, diff_cached[step].u, u, grad_u);
					Eigen::MatrixXd vec_term_mat;
					json params = {};
					params["elem"] = e;
					params["body_id"] = mesh->get_body_id(e);
					params["t"] = dt * step;
					params["step"] = step;
					j.evaluate(assembler.lame_params(), reference_points, global_points, u, grad_u, params, vec_term_mat);
					assert(vec_term.cols() == 1);
					vec_term = vec_term_mat;
				}};

			volume_integral(bases, gbases, integrand_functions, *mesh, result);
		}
		else
		{
			SurfaceIntegrandTerms integrand_functions = {
				[&](int e, const Eigen::VectorXi &global_primitive_ids, const Eigen::MatrixXd &reference_points, const Eigen::MatrixXd &global_points, const Eigen::MatrixXd &normals, Eigen::VectorXd &vec_term) {
					Eigen::MatrixXd u, grad_u;
					io::Evaluator::interpolate_at_local_vals(*mesh, problem->is_scalar(), bases, gbases, e, reference_points, diff_cached[step].u, u, grad_u);
					std::vector<int> boundary_ids = {};
					for (int i = 0; i < global_primitive_ids.size(); ++i)
						boundary_ids.push_back(mesh->get_boundary_id(global_primitive_ids(i)));
					json params = {};
					params["elem"] = e;
					params["body_id"] = mesh->get_body_id(e);
					params["boundary_ids"] = boundary_ids;
					params["t"] = dt * step;
					params["step"] = step;
					Eigen::MatrixXd vec_term_mat;
					j.evaluate(assembler.lame_params(), reference_points, global_points, u, grad_u, params, vec_term_mat);
					assert(vec_term.cols() == 1);
					vec_term = vec_term_mat;
				}};

			surface_integral(total_local_boundary, args["space"]["advanced"]["quadrature_order"], bases, gbases, integrand_functions, *mesh, result);
		}

		return result;
	}

	void State::cache_transient_adjoint_quantities(const int current_step, const Eigen::MatrixXd &sol)
	{
		StiffnessMatrix gradu_h(sol.size(), sol.size()), gradu_h_prev(sol.size(), sol.size());
		if (current_step == 0)
			diff_cached.clear();
		if (problem->is_time_dependent())
		{
			if (current_step > 0)
				compute_force_hessian(sol, gradu_h, gradu_h_prev);

			if (diff_cached.size() > 0)
				diff_cached.back().gradu_h_next = gradu_h_prev;
		}
		else
			compute_force_hessian(sol, gradu_h, gradu_h_prev);
		
		auto cur_contact_set = solve_data.contact_form ? solve_data.contact_form->get_constraint_set() : ipc::Constraints();
		auto cur_friction_set = solve_data.friction_form ? solve_data.friction_form->get_friction_constraint_set() : ipc::FrictionConstraints();
		diff_cached.push_back({gradu_h, StiffnessMatrix(sol.size(), sol.size()), sol, cur_contact_set, cur_friction_set});
	}

	void State::compute_force_hessian(const Eigen::MatrixXd &sol, StiffnessMatrix &hessian, StiffnessMatrix &hessian_prev) const
	{
		if (assembler.is_linear(formulation()) && !is_contact_enabled())
		{
			hessian = stiffness;
			hessian_prev = StiffnessMatrix(stiffness.rows(), stiffness.cols());
		}
		else
		{
			solve_data.nl_problem->FullNLProblem::solution_changed(sol);
			compute_force_hessian_nonlinear(sol, hessian, hessian_prev);
		}
	}

	void State::compute_force_hessian_nonlinear(const Eigen::MatrixXd &sol, StiffnessMatrix &hessian, StiffnessMatrix &hessian_prev) const
	{
		solve_data.nl_problem->FullNLProblem::hessian(sol, hessian);
		if (problem->is_time_dependent())
		{
			hessian -= mass;
			hessian /= sqrt(solve_data.time_integrator->acceleration_scaling());
		}

		hessian_prev = StiffnessMatrix(sol.size(), sol.size());
		if (problem->is_time_dependent() && diff_cached.size() > 0)
		{
			if (solve_data.friction_form)
			{
				Eigen::MatrixXd displaced = boundary_nodes_pos + utils::unflatten(sol, mesh->dimension());

				Eigen::MatrixXd displaced_prev;
				if (diff_cached.size())
					displaced_prev = boundary_nodes_pos + utils::unflatten(diff_cached.back().u, mesh->dimension());
				else
					displaced_prev = displaced;

				Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(displaced_prev - boundary_nodes_pos);
				Eigen::MatrixXd surface_solution = collision_mesh.vertices(displaced - boundary_nodes_pos);

				hessian_prev = -ipc::compute_friction_force_jacobian(
					collision_mesh,
					collision_mesh.vertices_at_rest(),
					surface_solution_prev,
					surface_solution,
					solve_data.friction_form->get_friction_constraint_set(),
					solve_data.contact_form->dhat(), solve_data.contact_form->barrier_stiffness(), solve_data.friction_form->epsv_dt(),
					ipc::FrictionConstraint::DiffWRT::Ut);
				hessian_prev = collision_mesh.to_full_dof(hessian_prev) / solve_data.time_integrator->acceleration_scaling();
			}

			if (assembler.has_damping())
			{
				utils::SpareMatrixCache mat_cache;
				StiffnessMatrix damping_hessian_prev(sol.size(), sol.size());
				assembler.assemble_energy_hessian("DampingPrev", mesh->is_volume(), n_bases, false, bases, geom_bases(), ass_vals_cache, solve_data.time_integrator->dt(), sol, diff_cached.back().u, mat_cache, damping_hessian_prev);

				hessian_prev += damping_hessian_prev;
			}
		}
	}

	void State::compute_adjoint_rhs(const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params)> &grad_j, const Eigen::MatrixXd &solution, Eigen::VectorXd &b, bool only_surface)
	{
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();
		b = Eigen::MatrixXd::Zero(n_bases * actual_dim, 1);

		const auto &gbases = geom_bases();

		const int n_elements = int(bases.size());
		if (!only_surface)
		{
			auto storage = utils::create_thread_storage(LocalThreadVecStorage(b.size()));

			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);
					// vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					const int n_loc_bases_ = int(vals.basis_values.size());
					Eigen::MatrixXd u, grad_u;
					io::Evaluator::interpolate_at_local_vals(e, mesh->dimension(), actual_dim, vals, solution, u, grad_u);

					json params = {};
					params["elem"] = e;
					params["body_id"] = mesh->get_body_id(e);
					Eigen::MatrixXd result_ = grad_j(quadrature.points, vals.val, u, grad_u, params);
					assert(result_.size() > 0);

					for (int i = 0; i < n_loc_bases_; ++i)
					{
						const assembler::AssemblyValues &v = vals.basis_values[i];
						assert(v.global.size() == 1);
						for (int d = 0; d < actual_dim; d++)
						{
							double val = 0;

							// j = j(x, grad u)
							if (result_.cols() == grad_u.cols())
							{
								for (int q = 0; q < local_storage.da.size(); ++q)
								{
									Eigen::Matrix<double, -1, -1, Eigen::RowMajor> grad_phi;
									grad_phi.setZero(actual_dim, mesh->dimension());
									grad_phi.row(d) = v.grad_t_m.row(q);
									for (int k = 0; k < result_.cols(); k++)
										val += result_(q, k) * grad_phi(k) * local_storage.da(q);
								}
							}
							// j = j(x, u)
							else
							{
								for (int q = 0; q < local_storage.da.size(); ++q)
									val += result_(q, d) * v.val(q) * local_storage.da(q);
							}
							local_storage.vec(v.global[0].index * actual_dim + d) += val;
						}
					}
				}
			});

			for (const LocalThreadVecStorage &local_storage : storage)
				b += local_storage.vec;
		}
		else
		{
			Eigen::MatrixXd uv, samples, gtmp;
			Eigen::VectorXi global_primitive_ids;
			Eigen::MatrixXd points, normals;
			Eigen::VectorXd weights;

			for (const auto &lb : total_local_boundary)
			{
				const int e = lb.element_id();
				bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, args["space"]["advanced"]["quadrature_order"], *mesh, false, uv, points, normals, weights, global_primitive_ids);

				if (!has_samples)
					continue;

				assembler::ElementAssemblyValues vals;
				vals.compute(e, mesh->is_volume(), points, bases[e], gbases[e]);

				const Eigen::VectorXd da = weights.array();

				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					normals.row(n) = normals.row(n) * vals.jac_it[n];
					normals.row(n).normalize();
				}

				const int n_loc_bases_ = int(vals.basis_values.size());
				Eigen::MatrixXd u, grad_u;
				io::Evaluator::interpolate_at_local_vals(e, mesh->dimension(), actual_dim, vals, solution, u, grad_u);

				std::vector<int> boundary_ids = {};
				for (int i = 0; i < global_primitive_ids.size(); ++i)
					boundary_ids.push_back(mesh->get_boundary_id(global_primitive_ids(i)));

				json params = {};
				params["elem"] = e;
				params["body_id"] = mesh->get_body_id(e);
				params["boundary_ids"] = boundary_ids;
				Eigen::MatrixXd result_ = grad_j(points, vals.val, u, grad_u, params);
				assert(result_.size() > 0);

				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const auto nodes = bases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

					for (long n = 0; n < nodes.size(); ++n)
					{
						const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
						assert(v.global.size() == 1);
						for (int d = 0; d < actual_dim; d++)
						{
							double val = 0;

							// j = j(x, grad u)
							if (result_.cols() == grad_u.cols())
							{
								for (int q = 0; q < da.size(); ++q)
								{
									for (int k = 0; k < mesh->dimension(); k++)
										val += result_(q, d * mesh->dimension() + k) * v.grad_t_m(q, k) * da(q);
								}
							}
							// j = j(x, u)
							else
							{
								for (int q = 0; q < da.size(); ++q)
									val += result_(q, d) * v.val(q) * da(q);
							}
							b(v.global[0].index * actual_dim + d) += val;
						}
					}
				}
			}
		}
	}

	void State::solve_adjoint(const Eigen::VectorXd &adjoint_rhs, Eigen::MatrixXd &adjoint_solution) const
	{
		StiffnessMatrix A;
		Eigen::VectorXd b = adjoint_rhs;

		{
			StiffnessMatrix unused;
			compute_force_hessian(diff_cached[0].u, A, unused);
		}
		
		if (lin_solver_cached)
		{
			for (int i : boundary_nodes)
				b(i) = 0;
			Eigen::VectorXd x;
			dirichlet_solve_prefactorized(*lin_solver_cached, A, b, boundary_nodes, x);
			adjoint_solution = x;
		}
		else
			solve_zero_dirichlet(args["solver"]["linear"], A, b, boundary_nodes, adjoint_solution);
	}

	void State::solve_adjoint(const IntegrableFunctional &j, Eigen::MatrixXd &adjoint_solution)
	{
		StiffnessMatrix A;
		Eigen::VectorXd b;

		if (!j.depend_on_u() && !j.depend_on_gradu())
		{
			const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();
			adjoint_solution.setZero(actual_dim * n_bases, 1);
			return;
		}

		{
			StiffnessMatrix unused;
			compute_force_hessian(diff_cached[0].u, A, unused);
		}

		compute_adjoint_rhs(
			[&](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params) 
			{
				return j.grad_j(assembler.lame_params(), local_pts, pts, u, grad_u, params); 
			}, diff_cached[0].u, b, j.is_surface_integral());

		if (lin_solver_cached)
		{
			for (int i : boundary_nodes)
				b(i) = 0;
			Eigen::VectorXd x;
			dirichlet_solve_prefactorized(*lin_solver_cached, A, b, boundary_nodes, x);
			adjoint_solution = x;
		}
		else
			solve_zero_dirichlet(args["solver"]["linear"], A, b, boundary_nodes, adjoint_solution);
	}

	void State::compute_shape_derivative_functional_term(const Eigen::MatrixXd &solution, const IntegrableFunctional &j, Eigen::VectorXd &term, const int cur_time_step) const
	{
		const auto &gbases = geom_bases();
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();

		const int n_elements = int(bases.size());
		term.setZero(n_geom_bases * mesh->dimension(), 1);

		if (j.is_volume_integral())
		{
			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					ass_vals_cache.compute(e, mesh->is_volume(), bases[e], gbases[e], vals);
					assembler::ElementAssemblyValues gvals;
					gvals.compute(e, mesh->is_volume(), vals.quadrature.points, gbases[e], gbases[e]);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u;
					io::Evaluator::interpolate_at_local_vals(e, mesh->dimension(), actual_dim, vals, solution, u, grad_u);

					Eigen::MatrixXd j_value;
					json params = {};
					params["elem"] = e;
					params["body_id"] = mesh->get_body_id(e);
					params["step"] = cur_time_step;
					j.evaluate(assembler.lame_params(), quadrature.points, gvals.val, u, grad_u, params, j_value);

					Eigen::MatrixXd grad_j_value;
					if (j.depend_on_gradu() || j.depend_on_u())
						grad_j_value = j.grad_j(assembler.lame_params(), quadrature.points, gvals.val, u, grad_u, params);

					Eigen::MatrixXd dj_dx;
					if (j.depend_on_x())
						j.dj_dx(assembler.lame_params(), quadrature.points, gvals.val, u, grad_u, params, dj_dx);

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						for (auto &v : gvals.basis_values)
						{
							for (int d = 0; d < mesh->dimension(); d++)
							{

								local_storage.vec(v.global[0].index * mesh->dimension() + d) += j_value(q) * v.grad_t_m(q, d) * local_storage.da(q);
								if (j.depend_on_x())
								{
									local_storage.vec(v.global[0].index * mesh->dimension() + d) += (v.val(q) * dj_dx(q, d)) * local_storage.da(q);
								}
								if (j.depend_on_gradu())
								{
									Eigen::MatrixXd tau_q;
									if (actual_dim == mesh->dimension())
										vector2matrix(grad_j_value.row(q), tau_q);
									else
										tau_q = grad_j_value.row(q);

									Eigen::MatrixXd grad_u_q;
									if (mesh->dimension() == actual_dim)
										vector2matrix(grad_u.row(q), grad_u_q);
									else
										grad_u_q = grad_u.row(q);
									local_storage.vec(v.global[0].index * mesh->dimension() + d) += -dot(tau_q, grad_u_q.col(d) * v.grad_t_m.row(q)) * local_storage.da(q);
								}
							}
						}
					}
				}
			});

			for (const LocalThreadVecStorage &local_storage : storage)
				term += local_storage.vec;
		}
		else
		{
			assert(!iso_parametric() || disc_orders[0] == 1);

			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

			utils::maybe_parallel_for(total_local_boundary.size(), [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				Eigen::VectorXi global_primitive_ids;
				Eigen::MatrixXd uv, points, normals;
				Eigen::VectorXd weights;

				for (int lb_id = start; lb_id < end; ++lb_id)
				{
					const auto &lb = total_local_boundary[lb_id];
					const int e = lb.element_id();

					bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, args["space"]["advanced"]["n_boundary_samples"], *mesh, false, uv, points, normals, weights, global_primitive_ids);

					if (!has_samples)
						continue;

					const ElementBases &gbs = gbases[e];
					const ElementBases &bs = bases[e];

					assembler::ElementAssemblyValues vals;
					vals.compute(e, mesh->is_volume(), points, gbs, gbs);

					const Eigen::VectorXd da = weights.array();

					for (int n = 0; n < vals.jac_it.size(); ++n)
					{
						normals.row(n) = normals.row(n) * vals.jac_it[n];
						normals.row(n).normalize();
					}

					Eigen::MatrixXd u, grad_u;
					// io::Evaluator::interpolate_at_local_vals(*mesh, e, points, solution, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(*mesh, problem->is_scalar(), bases, gbases, e, points, solution, u, grad_u);

					std::vector<int> boundary_ids = {};
					for (int i = 0; i < global_primitive_ids.size(); ++i)
						boundary_ids.push_back(mesh->get_boundary_id(global_primitive_ids(i)));

					Eigen::MatrixXd j_value;
					json params = {};
					params["elem"] = e;
					params["body_id"] = mesh->get_body_id(e);
					params["boundary_ids"] = boundary_ids;
					params["step"] = cur_time_step;
					j.evaluate(assembler.lame_params(), points, vals.val, u, grad_u, params, j_value);

					Eigen::MatrixXd grad_j_value;
					if (j.depend_on_gradu())
						grad_j_value = j.grad_j(assembler.lame_params(), points, vals.val, u, grad_u, params);

					Eigen::MatrixXd dj_dx;
					if (j.depend_on_x())
						j.dj_dx(assembler.lame_params(), points, vals.val, u, grad_u, params, dj_dx);

					const int n_samples_per_surface = da.size() / lb.size();

					for (int i = 0; i < lb.size(); ++i)
					{
						const int primitive_global_id = lb.global_primitive_id(i);
						const auto nodes = gbs.local_nodes_for_primitive(primitive_global_id, *mesh);

						assert(nodes.size() == mesh->dimension());
						for (long n = 0; n < nodes.size(); ++n)
						{
							const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
							// integrate j * div(gbases) over the whole boundary
							for (int d = 0; d < mesh->dimension(); d++)
							{
								double velocity_div = 0;
								if (mesh->is_volume())
								{
									Eigen::Vector3d dr_du = gbs.bases[nodes(1)].global()[0].node - gbs.bases[nodes(0)].global()[0].node;
									Eigen::Vector3d dr_dv = gbs.bases[nodes(2)].global()[0].node - gbs.bases[nodes(0)].global()[0].node;

									// compute dtheta
									Eigen::Vector3d dtheta_du, dtheta_dv;
									dtheta_du.setZero();
									dtheta_dv.setZero();
									if (0 == n)
									{
										dtheta_du(d) = -1;
										dtheta_dv(d) = -1;
									}
									else if (1 == n)
										dtheta_du(d) = 1;
									else if (2 == n)
										dtheta_dv(d) = 1;
									else
										assert(false);

									velocity_div = (dr_du.cross(dr_dv)).dot(dtheta_du.cross(dr_dv) + dr_du.cross(dtheta_dv)) / (dr_du.cross(dr_dv)).squaredNorm();
								}
								else
								{
									Eigen::VectorXd dr = gbs.bases[nodes(1)].global()[0].node - gbs.bases[nodes(0)].global()[0].node;

									// compute dtheta
									Eigen::VectorXd dtheta;
									dtheta.setZero(dr.rows(), dr.cols());
									if (0 == n)
										dtheta(d) = -1;
									else if (1 == n)
										dtheta(d) = 1;
									else
										assert(false);

									velocity_div = dr.dot(dtheta) / dr.squaredNorm();
								}

								for (int q = n_samples_per_surface * i; q < n_samples_per_surface * (i + 1); ++q)
								{
									local_storage.vec(v.global[0].index * mesh->dimension() + d) += j_value(q) * velocity_div * da(q);
									if (j.depend_on_x())
									{
										local_storage.vec(v.global[0].index * mesh->dimension() + d) += (v.val(q) * dj_dx(q, d)) * da(q);
									}
									if (j.depend_on_gradu())
									{
										Eigen::MatrixXd tau_i;
										if (actual_dim == mesh->dimension())
											vector2matrix(grad_j_value.row(q), tau_i);
										else
											tau_i = grad_j_value.row(q);
										Eigen::MatrixXd grad_u_i;
										if (mesh->dimension() == actual_dim)
											vector2matrix(grad_u.row(q), grad_u_i);
										else
											grad_u_i = grad_u.row(q);
										local_storage.vec(v.global[0].index * mesh->dimension() + d) += -dot(tau_i, grad_u_i.col(d) * v.grad_t_m.row(q)) * da(q);
									}
								}
							}
						}
					}
				}
			});

			for (const LocalThreadVecStorage &local_storage : storage)
				term += local_storage.vec;
		}
	}


	void State::solve_transient_adjoint(const std::vector<Eigen::VectorXd> &adjoint_rhs, std::vector<Eigen::MatrixXd> &adjoint_nu, std::vector<Eigen::MatrixXd> &adjoint_p, bool dirichlet_derivative) const
	{
		const int bdf_order = get_bdf_order();
		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];
		const auto &gbases = geom_bases();

		adjoint_p.assign(time_steps + 2, Eigen::MatrixXd::Zero(diff_cached[0].u.size(), 1));
		adjoint_nu.assign(time_steps + 2, Eigen::MatrixXd::Zero(diff_cached[0].u.size(), 1));

		// set dirichlet rows of mass to identity
		StiffnessMatrix reduced_mass;
		replace_rows_by_identity(reduced_mass, mass, boundary_nodes);

		Eigen::MatrixXd sum_alpha_p, sum_alpha_nu;
		for (int i = time_steps; i >= 0; --i)
		{
			double beta;
			get_bdf_parts(bdf_order, i, adjoint_p, adjoint_nu, sum_alpha_p, sum_alpha_nu, beta);
			double beta_dt = beta * dt;

			StiffnessMatrix gradu_h_next;
			replace_rows_by_zero(gradu_h_next, -beta_dt * diff_cached[i].gradu_h_next, boundary_nodes);

			if (i > 0)
			{
				StiffnessMatrix gradu_h;
				replace_rows_by_zero(gradu_h, -diff_cached[i].gradu_h, boundary_nodes);
				StiffnessMatrix A = (reduced_mass - beta_dt * gradu_h).transpose();
				Eigen::VectorXd rhs_ = -reduced_mass.transpose() * sum_alpha_nu - gradu_h.transpose() * sum_alpha_p + gradu_h_next.transpose() * adjoint_p[i + 1] - adjoint_rhs[i];
				for (const auto &b : boundary_nodes)
				{
					rhs_(b) += (1. / beta_dt) * (-2 * adjoint_p[i + 1](b));
					if ((i + 2) < adjoint_p.size() - 1)
						rhs_(b) += (1. / beta_dt) * adjoint_p[i + 2](b);
				}
				
				{
					StiffnessMatrix A_tmp = A;
					Eigen::VectorXd b_ = rhs_;
					solve_zero_dirichlet(args["solver"]["linear"], A_tmp, b_, boundary_nodes, adjoint_nu[i]);
				}

				if (dirichlet_derivative)
				{
					Eigen::VectorXd tmp = rhs_ - A * adjoint_nu[i];
					for (const auto &b : boundary_nodes)
					{
						adjoint_nu[i](b) = tmp(b);
					}
				}
				adjoint_p[i] = beta_dt * adjoint_nu[i] - sum_alpha_p;
			}
			else
			{
				adjoint_p[i] = -reduced_mass.transpose() * sum_alpha_p;
				adjoint_nu[i] = -adjoint_rhs[i] - reduced_mass.transpose() * sum_alpha_nu + beta_dt * diff_cached[i].gradu_h_next.transpose() * adjoint_p[i + 1]; // adjoint_nu[0] actually stores adjoint_mu[0]
			}
		}
	}

	void State::solve_transient_adjoint(const IntegrableFunctional &j, std::vector<Eigen::MatrixXd> &adjoint_nu, std::vector<Eigen::MatrixXd> &adjoint_p, bool dirichlet_derivative)
	{
		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		std::vector<Eigen::VectorXd> adjoint_rhs;
		adjoint_rhs.resize(time_steps + 1);

		std::vector<double> weights;
		j.get_transient_quadrature_weights(time_steps, dt, weights);
		for (int i = time_steps; i >= 0; --i)
		{
			auto grad_j_func = [&](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const json &params) {
				json params_extended = params;
				params_extended["step"] = i;
				params_extended["t"] = i * dt;
				return j.grad_j(assembler.lame_params(), local_pts, pts, u, grad_u, params_extended);
			};
			Eigen::VectorXd gradu_j;
			compute_adjoint_rhs(grad_j_func, diff_cached[i].u, gradu_j, j.is_surface_integral());
			adjoint_rhs[i] = gradu_j * weights[i];
		}

		solve_transient_adjoint(adjoint_rhs, adjoint_nu, adjoint_p, dirichlet_derivative);
	}

	void State::solve_transient_adjoint(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, std::vector<Eigen::MatrixXd> &adjoint_nu, std::vector<Eigen::MatrixXd> &adjoint_p)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());
		assert(js.size() > 0);

		int bdf_order = get_bdf_order();

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];
		const auto &gbases = geom_bases();

		adjoint_p.assign(time_steps + 2, Eigen::MatrixXd::Zero(diff_cached[0].u.size(), 1));
		adjoint_nu.assign(time_steps + 2, Eigen::MatrixXd::Zero(diff_cached[0].u.size(), 1));

		// set dirichlet rows of mass to identity
		StiffnessMatrix reduced_mass;
		replace_rows_by_identity(reduced_mass, mass, boundary_nodes);

		// different j should use the same quadrature rule in time
		std::vector<double> weights(js.size());
		js[0].get_transient_quadrature_weights(time_steps, dt, weights);

		Eigen::MatrixXd sum_alpha_p, sum_alpha_nu;
		double beta, beta_dt;
		for (int i = time_steps; i >= 0; --i)
		{
			get_bdf_parts(bdf_order, i, adjoint_p, adjoint_nu, sum_alpha_p, sum_alpha_nu, beta);
			beta_dt = beta * dt;

			StiffnessMatrix gradu_h_next;
			replace_rows_by_identity(gradu_h_next, -beta_dt * diff_cached[i].gradu_h_next, boundary_nodes);

			Eigen::VectorXd dJ_du;
			dJ_du.setZero(gradu_h_next.rows());

			Eigen::VectorXd integrals(js.size());
			for (int k = 0; k < js.size(); k++)
				integrals(k) = J_transient_step(js[k], i);
			json param;
			param["t"] = i * dt;
			param["step"] = i;
			Eigen::VectorXd outer_grad = dJi_dintegrals(integrals, param);

			for (int k = 0; k < js.size(); k++)
			{
				if (js[k].depend_on_u() || js[k].depend_on_gradu())
				{
					Eigen::VectorXd dJk_du;
					compute_adjoint_rhs([&](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, json &params) {
						json params_extended = params;
						params_extended["step"] = i;
						params_extended["t"] = i * args["time"]["dt"].get<double>();
						return js[k].grad_j(assembler.lame_params(), local_pts, pts, u, grad_u, params_extended);
					},
										diff_cached[i].u, dJk_du, js[k].is_surface_integral());

					dJ_du += dJk_du * outer_grad(k);
				}
			}
			dJ_du *= weights[i];

			if (i > 0)
			{
				StiffnessMatrix gradu_h;
				replace_rows_by_identity(gradu_h, -diff_cached[i].gradu_h, boundary_nodes);
				StiffnessMatrix A = (reduced_mass - beta_dt * gradu_h).transpose();
				Eigen::VectorXd rhs_ = -reduced_mass.transpose() * sum_alpha_nu - gradu_h.transpose() * sum_alpha_p + gradu_h_next.transpose() * adjoint_p[i + 1] - dJ_du;
				solve_zero_dirichlet(args["solver"]["linear"], A, rhs_, boundary_nodes, adjoint_nu[i]);
				adjoint_p[i] = beta_dt * adjoint_nu[i] - sum_alpha_p;
			}
			else
			{
				adjoint_p[i] = -reduced_mass.transpose() * sum_alpha_p;
				adjoint_nu[i] = -dJ_du - reduced_mass.transpose() * sum_alpha_nu + beta_dt * diff_cached[i].gradu_h_next.transpose() * adjoint_p[i + 1]; // adjoint_nu[0] actually stores adjoint_mu[0]
			}
		}
	}

	double State::J_transient(const std::vector<IntegrableFunctional> &js, const std::function<double(const Eigen::VectorXd &, const json &)> &Ji)
	{
		assert(problem->is_time_dependent());
		assert(js.size() > 0);
		const double dt = args["time"]["dt"];
		const int n_steps = args["time"]["time_steps"];
		double result = 0;

		std::vector<double> weights;
		js[0].get_transient_quadrature_weights(n_steps, dt, weights);

		json param;
		for (int i = 0; i <= n_steps; ++i)
		{
			param["t"] = i * args["time"]["dt"].get<double>();
			param["step"] = i;

			Eigen::VectorXd integrals(js.size());
			for (int k = 0; k < js.size(); k++)
				integrals(k) = J_transient_step(js[k], i);

			result += Ji(integrals, param) * weights[i];
		}

		return result;
	}

	void State::dJ_material_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(js, dJi_dintegrals, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		int bdf_order = get_bdf_order();

		Eigen::VectorXd elasticity_term;
		one_form.setZero(bases.size() * 2);

		for (int i = time_steps; i > 0; --i)
		{
			const int real_order = std::min(bdf_order, i);
			double beta = time_integrator::BDF::betas(real_order - 1);

			solve_data.elastic_form->foce_material_derivative(diff_cached[i].u, diff_cached[i].u, -adjoint_p[i], elasticity_term);
			one_form += beta * dt * elasticity_term;
		}
	}
	void State::dJ_friction_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, double &one_form)
	{
		assert(problem->is_time_dependent());
		assert(is_contact_enabled());
		assert(args["contact"]["friction_coefficient"].get<double>() > 0);

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(js, dJi_dintegrals, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const double mu = args["contact"]["friction_coefficient"].get<double>();
		const int time_steps = args["time"]["time_steps"];
		const int problem_dim = mesh->dimension();

		int bdf_order = get_bdf_order();

		one_form = 0;

		auto storage = utils::create_thread_storage(LocalThreadScalarStorage());

		utils::maybe_parallel_for(time_steps, [&](int start, int end, int thread_id) {
			LocalThreadScalarStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);
			for (int t_aux = start; t_aux < end; ++t_aux)
			{
				const int t = time_steps - t_aux;
				const int real_order = std::min(bdf_order, t);
				double beta = time_integrator::BDF::betas(real_order - 1);

				const ipc::FrictionConstraints &friction_constraint_set = diff_cached[t].friction_constraint_set;
				const auto &solution = diff_cached[t].u;
				const auto &prev_solution = diff_cached[t - 1].u;

				Eigen::MatrixXd solution_(n_bases, problem_dim), prev_solution_(n_bases, problem_dim);
				for (int i = 0; i < n_bases; ++i)
					for (int d = 0; d < problem_dim; d++)
					{
						solution_(i, d) = solution(i * problem_dim + d);
						prev_solution_(i, d) = prev_solution(i * problem_dim + d);
					}

				Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(prev_solution_);
				Eigen::MatrixXd surface_solution = collision_mesh.vertices(solution_);

				auto force = -ipc::compute_friction_force(collision_mesh, collision_mesh.vertices_at_rest(), surface_solution_prev, surface_solution, friction_constraint_set, solve_data.contact_form->dhat(), solve_data.contact_form->barrier_stiffness(), solve_data.friction_form->epsv_dt());

				local_storage.val += (adjoint_p[t].array() * collision_mesh.to_full_dof(force).array()).sum() / (beta * mu * dt);
			}
		});

		for (const LocalThreadScalarStorage &local_storage : storage)
			one_form += local_storage.val;
	}
	void State::dJ_damping_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(is_contact_enabled());

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(js, dJi_dintegrals, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		int bdf_order = get_bdf_order();

		Eigen::VectorXd damping_term;
		one_form.setZero(2);

		for (int t = time_steps; t > 0; --t)
		{
			const int real_order = std::min(bdf_order, t);
			double beta = time_integrator::BDF::betas(real_order - 1);

			solve_data.damping_form->foce_material_derivative(diff_cached[t].u, diff_cached[t - 1].u, -adjoint_p[t], damping_term);
			one_form += beta * dt * damping_term;
		}
	}
	void State::dJ_initial_condition(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(js, dJi_dintegrals, adjoint_nu, adjoint_p);

		one_form.setZero(n_bases * mesh->dimension() * 2); // half for initial solution, half for initial velocity

		// \partial_q \hat{J}^0 - p_0^T \partial_q g^v - \mu_0^T \partial_q g^u
		one_form.block(0, 0, n_bases * mesh->dimension(), 1) = -adjoint_nu[0]; // adjoint_nu[0] actually stores adjoint_mu[0]
		one_form.block(n_bases * mesh->dimension(), 0, n_bases * mesh->dimension(), 1) = -adjoint_p[0];
	}

	void State::dJ_shape_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(js, dJi_dintegrals, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		int bdf_order = get_bdf_order();

		Eigen::VectorXd elasticity_term, rhs_term, damping_term, mass_term, contact_term, friction_term, functional_term;
		one_form.setZero(n_geom_bases * mesh->dimension());
		functional_term.resize(n_geom_bases * mesh->dimension());

		std::vector<double> weights;
		js[0].get_transient_quadrature_weights(time_steps, dt, weights);
		for (int i = time_steps; i > 0; --i)
		{
			const int real_order = std::min(bdf_order, i);
			double beta = time_integrator::BDF::betas(real_order - 1);
			double beta_dt = beta * dt;

			ipc::FrictionConstraints friction_constraint_set;
			friction_constraint_set = diff_cached[i].friction_constraint_set;

			Eigen::MatrixXd velocity = diff_cached[i].u;
			for (int o = 1; o <= real_order; o++)
				velocity -= time_integrator::BDF::alphas(real_order - 1)[o - 1] * diff_cached[i - o].u;
			velocity /= beta_dt;

			Eigen::VectorXd integrals(js.size());
			for (int k = 0; k < js.size(); k++)
				integrals(k) = J_transient_step(js[k], i);
			json param;
			param["t"] = i * dt;
			param["step"] = i;
			Eigen::VectorXd outer_grad = dJi_dintegrals(integrals, param);

			functional_term.setZero();
			for (int k = 0; k < js.size(); k++)
			{
				Eigen::VectorXd functional_term_k;
				compute_shape_derivative_functional_term(diff_cached[i].u, js[k], functional_term_k, i);

				functional_term += functional_term_k * outer_grad(k);
			}

			solve_data.inertia_form->force_shape_derivative(mesh->is_volume(), n_geom_bases, bases, geom_bases(), assembler, mass_ass_vals_cache, velocity, adjoint_nu[i], mass_term);
			solve_data.elastic_form->force_shape_derivative(n_geom_bases, diff_cached[i].u, diff_cached[i].u, -adjoint_p[i], elasticity_term);
			// compute_shape_derivative_rhs_term(diff_cached[i].u, -adjoint_p[i], rhs_term);
			solve_data.body_form->force_shape_derivative(n_geom_bases, diff_cached[i].u, -adjoint_p[i], rhs_term);
			if (solve_data.damping_form)
				solve_data.damping_form->force_shape_derivative(n_geom_bases, diff_cached[i].u, diff_cached[i - 1].u, -adjoint_p[i], damping_term);
			else
				damping_term.setZero(mass_term.size());
			if (is_contact_enabled())
			{
				solve_data.contact_form->force_shape_derivative(diff_cached[i].contact_set, diff_cached[i].u, -adjoint_p[i], contact_term);
				contact_term = down_sampling_mat * contact_term;
			}
			else
				contact_term.setZero(mass_term.size());
			if (solve_data.friction_form)
			{
				solve_data.friction_form->force_shape_derivative(diff_cached[i - 1].u, diff_cached[i].u, -adjoint_p[i], diff_cached[i].friction_constraint_set, friction_term);
				friction_term = down_sampling_mat * friction_term;
			}
			else
				friction_term.setZero(mass_term.size());

			contact_term /= beta_dt * beta_dt;
			friction_term /= beta_dt * beta_dt;

			one_form += weights[i] * functional_term + beta_dt * (elasticity_term + rhs_term + damping_term + contact_term + friction_term + mass_term);

			// logger().info("functional: {}, elasticity: {}, damping: {}, contact: {}, friction: {}, mass: {}", weights[i] * functional_term.norm(), beta_dt * elasticity_term.norm(), beta_dt * damping_term.norm(), beta_dt * contact_term.norm(), beta_dt * friction_term.norm(), beta_dt * mass_term.norm());
		}

		double beta;
		Eigen::MatrixXd sum_alpha_p, sum_alpha_nu;
		get_bdf_parts(bdf_order, 0, adjoint_p, adjoint_nu, sum_alpha_p, sum_alpha_nu, beta);
		solve_data.inertia_form->force_shape_derivative(mesh->is_volume(), n_geom_bases, bases, geom_bases(), assembler, mass_ass_vals_cache, initial_velocity_cache, sum_alpha_p, mass_term);

		Eigen::VectorXd integrals(js.size());
		for (int k = 0; k < js.size(); k++)
			integrals(k) = J_transient_step(js[k], 0);
		json param;
		param["t"] = 0.;
		param["step"] = (int)0;
		Eigen::VectorXd outer_grad = dJi_dintegrals(integrals, param);

		functional_term.setZero();
		for (int k = 0; k < js.size(); k++)
		{
			Eigen::VectorXd functional_term_k;
			compute_shape_derivative_functional_term(diff_cached[0].u, js[k], functional_term_k, 0);

			functional_term += functional_term_k * outer_grad(k);
		}

		one_form += weights[0] * functional_term + mass_term;
	}

	Eigen::VectorXd State::integral_gradient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, const std::string &type)
	{
		assert(problem->is_time_dependent() && diff_cached.size() > 0);

		Eigen::VectorXd grad;
		if (type == "material")
			dJ_material_transient(js, dJi_dintegrals, grad);
		else if (type == "shape")
			dJ_shape_transient(js, dJi_dintegrals, grad);
		else if (type == "initial-velocity")
		{
			Eigen::VectorXd tmp;
			dJ_initial_condition(js, dJi_dintegrals, tmp);
			grad = tmp.tail(tmp.size() / 2);
		}
		else if (type == "initial-position")
		{
			Eigen::VectorXd tmp;
			dJ_initial_condition(js, dJi_dintegrals, tmp);
			grad = tmp.head(tmp.size() / 2);
		}
		else if (type == "initial")
			dJ_initial_condition(js, dJi_dintegrals, grad);
		else if (type == "friction-coefficient")
		{
			grad.resize(1);
			dJ_friction_transient(js, dJi_dintegrals, grad(0));
		}
		else if (type == "damping-parameter")
			dJ_damping_transient(js, dJi_dintegrals, grad);
		else
			log_and_throw_error("Unknown derivative type!");

		return grad;
	}
} // namespace polyfem
