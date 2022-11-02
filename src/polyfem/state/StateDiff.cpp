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
} // namespace polyfem
