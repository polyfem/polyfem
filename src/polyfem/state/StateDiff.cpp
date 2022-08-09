#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polysolve/FEMSolver.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/LBFGSSolver.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/utils/local_to_global.hpp>

#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <deque>
#include <map>
#include <algorithm>

#include <fstream>

namespace polyfem
{
	namespace
	{
		typedef std::vector<std::function<void(int, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::VectorXd &)>> VolumeIntegrandTerms;
		typedef std::vector<std::function<void(int, const Eigen::VectorXi &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &, Eigen::VectorXd &)>> SurfaceIntegrandTerms;

		class LocalThreadMatStorage
		{
		public:
			utils::SpareMatrixCache cache;
			assembler::ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadMatStorage()
			{
			}

			LocalThreadMatStorage(const int buffer_size, const int rows, const int cols)
			{
				init(buffer_size, rows, cols);
			}

			LocalThreadMatStorage(const int buffer_size, const utils::SpareMatrixCache &c)
			{
				init(buffer_size, c);
			}

			void init(const int buffer_size, const int rows, const int cols)
			{
				// assert(rows == cols);
				cache.reserve(buffer_size);
				cache.init(rows, cols);
			}

			void init(const int buffer_size, const utils::SpareMatrixCache &c)
			{
				cache.reserve(buffer_size);
				cache.init(c);
			}
		};

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

		void volume_integral(
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const VolumeIntegrandTerms &integrand_functions,
			const mesh::Mesh &mesh,
			double &integral)
		{
			integral = 0;

			const int n_elements = int(bases.size());
			assembler::ElementAssemblyValues vals;
			for (int e = 0; e < n_elements; ++e)
			{
				vals.compute(e, mesh.is_volume(), bases[e], gbases[e]);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

				std::vector<Eigen::VectorXd> integrands;
				for (const auto &integrand_function : integrand_functions)
				{
					Eigen::VectorXd vec_term;
					integrand_function(e, quadrature.points, vals.val, vec_term);
					integrands.push_back(vec_term);
				}

				for (int q = 0; q < da.size(); ++q)
				{
					double integrand_product = 1;
					for (const auto &integrand : integrands)
					{
						integrand_product *= integrand(q);
					}
					integral += integrand_product * da(q);
				}
			}
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

			Eigen::MatrixXd uv, samples, gtmp;
			Eigen::VectorXi global_primitive_ids;
			Eigen::MatrixXd points, normals;
			Eigen::VectorXd weights;

			assembler::ElementAssemblyValues vals;

			for (const auto &lb : local_boundary)
			{
				const int e = lb.element_id();
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
					integral += integrand_product * da(q);
				}
			}
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

	void State::get_vf(Eigen::MatrixXd &vertices, Eigen::MatrixXi &faces, const bool geometric)
	{
		const auto &cur_bases = (iso_parametric() || !geometric) ? bases : geom_bases;
		const int n_elements = int(cur_bases.size());
		vertices = Eigen::MatrixXd::Zero((iso_parametric() || !geometric) ? n_bases : n_geom_bases, 3);
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
		auto &gbases = iso_parametric() ? bases : geom_bases;
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
				mesh->set_point(v, vertices.block(primitive_to_node[v], 0, 1, mesh->dimension()));
		}

		// update assembly cache
		ass_vals_cache.clear();
		if (n_bases <= args["cache_size"])
		{
			ass_vals_cache.init(mesh->is_volume(), bases, gbases);
			if (assembler.is_mixed(formulation()))
				pressure_ass_vals_cache.init(mesh->is_volume(), pressure_bases, gbases);
		}

		build_collision_mesh(collision_mesh, boundary_nodes_pos, boundary_edges, boundary_triangles, n_bases, bases);
		extract_vis_boundary_mesh();
	}

	/**
	 * @brief Computes a volume integral of a given functional over the current mesh and returns it.
	 *
	 * @param j Functional to integrate over the volume of the mesh. Takes in the global points, solution, grad of solution and parameters.
	 */
	double State::J_static(const IntegrableFunctional &j)
	{
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		double result = 0;
		if (j.is_volume_integral())
		{
			VolumeIntegrandTerms integrand_functions = {
				[&](int e, const Eigen::MatrixXd &reference_points, const Eigen::MatrixXd &global_points, Eigen::VectorXd &vec_term) {
					Eigen::MatrixXd u, grad_u;
					interpolate_at_local_vals(e, reference_points, sol, u, grad_u);
					Eigen::MatrixXd vec_term_mat;
					json params = {};
					params["elem"] = e;
					params["body_id"] = mesh->get_body_id(e);
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
					interpolate_at_local_vals(e, reference_points, sol, u, grad_u);
					std::vector<int> boundary_ids = {};
					for (int i = 0; i < global_primitive_ids.size(); ++i)
						boundary_ids.push_back(mesh->get_boundary_id(global_primitive_ids(i)));
					json params = {};
					params["elem"] = e;
					params["body_id"] = mesh->get_body_id(e);
					params["boundary_ids"] = boundary_ids;
					Eigen::MatrixXd vec_term_mat;
					j.evaluate(assembler.lame_params(), reference_points, global_points, u, grad_u, params, vec_term_mat);
					assert(vec_term.cols() == 1);
					vec_term = vec_term_mat;
				}};

			surface_integral(total_local_boundary, args["space"]["advanced"]["quadrature_order"], bases, gbases, integrand_functions, *mesh, result);
		}
		return result;
	}

	double State::J_transient_step(const IntegrableFunctional &j, const int step)
	{
		const auto &gbases = iso_parametric() ? bases : geom_bases;
		const double dt = args["time"]["dt"];

		double result = 0;
		if (j.is_volume_integral())
		{
			VolumeIntegrandTerms integrand_functions = {
				[&](int e, const Eigen::MatrixXd &reference_points, const Eigen::MatrixXd &global_points, Eigen::VectorXd &vec_term) {
					Eigen::MatrixXd u, grad_u;
					interpolate_at_local_vals(e, reference_points, diff_cached[step].u, u, grad_u);
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
					interpolate_at_local_vals(e, reference_points, diff_cached[step].u, u, grad_u);
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

	double State::J_transient(const IntegrableFunctional &j)
	{
		const double dt = args["time"]["dt"];
		const int n_steps = args["time"]["time_steps"];
		double result = 0;

		std::vector<double> weights;
		j.get_transient_quadrature_weights(n_steps, dt, weights);
		for (int i = 0; i <= n_steps; ++i)
		{
			if (weights[i] == 0)
				continue;
			result += weights[i] * J_transient_step(j, i);
		}

		return result;
	}

	void State::cache_transient_adjoint_quantities(const int current_step)
	{
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
		{
			logger().warn("Integrator type not supported for differentiability.");
			return;
		}

		StiffnessMatrix gradu_h;
		StiffnessMatrix gradu_h_prev;
		if (args["differentiable"])
			compute_force_hessian(gradu_h, gradu_h_prev, std::min(current_step, bdf_order));

		StiffnessMatrix gradu_h_next(gradu_h.rows(), gradu_h.cols());

		if (args["contact"]["enabled"])
		{
			if (diff_cached.size() > 0)
				diff_cached.back().gradu_h_next = gradu_h_prev;
			diff_cached.push_back({gradu_h, gradu_h_next, sol, 1, step_data.nl_problem->get_constraint_set(), step_data.nl_problem->get_friction_constraint_set()});
		}
		else
			diff_cached.push_back({gradu_h, gradu_h_next, sol, 1, ipc::Constraints(), ipc::FrictionConstraints()});
	}

	void State::compute_force_hessian(StiffnessMatrix &hessian, StiffnessMatrix &hessian_prev, const int bdf_order)
	{
		if (assembler.is_linear(formulation()) && !args["contact"]["enabled"])
		{
			hessian = stiffness;
			hessian_prev = StiffnessMatrix(stiffness.rows(), stiffness.cols());
		}
		else
		{
			if (step_data.nl_problem)
			{
				step_data.nl_problem->solution_changed(sol);
				compute_force_hessian_nonlinear(step_data.nl_problem, hessian, hessian_prev, bdf_order);
			}
		}
	}

	void State::compute_force_hessian_nonlinear(std::shared_ptr<solver::NLProblem> nl, StiffnessMatrix &hessian, StiffnessMatrix &hessian_prev, const int bdf_order)
	{
		int full_size = nl->get_full_size();
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		Eigen::VectorXd full;
		{
			if (sol.size() == nl->get_reduced_size())
				nl->reduced_to_full(sol, full);
			else
				full = sol;
		}
		assert(full.size() == full_size);

		hessian_prev = StiffnessMatrix(full_size, full_size);

		StiffnessMatrix energy_hessian(full_size, full_size);
		{
			if (assembler.is_linear(nl->rhs_assembler.formulation()))
			{
				nl->compute_cached_stiffness();
				energy_hessian = nl->cached_stiffness;
			}
			else
			{
				assembler.assemble_energy_hessian(nl->rhs_assembler.formulation(), mesh->is_volume(), n_bases, false, bases, gbases, ass_vals_cache, full, nl->mat_cache, energy_hessian);
			}

			// if (problem->is_time_dependent() && (args["materials"]["phi"].get<double>() > 0 || args["materials"]["psi"].get<double>() > 0) && diff_cached.size() > 0)
			// {
			// 	StiffnessMatrix damping_hessian(full_size, full_size);
			// 	damping_assembler.assemble_hessian(mesh->is_volume(), n_bases, args["time"]["dt"].get<double>(), false, bases, gbases, ass_vals_cache, full, diff_cached.back().u, nl->mat_cache, damping_hessian);

			// 	energy_hessian += damping_hessian;
			// }
		}

		StiffnessMatrix barrier_hessian(full_size, full_size), friction_hessian(full_size, full_size);
		if (args["contact"]["enabled"])
		{
			Eigen::MatrixXd displaced = boundary_nodes_pos + utils::unflatten(full, mesh->dimension());
			Eigen::MatrixXd displaced_surface = collision_mesh.vertices(displaced);

			const double mu = nl->mu();
			const double kappa = nl->barrier_stiffness();
			const double dhat = nl->dhat();
			const double epsv = nl->epsv_dt();

			barrier_hessian = ipc::compute_barrier_potential_hessian(
				collision_mesh, displaced_surface, nl->get_constraint_set(), dhat, false);
			barrier_hessian = collision_mesh.to_full_dof(barrier_hessian);

			Eigen::MatrixXd displaced_prev;
			if (diff_cached.size())
				displaced_prev = boundary_nodes_pos + utils::unflatten(diff_cached.back().u, mesh->dimension());
			else
				displaced_prev = displaced;

			Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(displaced_prev - boundary_nodes_pos);
			Eigen::MatrixXd surface_solution = collision_mesh.vertices(displaced - boundary_nodes_pos);

			// ipc::FrictionConstraints friction_constraint_set;
			// ipc::construct_friction_constraint_set(collision_mesh, collision_mesh.vertices(displaced_prev), _constraint_set, dhat, kappa, mu, friction_constraint_set);

			friction_hessian = -ipc::compute_friction_force_jacobian(
				collision_mesh,
				collision_mesh.vertices_at_rest(),
				surface_solution_prev,
				surface_solution,
				nl->get_friction_constraint_set(),
				dhat, kappa, epsv,
				ipc::FrictionConstraint::DiffWRT::U);
			friction_hessian = collision_mesh.to_full_dof(friction_hessian);

			hessian_prev = -ipc::compute_friction_force_jacobian(
				collision_mesh,
				collision_mesh.vertices_at_rest(),
				surface_solution_prev,
				surface_solution,
				nl->get_friction_constraint_set(),
				dhat, kappa, epsv,
				ipc::FrictionConstraint::DiffWRT::Ut);
			hessian_prev = collision_mesh.to_full_dof(hessian_prev);

			if (nl->get_is_time_dependent())
			{
				double beta = bdf_order < 1 ? std::nan("") : time_integrator::BDF::betas(bdf_order - 1);
				double dt = args["time"]["dt"];
				double acceleration_scaling = beta * beta * dt * dt;
				barrier_hessian /= acceleration_scaling;
				friction_hessian /= acceleration_scaling;
				hessian_prev /= acceleration_scaling;
			}
		}

		// if (problem->is_time_dependent() && (args["materials"]["phi"].get<double>() > 0 || args["materials"]["psi"].get<double>() > 0) && diff_cached.size() > 0)
		// {
		// 	StiffnessMatrix damping_hessian_prev(full_size, full_size);
		// 	damping_assembler.assemble_stress_prev_grad(mesh->is_volume(), n_bases, args["time"]["dt"].get<double>(), false, bases, gbases, ass_vals_cache, full, diff_cached.back().u, nl->mat_cache, damping_hessian_prev);

		// 	hessian_prev += damping_hessian_prev;
		// }

		hessian = energy_hessian + nl->barrier_stiffness() * barrier_hessian + friction_hessian;
	}

	double State::J_static(const SummableFunctional &j)
	{
		std::vector<bool> traversed(n_bases, false);
		const auto &gbases = iso_parametric() ? bases : geom_bases;
		json param;
		double energy = 0;
		Eigen::MatrixXd solution = utils::unflatten(sol, mesh->dimension());
		for (int e = 0; e < bases.size(); e++)
		{
			const auto &bs = bases[e];
			for (int i = 0; i < bs.bases.size(); i++)
			{
				const auto &b = bs.bases[i];
				assert(b.global().size() == 1);
				const auto &g = b.global()[0];
				if (traversed[g.index])
					continue;

				param["node"] = g.index;
				Eigen::MatrixXd val;
				j.evaluate(g.node, solution.row(g.index), param, val);
				energy += val(0);
				traversed[g.index] = true;
			}
		}
		return energy;
	}

	void State::compute_adjoint_rhs(const SummableFunctional &j, const Eigen::MatrixXd &solution, Eigen::VectorXd &b)
	{
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();
		b = Eigen::MatrixXd::Zero(n_bases * actual_dim, 1);

		if (!j.depend_on_u())
			return;

		const auto &gbases = iso_parametric() ? bases : geom_bases;
		std::vector<bool> traversed(n_bases, false);

		json param;
		Eigen::MatrixXd solution_ = utils::unflatten(solution, actual_dim);
		for (int e = 0; e < bases.size(); e++)
		{
			const auto &bs = bases[e];
			for (int i = 0; i < bs.bases.size(); i++)
			{
				assert(bs.bases[i].global().size() == 1);
				const auto &g = bs.bases[i].global()[0];
				if (traversed[g.index])
					continue;

				param["node"] = g.index;
				Eigen::MatrixXd val;
				j.dj_du(g.node, solution_.row(g.index), param, val);
				b.block(g.index * actual_dim, 0, actual_dim, 1) += val.transpose();
				traversed[g.index] = true;
			}
		}
	}

	void State::solve_adjoint(const SummableFunctional &j, Eigen::MatrixXd &adjoint_solution)
	{
		StiffnessMatrix A;
		Eigen::VectorXd b;

		if (!j.depend_on_u())
		{
			const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();
			adjoint_solution.setZero(actual_dim * n_bases, 1);
			return;
		}

		StiffnessMatrix unused;
		compute_force_hessian(A, unused);
		compute_adjoint_rhs(j, sol, b);

		solve_zero_dirichlet(A, b, boundary_nodes, adjoint_solution);
	}

	void State::compute_adjoint_rhs(const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const json &params)> &grad_j, const Eigen::MatrixXd &solution, Eigen::VectorXd &b, bool only_surface)
	{
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();
		b = Eigen::MatrixXd::Zero(n_bases * actual_dim, 1);

		const auto &gbases = iso_parametric() ? bases : geom_bases;

		const int n_elements = int(bases.size());
		assembler::ElementAssemblyValues vals;
		if (!only_surface)
		{
			for (int e = 0; e < n_elements; ++e)
			{
				vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

				const int n_loc_bases_ = int(vals.basis_values.size());
				Eigen::MatrixXd u, grad_u;
				interpolate_at_local_vals(e, quadrature.points, solution, u, grad_u);

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
							for (int q = 0; q < da.size(); ++q)
							{
								Eigen::Matrix<double, -1, -1, RowMajor> grad_phi;
								grad_phi.setZero(actual_dim, mesh->dimension());
								grad_phi.row(d) = v.grad_t_m.row(q);
								for (int k = 0; k < result_.cols(); k++)
									val += result_(q, k) * grad_phi(k) * da(q);
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

				vals.compute(e, mesh->is_volume(), points, bases[e], gbases[e]);

				const Eigen::VectorXd da = weights.array();

				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					normals.row(n) = normals.row(n) * vals.jac_it[n];
					normals.row(n).normalize();
				}

				const int n_loc_bases_ = int(vals.basis_values.size());
				Eigen::MatrixXd u, grad_u;
				interpolate_at_local_vals(e, points, solution, u, grad_u);

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
									Eigen::Matrix<double, -1, -1, RowMajor> grad_phi;
									grad_phi.setZero(actual_dim, mesh->dimension());
									grad_phi.row(d) = v.grad_t_m.row(q);
									for (int k = 0; k < result_.cols(); k++)
										val += result_(q, k) * grad_phi(k) * da(q);
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

	void State::setup_adjoint(const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const json &params)> &grad_j, StiffnessMatrix &A, Eigen::VectorXd &b, bool only_surface)
	{
		StiffnessMatrix unused;
		compute_force_hessian(A, unused);
		if (!problem->is_time_dependent())
			A += unused;

		compute_adjoint_rhs(grad_j, sol, b, only_surface);
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

		auto grad_j_func = [&](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const json &params) { return j.grad_j(assembler.lame_params(), local_pts, pts, u, grad_u, params); };
		setup_adjoint(grad_j_func, A, b, j.is_surface_integral());

		solve_zero_dirichlet(A, b, boundary_nodes, adjoint_solution);
	}

	void State::solve_zero_dirichlet(StiffnessMatrix &A, Eigen::VectorXd &b, const std::vector<int> &indices, Eigen::MatrixXd &adjoint_solution)
	{
		if (!args["solver"].contains("adjoint_linear"))
			args["solver"]["adjoint_linear"] = args["solver"]["linear"];

		auto solver = polysolve::LinearSolver::create(args["solver"]["adjoint_linear"]["solver"], args["solver"]["adjoint_linear"]["precond"]);
		solver->setParameters(args["solver"]["adjoint_linear"]);
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = A.rows();

		for (int i : indices)
			b(i) = 0;

		Eigen::Vector4d adjoint_spectrum;
		Eigen::VectorXd x;
		adjoint_spectrum = dirichlet_solve(*solver, A, b, indices, x, precond_num, args["output"]["data"]["stiffness_mat"], args["output"]["advanced"]["spectrum"], false, false);
		adjoint_solution = x;
	}

	void State::compute_shape_derivative_functional_term(const Eigen::MatrixXd &solution, const IntegrableFunctional &j, Eigen::VectorXd &term, const int cur_time_step)
	{
		const auto &gbases = iso_parametric() ? bases : geom_bases;
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();

		const int n_elements = int(bases.size());
		term.setZero(n_geom_bases * mesh->dimension(), 1);

		if (j.is_volume_integral())
		{
			for (int e = 0; e < n_elements; ++e)
			{
				assembler::ElementAssemblyValues vals;
				vals.compute(e, mesh->is_volume(), gbases[e], gbases[e]);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

				Eigen::MatrixXd u, grad_u;
				interpolate_at_local_vals(e, quadrature.points, solution, u, grad_u);

				Eigen::MatrixXd j_value;
				json params = {};
				params["elem"] = e;
				params["body_id"] = mesh->get_body_id(e);
				params["step"] = cur_time_step;
				j.evaluate(assembler.lame_params(), quadrature.points, vals.val, u, grad_u, params, j_value);

				Eigen::MatrixXd grad_j_value;
				if (j.depend_on_gradu() || j.depend_on_u())
					grad_j_value = j.grad_j(assembler.lame_params(), quadrature.points, vals.val, u, grad_u, params);

				Eigen::MatrixXd dj_dx;
				if (j.depend_on_x())
					j.dj_dx(assembler.lame_params(), quadrature.points, vals.val, u, grad_u, params, dj_dx);

				for (int q = 0; q < da.size(); ++q)
				{

					for (auto &v : vals.basis_values)
					{
						for (int d = 0; d < mesh->dimension(); d++)
						{
							Eigen::MatrixXd grad_v_q;
							grad_v_q.setZero(mesh->dimension(), mesh->dimension());
							grad_v_q.row(d) = v.grad_t_m.row(q);

							double velocity_div = grad_v_q.trace();

							term(v.global[0].index * mesh->dimension() + d) += j_value(q) * velocity_div * da(q);
							if (j.depend_on_x())
							{
								term(v.global[0].index * mesh->dimension() + d) += (v.val(q) * dj_dx(q, d)) * da(q);
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
								term(v.global[0].index * mesh->dimension() + d) += -dot(tau_q, grad_u_q * grad_v_q) * da(q);
							}
						}
					}
				}
			}
		}
		else
		{
			assert(!iso_parametric() || disc_orders[0] == 1);
			Eigen::VectorXi global_primitive_ids;
			Eigen::MatrixXd uv, points, normals;
			Eigen::VectorXd weights;

			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			get_vf(V, F);

			for (const auto &lb : total_local_boundary)
			{
				const int e = lb.element_id();
				bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, args["space"]["advanced"]["quadrature_order"], *mesh, false, uv, points, normals, weights, global_primitive_ids);

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
				interpolate_at_local_vals(e, points, solution, u, grad_u);

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
				if (j.depend_on_u() || j.depend_on_gradu())
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
						for (int q = n_samples_per_surface * i; q < n_samples_per_surface * (i + 1); ++q)
						{
							Eigen::MatrixXd grad_u_i, grad_p_i;
							if (mesh->dimension() == actual_dim)
								vector2matrix(grad_u.row(q), grad_u_i);
							else
								grad_u_i = grad_u.row(q);

							for (int d = 0; d < mesh->dimension(); d++)
							{
								Eigen::MatrixXd grad_v_i;
								grad_v_i.setZero(mesh->dimension(), mesh->dimension());
								grad_v_i.row(d) = v.grad_t_m.row(q);

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

								term(v.global[0].index * mesh->dimension() + d) += j_value(q) * velocity_div * da(q);
								if (j.depend_on_x())
								{
									term(v.global[0].index * mesh->dimension() + d) += (v.val(q) * dj_dx(q, d)) * da(q);
								}
								if (j.depend_on_gradu())
								{
									Eigen::MatrixXd tau_i;
									if (actual_dim == mesh->dimension())
										vector2matrix(grad_j_value.row(q), tau_i);
									else
										tau_i = grad_j_value.row(q);
									term(v.global[0].index * mesh->dimension() + d) += -dot(tau_i, grad_u_i * grad_v_i) * da(q);
								}
							}
						}
					}
				}
			}
		}
	}

	// assumes constant rhs over time
	void State::compute_shape_derivative_elasticity_term(const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term)
	{
		const auto &gbases = iso_parametric() ? bases : geom_bases;
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();

		const int n_elements = int(bases.size());
		term.setZero(n_geom_bases * mesh->dimension(), 1);

		auto f_prime_gradu_gradv_function = assembler.get_stress_grad_multiply_mat_function(formulation());

		for (int e = 0; e < n_elements; ++e)
		{
			assembler::ElementAssemblyValues vals;
			vals.compute(e, mesh->is_volume(), gbases[e], gbases[e]);

			const quadrature::Quadrature &quadrature = vals.quadrature;
			const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

			Eigen::MatrixXd u, grad_u, p, grad_p;
			interpolate_at_local_vals(e, quadrature.points, solution, u, grad_u);
			interpolate_at_local_vals(e, quadrature.points, adjoint_sol, p, grad_p);

			Eigen::MatrixXd rhs_function;
			problem->rhs(assembler, formulation(), vals.val, 0, rhs_function);
			rhs_function *= -1;
			for (int q = 0; q < vals.val.rows(); q++)
			{
				const double rho = density(quadrature.points.row(q), vals.val.row(q), e);
				rhs_function.row(q) *= rho;
			}

			for (int q = 0; q < da.size(); ++q)
			{
				Eigen::MatrixXd grad_u_i, grad_p_i;
				if (mesh->dimension() == actual_dim)
				{
					vector2matrix(grad_u.row(q), grad_u_i);
					vector2matrix(grad_p.row(q), grad_p_i);
				}
				else
				{
					grad_u_i = grad_u.row(q);
					grad_p_i = grad_p.row(q);
				}

				for (auto &v : vals.basis_values)
				{
					for (int d = 0; d < mesh->dimension(); d++)
					{
						Eigen::MatrixXd grad_v_i;
						grad_v_i.setZero(mesh->dimension(), mesh->dimension());
						grad_v_i.row(d) = v.grad_t_m.row(q);

						double velocity_div = grad_v_i.trace();

						Eigen::MatrixXd stress_tensor, f_prime_gradu_gradv;
						f_prime_gradu_gradv_function(e, quadrature.points.row(q), vals.val.row(q), grad_u_i, grad_u_i * grad_v_i, stress_tensor, f_prime_gradu_gradv);

						Eigen::MatrixXd tmp = grad_v_i - grad_v_i.trace() * Eigen::MatrixXd::Identity(mesh->dimension(), mesh->dimension());
						term(v.global[0].index * mesh->dimension() + d) += (dot(f_prime_gradu_gradv + stress_tensor * tmp.transpose(), grad_p_i) + dot(p.row(q), rhs_function.row(q)) * velocity_div) * da(q);
					}
				}
			}
		}
	}

	void State::compute_shape_derivative_damping_term(const Eigen::MatrixXd &solution, const Eigen::MatrixXd &prev_solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term)
	{
		const auto &gbases = iso_parametric() ? bases : geom_bases;
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();

		const int n_elements = int(bases.size());
		term.setZero(n_geom_bases * mesh->dimension(), 1);

		if (problem->is_scalar())
			return;

		const auto params = args["materials"];
		// if (params["phi"].get<double>() == 0 && params["psi"].get<double>() == 0)
			return;

		const double dt = args["time"]["dt"];

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			for (int e = start; e < end; ++e)
			{
				assembler::ElementAssemblyValues &vals = local_storage.vals;
				vals.compute(e, mesh->is_volume(), gbases[e], gbases[e]);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				local_storage.da = vals.det.array() * quadrature.weights.array();

				Eigen::MatrixXd u, grad_u, prev_u, prev_grad_u, p, grad_p;
				interpolate_at_local_vals(e, quadrature.points, solution, u, grad_u);
				interpolate_at_local_vals(e, quadrature.points, prev_solution, prev_u, prev_grad_u);
				interpolate_at_local_vals(e, quadrature.points, adjoint_sol, p, grad_p);

				Eigen::MatrixXd grad_u_i, grad_p_i, prev_grad_u_i;
				Eigen::MatrixXd grad_v_i;
				Eigen::MatrixXd stress_tensor, f_prime_gradu_gradv;
				Eigen::MatrixXd f_prev_prime_prev_gradu_gradv;

				for (int q = 0; q < local_storage.da.size(); ++q)
				{
					vector2matrix(grad_u.row(q), grad_u_i);
					vector2matrix(grad_p.row(q), grad_p_i);
					vector2matrix(prev_grad_u.row(q), prev_grad_u_i);

					for (auto &v : vals.basis_values)
					{
						Eigen::MatrixXd stress_grad, stress_prev_grad;
						damping_assembler.local_assembler().compute_stress_grad(e, dt, quadrature.points.row(q), vals.val.row(q), grad_u_i, prev_grad_u_i, stress_tensor, stress_grad);
						damping_assembler.local_assembler().compute_stress_prev_grad(e, dt, quadrature.points.row(q), vals.val.row(q), grad_u_i, prev_grad_u_i, stress_prev_grad);
						for (int d = 0; d < actual_dim; d++)
						{
							grad_v_i.setZero(actual_dim, actual_dim);
							grad_v_i.row(d) = v.grad_t_m.row(q);

							f_prime_gradu_gradv.setZero(actual_dim, actual_dim);
							Eigen::MatrixXd tmp = grad_u_i * grad_v_i;
							for (int i = 0; i < f_prime_gradu_gradv.rows(); i++)
								for (int j = 0; j < f_prime_gradu_gradv.cols(); j++)
									for (int k = 0; k < tmp.rows(); k++)
										for (int l = 0; l < tmp.cols(); l++)
											f_prime_gradu_gradv(i, j) += stress_grad(i * actual_dim + j, k * actual_dim + l) * tmp(k, l);

							f_prev_prime_prev_gradu_gradv.setZero(actual_dim, actual_dim);
							tmp = prev_grad_u_i * grad_v_i;
							for (int i = 0; i < f_prev_prime_prev_gradu_gradv.rows(); i++)
								for (int j = 0; j < f_prev_prime_prev_gradu_gradv.cols(); j++)
									for (int k = 0; k < tmp.rows(); k++)
										for (int l = 0; l < tmp.cols(); l++)
											f_prev_prime_prev_gradu_gradv(i, j) += stress_prev_grad(i * actual_dim + j, k * actual_dim + l) * tmp(k, l);

							tmp = grad_v_i - grad_v_i.trace() * Eigen::MatrixXd::Identity(actual_dim, actual_dim);
							// term(v.global[0].index * actual_dim + d) += dot(f_prime_gradu_gradv + f_prev_prime_prev_gradu_gradv + stress_tensor * tmp.transpose(), grad_p_i) * da(q);
							local_storage.vec(v.global[0].index * actual_dim + d) += dot(f_prime_gradu_gradv + f_prev_prime_prev_gradu_gradv + stress_tensor * tmp.transpose(), grad_p_i) * local_storage.da(q);
						}
					}
				}
			}
		});

		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;
	}

	void State::compute_material_derivative_elasticity_term(const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term)
	{
		const auto &gbases = iso_parametric() ? bases : geom_bases;
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();

		const int n_elements = int(bases.size());
		term.setZero(n_elements * 2, 1);

		auto df_dmu_dlambda_function = assembler.get_dstress_dmu_dlambda_function(formulation());

		for (int e = 0; e < n_elements; ++e)
		{
			assembler::ElementAssemblyValues vals;
			vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

			const quadrature::Quadrature &quadrature = vals.quadrature;
			const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

			Eigen::MatrixXd u, grad_u, p, grad_p;
			interpolate_at_local_vals(e, quadrature.points, solution, u, grad_u);
			interpolate_at_local_vals(e, quadrature.points, adjoint_sol, p, grad_p);

			for (int q = 0; q < da.size(); ++q)
			{
				Eigen::MatrixXd grad_p_i, grad_u_i;
				vector2matrix(grad_p.row(q), grad_p_i);
				vector2matrix(grad_u.row(q), grad_u_i);

				Eigen::MatrixXd f_prime_dmu, f_prime_dlambda;
				df_dmu_dlambda_function(e, quadrature.points.row(q), vals.val.row(q), grad_u_i, f_prime_dmu, f_prime_dlambda);

				// This needs to be a sum over material parameter basis.
				term(e + n_elements) += -dot(f_prime_dmu, grad_p_i) * da(q);
				term(e) += -dot(f_prime_dlambda, grad_p_i) * da(q);
			}
		}
	}

	void State::compute_damping_derivative_damping_term(const Eigen::MatrixXd &solution, const Eigen::MatrixXd &prev_solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term)
	{
		const auto &gbases = iso_parametric() ? bases : geom_bases;
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();

		const int n_elements = int(bases.size());
		term.setZero(2);

		for (int e = 0; e < n_elements; ++e)
		{
			assembler::ElementAssemblyValues vals;
			vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

			const quadrature::Quadrature &quadrature = vals.quadrature;
			const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

			Eigen::MatrixXd u, grad_u, prev_u, prev_grad_u, p, grad_p;
			interpolate_at_local_vals(e, quadrature.points, solution, u, grad_u);
			interpolate_at_local_vals(e, quadrature.points, prev_solution, prev_u, prev_grad_u);
			interpolate_at_local_vals(e, quadrature.points, adjoint_sol, p, grad_p);

			for (int q = 0; q < da.size(); ++q)
			{
				Eigen::MatrixXd grad_p_i, grad_u_i, prev_grad_u_i;
				vector2matrix(grad_p.row(q), grad_p_i);
				vector2matrix(grad_u.row(q), grad_u_i);
				vector2matrix(prev_grad_u.row(q), prev_grad_u_i);

				Eigen::MatrixXd f_prime_dpsi, f_prime_dphi;
				damping_assembler.local_assembler().compute_dstress_dpsi_dphi(e, args["time"]["dt"].get<double>(), quadrature.points.row(q), vals.val.row(q), grad_u_i, prev_grad_u_i, f_prime_dpsi, f_prime_dphi);

				// This needs to be a sum over material parameter basis.
				term(0) += -dot(f_prime_dpsi, grad_p_i) * da(q);
				term(1) += -dot(f_prime_dphi, grad_p_i) * da(q);
			}
		}
	}

	void State::compute_mass_derivative_term(const Eigen::MatrixXd &adjoint_sol, const Eigen::MatrixXd &velocity, Eigen::VectorXd &term)
	{
		const auto &gbases = iso_parametric() ? bases : geom_bases;
		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();

		const int n_elements = int(bases.size());
		term.setZero(n_geom_bases * mesh->dimension(), 1);

		for (int e = 0; e < n_elements; ++e)
		{
			assembler::ElementAssemblyValues vals;
			vals.compute(e, mesh->is_volume(), gbases[e], gbases[e]);

			const quadrature::Quadrature &quadrature = vals.quadrature;
			const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

			Eigen::MatrixXd vel, grad_vel, p, grad_p;
			interpolate_at_local_vals(e, quadrature.points, adjoint_sol, p, grad_p);
			interpolate_at_local_vals(e, quadrature.points, velocity, vel, grad_vel);

			for (int q = 0; q < da.size(); ++q)
			{
				const double rho = density(quadrature.points.row(q), vals.val.row(q), e);

				for (const auto &v : vals.basis_values)
				{
					for (int d = 0; d < mesh->dimension(); d++)
					{
						Eigen::MatrixXd grad_v_i;
						grad_v_i.setZero(mesh->dimension(), mesh->dimension());
						grad_v_i.row(d) = v.grad_t_m.row(q);

						double shape_velocity_div = grad_v_i.trace();

						term(v.global[0].index * mesh->dimension() + d) += rho * dot(p.row(q), vel.row(q)) * shape_velocity_div * da(q);
					}
				}
			}
		}
	}

	void State::compute_derivative_contact_term(const ipc::Constraints &contact_set, const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term)
	{
		term.setZero(n_geom_bases * mesh->dimension(), 1);
		if (!args["contact"]["enabled"])
			return;

		const double dhat = step_data.nl_problem->dhat();
		Eigen::MatrixXd U = collision_mesh.vertices(utils::unflatten(solution, mesh->dimension()));
		Eigen::MatrixXd X = collision_mesh.vertices(boundary_nodes_pos);

		StiffnessMatrix dq_h = collision_mesh.to_full_dof(ipc::compute_barrier_shape_derivative(collision_mesh, X + U, contact_set, dhat));
		term = -step_data.nl_problem->barrier_stiffness() * down_sampling_mat * (adjoint_sol.transpose() * dq_h).transpose();

		// const double eps = 1e-6;
		// Eigen::MatrixXd target = dq_h;
		// Eigen::MatrixXd hessian_fd;
		// hessian_fd.setZero(dq_h.rows(), dq_h.cols());

		// Eigen::VectorXd theta(n_geom_bases * mesh->dimension());
		// for (int i = 0; i < theta.size(); i++)
		// 	theta(i) = (rand() % 10000) / 10000.0;

		// Eigen::VectorXd deriv = dq_h * down_sampling_mat.transpose() * theta;
		// Eigen::VectorXd deriv_FD = Eigen::VectorXd::Zero(deriv.size());

		// for (int k = 0; k < 2; k++)
		// {
		// 	double sign = k ? 1 : -1;
		// 	perturb_mesh(theta * eps * sign);
		// 	auto X1 = collision_mesh.vertices(boundary_nodes_pos);

		// 	ipc::Constraints constraint_set1;
		// 	ipc::construct_constraint_set(collision_mesh, X1 + U, dhat, constraint_set1);

		// 	Eigen::VectorXd H = collision_mesh.to_full_dof(ipc::compute_barrier_potential_gradient(collision_mesh, X1 + U, constraint_set1, dhat));
		// 	deriv_FD += H * sign / (2 * eps);

		// 	perturb_mesh(theta * eps * -sign);
		// }

		// logger().error("FD error: {}, norm: {}", (deriv_FD - deriv).norm(), deriv.norm());
	}

	void State::compute_derivative_friction_term(const Eigen::MatrixXd &prev_solution, const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint_sol, const ipc::FrictionConstraints &friction_constraint_set, Eigen::VectorXd &term)
	{
		term.setZero(n_geom_bases * mesh->dimension(), 1);
		if (!args["contact"]["enabled"] || args["contact"]["friction_coefficient"].get<double>() == 0)
			return;

		Eigen::MatrixXd U = collision_mesh.vertices(utils::unflatten(solution, mesh->dimension()));
		Eigen::MatrixXd U_prev = collision_mesh.vertices(utils::unflatten(prev_solution, mesh->dimension()));
		Eigen::MatrixXd X = collision_mesh.vertices(boundary_nodes_pos);

		const double kappa = step_data.nl_problem->barrier_stiffness();
		const double dhat = step_data.nl_problem->dhat();
		const double epsv = step_data.nl_problem->epsv_dt();

		StiffnessMatrix hess = ipc::compute_friction_force_jacobian(
			collision_mesh,
			collision_mesh.vertices_at_rest(),
			U_prev, U,
			friction_constraint_set,
			dhat, kappa, epsv,
			ipc::FrictionConstraint::DiffWRT::X);

		term = down_sampling_mat * (adjoint_sol.transpose() * collision_mesh.to_full_dof(hess)).transpose();
	}

	/**
	 * @brief Computes the shape derivative in one form.
	 *
	 * For now, the basis of the mesh fe discretization is coupled with the velocity basis. This eventually needs to be
	 * decoupled as velocity must have first order basis.
	 *
	 * @param j The functional and its derivatives with respect to the input, either j(u, x) or j(grad u, x).
	 * @param one_form The one form of the derivative, such that dJ[v] = <one_form, v>.
	 */
	void State::dJ_shape_static(
		const IntegrableFunctional &j,
		Eigen::VectorXd &one_form)
	{
		assert(!problem->is_time_dependent());
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		Eigen::MatrixXd adjoint_sol;
		solve_adjoint(j, adjoint_sol);

		Eigen::VectorXd functional_term, elasticity_term, contact_term, friction_term;

		compute_shape_derivative_functional_term(sol, j, functional_term);
		one_form = functional_term;

		if (j.depend_on_u() || j.depend_on_gradu())
		{
			compute_shape_derivative_elasticity_term(sol, adjoint_sol, elasticity_term);
			compute_derivative_contact_term(step_data.nl_problem->get_constraint_set(), sol, adjoint_sol, contact_term);
			// compute_derivative_friction_term(sol, sol, adjoint_sol, step_data.nl_problem->get_friction_constraint_set(), friction_term);
			one_form += elasticity_term + contact_term; // + friction_term;
		}
	}

	void State::dJ_material_static(
		const SummableFunctional &j,
		Eigen::VectorXd &one_form)
	{
		assert(!problem->is_time_dependent());

		Eigen::MatrixXd adjoint_sol;
		solve_adjoint(j, adjoint_sol);

		one_form.setZero(bases.size() * 2, 1);
		if (j.depend_on_u())
		{
			compute_material_derivative_elasticity_term(sol, adjoint_sol, one_form);
		}
	}

	void State::dJ_friction_transient(const IntegrableFunctional &j, double &one_form)
	{
		assert(problem->is_time_dependent());
		assert(args["contact"]["enabled"]);

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(j, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];
		const int problem_dim = mesh->dimension();

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		one_form = 0;

		for (int t = time_steps; t > 0; --t)
		{
			const int real_order = std::min(bdf_order, t);
			double beta = time_integrator::BDF::betas(real_order - 1);

			ipc::FrictionConstraints friction_constraint_set = diff_cached[t].friction_constraint_set;
			const auto &solution = diff_cached[t].u;
			const auto &prev_solution = diff_cached[t - 1].u;

			Eigen::MatrixXd surface_solution_prev = collision_mesh.vertices(utils::unflatten(prev_solution, problem_dim));
			Eigen::MatrixXd surface_solution = collision_mesh.vertices(utils::unflatten(solution, problem_dim));

			auto force = -ipc::compute_friction_force(collision_mesh, collision_mesh.vertices_at_rest(), surface_solution_prev, surface_solution, friction_constraint_set, step_data.nl_problem->dhat(), step_data.nl_problem->barrier_stiffness(), step_data.nl_problem->epsv_dt(), 0, true);

			one_form += (adjoint_p[t].array() * collision_mesh.to_full_dof(force).array()).sum() / (beta * dt);
		}
	}

	void State::dJ_damping_transient(const IntegrableFunctional &j, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(args["contact"]["enabled"]);

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(j, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		Eigen::VectorXd damping_term;
		one_form.setZero(2);

		for (int i = time_steps; i > 0; --i)
		{
			const int real_order = std::min(bdf_order, i);
			double beta = time_integrator::BDF::betas(real_order - 1);

			compute_damping_derivative_damping_term(diff_cached[i].u, diff_cached[i - 1].u, -adjoint_p[i], damping_term);
			one_form += beta * dt * damping_term;
		}
	}

	void State::dJ_material_static(const IntegrableFunctional &j, Eigen::VectorXd &one_form)
	{
		assert(!problem->is_time_dependent());
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		Eigen::MatrixXd adjoint_sol;
		solve_adjoint(j, adjoint_sol);

		Eigen::VectorXd elasticity_term;
		compute_material_derivative_elasticity_term(sol, adjoint_sol, elasticity_term);

		one_form = elasticity_term;
		logger().debug("material derivative: elasticity: {}", elasticity_term.norm());
	}

	void State::sample_field(std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> field, Eigen::MatrixXd &discrete_field, const int order)
	{
		Eigen::MatrixXd tmp;
		tmp.setZero(1, mesh->dimension());
		tmp = field(tmp);
		const int actual_dim = tmp.cols();

		if (order >= 1)
		{
			const bool use_bases = order > 1;
			const int n_current_bases = (iso_parametric() || use_bases) ? n_bases : n_geom_bases;
			const auto &current_bases = (iso_parametric() || use_bases) ? bases : geom_bases;
			discrete_field.setZero(n_current_bases * actual_dim, 1);

			for (int e = 0; e < bases.size(); e++)
			{
				Eigen::MatrixXd local_pts, pts;
				if (!mesh->is_volume())
					autogen::p_nodes_2d(current_bases[e].bases.front().order(), local_pts);
				else
					autogen::p_nodes_3d(current_bases[e].bases.front().order(), local_pts);

				current_bases[e].eval_geom_mapping(local_pts, pts);
				Eigen::MatrixXd result = field(pts);
				for (int i = 0; i < local_pts.rows(); i++)
				{
					assert(current_bases[e].bases[i].global().size() == 1);
					for (int d = 0; d < actual_dim; d++)
						discrete_field(current_bases[e].bases[i].global()[0].index * actual_dim + d) = result(i, d);
				}
			}
		}
		else if (order == 0)
		{
			discrete_field.setZero(bases.size() * actual_dim, 1);
			Eigen::MatrixXd centers;
			if (mesh->is_volume())
				mesh->cell_barycenters(centers);
			else
				mesh->face_barycenters(centers);
			Eigen::MatrixXd result = field(centers);
			for (int e = 0; e < bases.size(); e++)
				for (int d = 0; d < actual_dim; d++)
					discrete_field(e * actual_dim + d) = result(e, d);
		}
	}

	void State::solve_transient_adjoint(const IntegrableFunctional &j, std::vector<Eigen::MatrixXd> &adjoint_nu, std::vector<Eigen::MatrixXd> &adjoint_p)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		adjoint_p.assign(time_steps + 2, Eigen::MatrixXd::Zero(sol.size(), 1));
		adjoint_nu.assign(time_steps + 2, Eigen::MatrixXd::Zero(sol.size(), 1));

		if (!j.depend_on_u() && !j.depend_on_gradu())
			return;

		// set dirichlet rows of mass to identity
		StiffnessMatrix reduced_mass;
		replace_rows_by_identity(reduced_mass, mass, boundary_nodes);

		std::vector<double> weights;
		j.get_transient_quadrature_weights(time_steps, dt, weights);
		Eigen::MatrixXd sum_alpha_p, sum_alpha_nu;
		for (int i = time_steps; i >= 0; --i)
		{
			double beta;
			get_bdf_parts(bdf_order, i, adjoint_p, adjoint_nu, sum_alpha_p, sum_alpha_nu, beta);
			double beta_dt = beta * dt;

			StiffnessMatrix gradu_h, gradu_h_next;
			if (i > 0)
				replace_rows_by_identity(gradu_h, -beta_dt * diff_cached[i].gradu_h, boundary_nodes);
			replace_rows_by_identity(gradu_h_next, -beta_dt * diff_cached[i].gradu_h_next, boundary_nodes);

			auto grad_j_func = [&](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const json &params) {
				json params_extended = params;
				params_extended["step"] = i;
				params_extended["t"] = i * args["time"]["dt"].get<double>();
				return j.grad_j(assembler.lame_params(), local_pts, pts, u, grad_u, params_extended);
			};
			Eigen::VectorXd gradu_j;
			compute_adjoint_rhs(grad_j_func, diff_cached[i].u, gradu_j, j.is_surface_integral());

			if (i > 0)
			{
				StiffnessMatrix A = (reduced_mass - beta_dt * gradu_h).transpose();
				Eigen::VectorXd rhs_ = -reduced_mass.transpose() * sum_alpha_nu - gradu_h.transpose() * sum_alpha_p + gradu_h_next.transpose() * adjoint_p[i + 1] - weights[i] * gradu_j;
				solve_zero_dirichlet(A, rhs_, boundary_nodes, adjoint_nu[i]);
				adjoint_p[i] = beta_dt * adjoint_nu[i] - sum_alpha_p;
			}
			else
			{
				adjoint_p[i] = -reduced_mass.transpose() * sum_alpha_p;
				adjoint_nu[i] = -weights[i] * gradu_j - reduced_mass.transpose() * sum_alpha_nu + beta_dt * diff_cached[i].gradu_h_next.transpose() * adjoint_p[i + 1]; // adjoint_nu[0] actually stores adjoint_mu[0]
			}
		}
	}

	void State::dJ_initial_condition(const IntegrableFunctional &j, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(j, adjoint_nu, adjoint_p);

		one_form.setZero(n_bases * mesh->dimension() * 2); // half for initial solution, half for initial velocity

		// \partial_q \hat{J}^0 - p_0^T \partial_q g^v - \mu_0^T \partial_q g^u
		one_form.block(0, 0, n_bases * mesh->dimension(), 1) = -adjoint_nu[0]; // adjoint_nu[0] actually stores adjoint_mu[0]
		one_form.block(n_bases * mesh->dimension(), 0, n_bases * mesh->dimension(), 1) = -adjoint_p[0];
	}

	void State::dJ_full_material_transient(const IntegrableFunctional &j, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());
		assert(args["contact"]["enabled"]);
		assert(args["contact"]["friction_coefficient"].get<double>() > 0);

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(j, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const double mu = args["contact"]["friction_coefficient"].get<double>();
		const int time_steps = args["time"]["time_steps"];
		const int problem_dim = mesh->dimension();

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		Eigen::VectorXd elasticity_term;
		one_form.setZero(bases.size() * 2 + 3);

		for (int t = time_steps; t > 0; --t)
		{
			const int real_order = std::min(bdf_order, t);
			const double beta = time_integrator::BDF::betas(real_order - 1);
			const double beta_dt = beta * dt;

			// lame paramters
			compute_material_derivative_elasticity_term(diff_cached[t].u, -adjoint_p[t], elasticity_term);
			one_form.head(2 * bases.size()) += beta_dt * elasticity_term;

			// friction coefficients
			ipc::FrictionConstraints friction_constraint_set = diff_cached[t].friction_constraint_set;
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

			auto force = -ipc::compute_friction_force(collision_mesh, collision_mesh.vertices_at_rest(), surface_solution_prev, surface_solution, friction_constraint_set, step_data.nl_problem->dhat(), step_data.nl_problem->barrier_stiffness(), step_data.nl_problem->epsv_dt());

			one_form(2 * bases.size()) += (adjoint_p[t].array() * collision_mesh.to_full_dof(force).array()).sum() / (beta_dt * mu);

			// damping coefficients
			Eigen::VectorXd damping_term;
			compute_damping_derivative_damping_term(diff_cached[t].u, diff_cached[t - 1].u, -adjoint_p[t], damping_term);
			one_form.tail(2) += beta_dt * damping_term;
		}
	}

	void State::dJ_material_transient(const IntegrableFunctional &j, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(j, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		Eigen::VectorXd elasticity_term;
		one_form.setZero(bases.size() * 2);

		for (int i = time_steps; i > 0; --i)
		{
			const int real_order = std::min(bdf_order, i);
			double beta = time_integrator::BDF::betas(real_order - 1);

			compute_material_derivative_elasticity_term(diff_cached[i].u, -adjoint_p[i], elasticity_term);
			one_form += beta * dt * elasticity_term;
		}
	}

	void State::dJ_shape_transient(const IntegrableFunctional &j, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(j, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		Eigen::VectorXd elasticity_term, damping_term, mass_term, contact_term, friction_term, functional_term;
		one_form.setZero(n_geom_bases * mesh->dimension());

		std::vector<double> weights;
		j.get_transient_quadrature_weights(time_steps, dt, weights);
		for (int i = time_steps; i > 0; --i)
		{
			const int real_order = std::min(bdf_order, i);
			double beta = time_integrator::BDF::betas(real_order - 1);
			double beta_dt = beta * dt;

			Eigen::MatrixXd velocity = diff_cached[i].u;
			for (int o = 1; o <= real_order; o++)
				velocity -= time_integrator::BDF::alphas(real_order - 1)[o - 1] * diff_cached[i - o].u;
			velocity /= beta_dt;

			if (weights[i] == 0)
				functional_term.setZero(one_form.rows(), one_form.cols());
			else
				compute_shape_derivative_functional_term(diff_cached[i].u, j, functional_term, i);

			if (j.depend_on_u() || j.depend_on_gradu())
			{
				compute_mass_derivative_term(adjoint_nu[i], velocity, mass_term);
				compute_shape_derivative_elasticity_term(diff_cached[i].u, -adjoint_p[i], elasticity_term);
				compute_shape_derivative_damping_term(diff_cached[i].u, diff_cached[i - 1].u, -adjoint_p[i], damping_term);
				compute_derivative_contact_term(diff_cached[i].contact_set, diff_cached[i].u, -adjoint_p[i], contact_term);
				compute_derivative_friction_term(diff_cached[i - 1].u, diff_cached[i].u, -adjoint_p[i], diff_cached[i].friction_constraint_set, friction_term);

				contact_term /= beta_dt * beta_dt;
				friction_term /= beta_dt * beta_dt;
			}
			else
			{
				mass_term.setZero(one_form.rows(), one_form.cols());
				elasticity_term.setZero(one_form.rows(), one_form.cols());
				damping_term.setZero(one_form.rows(), one_form.cols());
				contact_term.setZero(one_form.rows(), one_form.cols());
				friction_term.setZero(one_form.rows(), one_form.cols());
			}

			one_form += weights[i] * functional_term + beta_dt * (elasticity_term + damping_term + contact_term + friction_term + mass_term);

			// logger().info("functional: {}, elasticity: {}, damping: {}, contact: {}, friction: {}, mass: {}", weights[i] * functional_term.norm(), beta_dt * elasticity_term.norm(), beta_dt * damping_term.norm(), beta_dt * contact_term.norm(), beta_dt * friction_term.norm(), beta_dt * mass_term.norm());
		}

		if (j.depend_on_u() || j.depend_on_gradu())
		{
			double beta;
			Eigen::MatrixXd sum_alpha_p, sum_alpha_nu;
			get_bdf_parts(bdf_order, 0, adjoint_p, adjoint_nu, sum_alpha_p, sum_alpha_nu, beta);
			compute_mass_derivative_term(sum_alpha_p, initial_velocity_cache, mass_term);
		}
		else
		{
			mass_term.setZero(one_form.rows(), one_form.cols());
		}

		compute_shape_derivative_functional_term(diff_cached[0].u, j, functional_term, 0);
		one_form += weights[0] * functional_term + mass_term;
	}

	void State::solve_transient_adjoint(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, std::vector<Eigen::MatrixXd> &adjoint_nu, std::vector<Eigen::MatrixXd> &adjoint_p)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());
		assert(js.size() > 0);

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];
		const auto &gbases = iso_parametric() ? bases : geom_bases;

		adjoint_p.assign(time_steps + 2, Eigen::MatrixXd::Zero(sol.size(), 1));
		adjoint_nu.assign(time_steps + 2, Eigen::MatrixXd::Zero(sol.size(), 1));

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

			StiffnessMatrix gradu_h, gradu_h_next;
			replace_rows_by_identity(gradu_h, -beta_dt * diff_cached[i].gradu_h, boundary_nodes);
			replace_rows_by_identity(gradu_h_next, -beta_dt * diff_cached[i].gradu_h_next, boundary_nodes);

			Eigen::VectorXd dJ_du;
			dJ_du.setZero(gradu_h.rows());

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
					compute_adjoint_rhs([&](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const json &params) {
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
				StiffnessMatrix A = (reduced_mass - beta_dt * gradu_h).transpose();
				Eigen::VectorXd rhs_ = -reduced_mass.transpose() * sum_alpha_nu - gradu_h.transpose() * sum_alpha_p + gradu_h_next.transpose() * adjoint_p[i + 1] - dJ_du;
				solve_zero_dirichlet(A, rhs_, boundary_nodes, adjoint_nu[i]);
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

	void State::dJ_full_material_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());
		assert(args["contact"]["enabled"]);
		assert(args["contact"]["friction_coefficient"].get<double>() > 0);

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(js, dJi_dintegrals, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const double mu = args["contact"]["friction_coefficient"].get<double>();
		const int time_steps = args["time"]["time_steps"];
		const int problem_dim = mesh->dimension();

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		Eigen::VectorXd elasticity_term;
		one_form.setZero(bases.size() * 2 + 3);

		for (int t = time_steps; t > 0; --t)
		{
			const int real_order = std::min(bdf_order, t);
			const double beta = time_integrator::BDF::betas(real_order - 1);
			const double beta_dt = beta * dt;

			// lame paramters
			compute_material_derivative_elasticity_term(diff_cached[t].u, -adjoint_p[t], elasticity_term);
			one_form.head(2 * bases.size()) += beta_dt * elasticity_term;

			// friction coefficients
			ipc::FrictionConstraints friction_constraint_set = diff_cached[t].friction_constraint_set;
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

			auto force = -ipc::compute_friction_force(collision_mesh, collision_mesh.vertices_at_rest(), surface_solution_prev, surface_solution, friction_constraint_set, step_data.nl_problem->dhat(), step_data.nl_problem->barrier_stiffness(), step_data.nl_problem->epsv_dt());

			one_form(2 * bases.size()) += (adjoint_p[t].array() * collision_mesh.to_full_dof(force).array()).sum() / (beta_dt * mu);

			// damping coefficients
			Eigen::VectorXd damping_term;
			compute_damping_derivative_damping_term(diff_cached[t].u, diff_cached[t - 1].u, -adjoint_p[t], damping_term);
			one_form.tail(2) += beta_dt * damping_term;
		}
	}

	void State::dJ_material_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(!problem->is_scalar());

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(js, dJi_dintegrals, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		Eigen::VectorXd elasticity_term;
		one_form.setZero(bases.size() * 2);

		for (int i = time_steps; i > 0; --i)
		{
			const int real_order = std::min(bdf_order, i);
			double beta = time_integrator::BDF::betas(real_order - 1);

			compute_material_derivative_elasticity_term(diff_cached[i].u, -adjoint_p[i], elasticity_term);
			one_form += beta * dt * elasticity_term;
		}
	}
	void State::dJ_friction_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, double &one_form)
	{
		assert(problem->is_time_dependent());
		assert(args["contact"]["enabled"]);
		assert(args["contact"]["friction_coefficient"].get<double>() > 0);

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(js, dJi_dintegrals, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const double mu = args["contact"]["friction_coefficient"].get<double>();
		const int time_steps = args["time"]["time_steps"];
		const int problem_dim = mesh->dimension();

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		one_form = 0;

		for (int t = time_steps; t > 0; --t)
		{
			const int real_order = std::min(bdf_order, t);
			double beta = time_integrator::BDF::betas(real_order - 1);

			ipc::FrictionConstraints friction_constraint_set = diff_cached[t].friction_constraint_set;
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

			auto force = -ipc::compute_friction_force(collision_mesh, collision_mesh.vertices_at_rest(), surface_solution_prev, surface_solution, friction_constraint_set, step_data.nl_problem->dhat(), step_data.nl_problem->barrier_stiffness(), step_data.nl_problem->epsv_dt());

			one_form += (adjoint_p[t].array() * collision_mesh.to_full_dof(force).array()).sum() / (beta * mu * dt);
		}
	}
	void State::dJ_damping_transient(const std::vector<IntegrableFunctional> &js, const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const json &)> &dJi_dintegrals, Eigen::VectorXd &one_form)
	{
		assert(problem->is_time_dependent());
		assert(args["contact"]["enabled"]);

		std::vector<Eigen::MatrixXd> adjoint_nu, adjoint_p;
		solve_transient_adjoint(js, dJi_dintegrals, adjoint_nu, adjoint_p);

		const double dt = args["time"]["dt"];
		const int time_steps = args["time"]["time_steps"];

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		Eigen::VectorXd damping_term;
		one_form.setZero(2);

		for (int t = time_steps; t > 0; --t)
		{
			const int real_order = std::min(bdf_order, t);
			double beta = time_integrator::BDF::betas(real_order - 1);

			compute_damping_derivative_damping_term(diff_cached[t].u, diff_cached[t - 1].u, -adjoint_p[t], damping_term);
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

		int bdf_order = -1;
		if (args["time"]["integrator"] == "ImplicitEuler")
			bdf_order = 1;
		else if (args["time"]["integrator"] == "BDF")
			bdf_order = args["time"]["BDF"]["steps"].get<int>();
		else
			throw("Integrator type not supported for differentiability.");

		Eigen::VectorXd elasticity_term, damping_term, mass_term, contact_term, friction_term, functional_term;
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

			compute_mass_derivative_term(adjoint_nu[i], velocity, mass_term);
			compute_shape_derivative_elasticity_term(diff_cached[i].u, -adjoint_p[i], elasticity_term);
			compute_shape_derivative_damping_term(diff_cached[i].u, diff_cached[i - 1].u, -adjoint_p[i], damping_term);
			compute_derivative_contact_term(diff_cached[i].contact_set, diff_cached[i].u, -adjoint_p[i], contact_term);
			compute_derivative_friction_term(diff_cached[i - 1].u, diff_cached[i].u, -adjoint_p[i], friction_constraint_set, friction_term);

			contact_term /= beta_dt * beta_dt;
			friction_term /= beta_dt * beta_dt;

			one_form += weights[i] * functional_term + beta_dt * (elasticity_term + damping_term + contact_term + friction_term + mass_term);

			// logger().info("functional: {}, elasticity: {}, damping: {}, contact: {}, friction: {}, mass: {}", weights[i] * functional_term.norm(), beta_dt * elasticity_term.norm(), beta_dt * damping_term.norm(), beta_dt * contact_term.norm(), beta_dt * friction_term.norm(), beta_dt * mass_term.norm());
		}

		double beta;
		Eigen::MatrixXd sum_alpha_p, sum_alpha_nu;
		get_bdf_parts(bdf_order, 0, adjoint_p, adjoint_nu, sum_alpha_p, sum_alpha_nu, beta);
		compute_mass_derivative_term(sum_alpha_p, initial_velocity_cache, mass_term);

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
} // namespace polyfem
