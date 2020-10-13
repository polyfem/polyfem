#include <polyfem/NLProblem.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/Types.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>

#include <igl/write_triangle_mesh.h>

#include <unsupported/Eigen/SparseExtra>

static bool disable_collision = false;

namespace polyfem
{
	using namespace polysolve;

	NLProblem::NLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat, const bool project_to_psd)
		: state(state), assembler(state.assembler), rhs_assembler(rhs_assembler),
		  full_size((assembler.is_mixed(state.formulation()) ? state.n_pressure_bases : 0) + state.n_bases * state.mesh->dimension()),
		  reduced_size(full_size - state.boundary_nodes.size()),
		  t(t), rhs_computed(false), is_time_dependent(state.problem->is_time_dependent()), project_to_psd(project_to_psd)
	{
		assert(!assembler.is_mixed(state.formulation()));

		_dhat = dhat;
		_barrier_stiffness = 1;
	}

	void NLProblem::init(const TVector &full)
	{
		_barrier_stiffness = 1;
		if (disable_collision || !state.args["has_collision"])
			return;

		assert(full.size() == full_size);

		Eigen::MatrixXd grad;
		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, grad);
		// std::cout << grad << std::endl;
		Eigen::MatrixXd displaced;
		compute_displaced_points(full, displaced);
		double max_barrier_stiffness = 0;
		_barrier_stiffness = ipc::initial_barrier_stiffness(
			state.boundary_nodes_pos,
			displaced,
			state.boundary_edges, state.boundary_triangles,
			_dhat,
			state.avg_mass,
			grad,
			max_barrier_stiffness);
		polyfem::logger().debug("adaptive stiffness {}", _barrier_stiffness);
		// exit(0);
	}

	void NLProblem::init_timestep(const TVector &x_prev, const TVector &v_prev, const TVector &a_prev, const double dt)
	{
		this->x_prev = x_prev;
		this->v_prev = v_prev;
		this->a_prev = a_prev;
		this->dt = dt;
	}

	void NLProblem::update_quantities(const double t, const TVector &x)
	{
		if (is_time_dependent)
		{
			const double gamma = 0.5;
			const double beta = 0.25;

			v_prev = (x - x_prev) / dt;
			x_prev = x;

			// //newmark?
			// v_prev += dt * (1 - gamma) * a_prev;
			// a_prev = (x - x_prev) / (dt * dt * beta);
			// v_prev += dt * gamma * a_prev;
			// x_prev = x;

			// rhs_assembler.set_velocity_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, velocity, t);
			// rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, acceleration, t);

			rhs_computed = false;
			this->t = t;
		}
	}

	void NLProblem::substepping(const double t)
	{
		if (is_time_dependent)
		{
			rhs_computed = false;
			this->t = t;

			// rhs_assembler.set_velocity_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, velocity, t);
			// rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, acceleration, t);
		}
	}

	const Eigen::MatrixXd &NLProblem::current_rhs()
	{
		if (!rhs_computed)
		{
			rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.density, state.args["n_boundary_samples"], state.local_neumann_boundary, state.rhs, t, _current_rhs);
			rhs_computed = true;

			if (assembler.is_mixed(state.formulation()))
			{
				const int prev_size = _current_rhs.size();
				if (prev_size < full_size)
				{
					_current_rhs.conservativeResize(prev_size + state.n_pressure_bases, _current_rhs.cols());
					_current_rhs.block(prev_size, 0, state.n_pressure_bases, _current_rhs.cols()).setZero();
				}
			}
			assert(_current_rhs.size() == full_size);

			if (is_time_dependent)
			{
				const TVector tmp = state.mass * (x_prev + dt * v_prev);

				_current_rhs *= dt * dt / 2;
				_current_rhs += tmp;
			}
			rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, _current_rhs, t);
		}

		return _current_rhs;
	}

	void NLProblem::compute_displaced_points(const Eigen::MatrixXd &full, Eigen::MatrixXd &displaced)
	{
		assert(full.size() == full_size);

		const int problem_dim = state.mesh->dimension();
		displaced.resize(full.size() / problem_dim, problem_dim);
		assert(displaced.rows() * problem_dim == full.size());
		for (int i = 0; i < full.size(); i += problem_dim)
		{
			for (int d = 0; d < problem_dim; ++d)
			{
				displaced(i / problem_dim, d) = full(i + d);
			}
		}

		assert(displaced(0, 0) == full(0));
		assert(displaced(0, 1) == full(1));

		displaced += state.boundary_nodes_pos;
	}

	double NLProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		if (disable_collision)
			return 1;
		if (!state.args["has_collision"])
			return 1;

		Eigen::MatrixXd full0, full1;
		if (x0.size() == reduced_size)
			reduced_to_full(x0, full0);
		else
			full0 = x0;
		if (x1.size() == reduced_size)
			reduced_to_full(x1, full1);
		else
			full1 = x1;
		assert(full0.size() == full_size);
		assert(full1.size() == full_size);

		Eigen::MatrixXd displaced0, displaced1;

		compute_displaced_points(full0, displaced0);
		compute_displaced_points(full1, displaced1);

		const double max_step = ipc::compute_collision_free_stepsize(displaced0, displaced1, state.boundary_edges, state.boundary_triangles);
		polyfem::logger().trace("best step {}", max_step);
		return max_step;
	}

	bool NLProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		if (disable_collision)
			return true;
		if (!state.args["has_collision"])
			return true;

		Eigen::MatrixXd full0, full1;
		if (x0.size() == reduced_size)
			reduced_to_full(x0, full0);
		else
			full0 = x0;
		if (x1.size() == reduced_size)
			reduced_to_full(x1, full1);
		else
			full1 = x1;
		assert(full0.size() == full_size);
		assert(full1.size() == full_size);

		Eigen::MatrixXd displaced0, displaced1;

		compute_displaced_points(full0, displaced0);
		compute_displaced_points(full1, displaced1);

		const bool is_valid = ipc::is_step_collision_free(displaced0, displaced1, state.boundary_edges, state.boundary_triangles);

		return is_valid;
	}

	double NLProblem::value(const TVector &x)
	{
		Eigen::MatrixXd full;
		if (x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;
		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		const double elastic_energy = assembler.assemble_energy(rhs_assembler.formulation(), state.mesh->is_volume(), state.bases, gbases, full);
		const double body_energy = rhs_assembler.compute_energy(full, state.local_neumann_boundary, state.density, state.args["n_boundary_samples"], t);

		double intertia_energy = 0;
		double collision_energy = 0;
		double scaling = 1;

		if (is_time_dependent)
		{
			scaling = dt * dt / 2.0;
			const TVector tmp = full - (x_prev + dt * v_prev);

			intertia_energy = 0.5 * tmp.transpose() * state.mass * tmp;
		}

		if (!disable_collision && state.args["has_collision"])
		{
			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

			ipc::Constraints constraint_set;
			ipc::construct_constraint_set(state.boundary_nodes_pos, displaced, state.boundary_edges, state.boundary_triangles, _dhat, constraint_set);
			collision_energy = ipc::compute_barrier_potential(displaced, state.boundary_edges, state.boundary_triangles, constraint_set, _dhat);

			polyfem::logger().trace("collision_energy {}", collision_energy);
		}

		return (scaling * (elastic_energy + body_energy) + intertia_energy) / _barrier_stiffness + collision_energy;
	}

	void NLProblem::compute_cached_stiffness()
	{
		if (cached_stiffness.size() == 0)
		{
			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			if (assembler.is_linear(state.formulation()))
			{
				assembler.assemble_problem(state.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, cached_stiffness);
			}
		}
	}

	void NLProblem::gradient(const TVector &x, TVector &gradv)
	{
		Eigen::MatrixXd grad;
		gradient_no_rhs(x, grad);

		grad -= current_rhs() / _barrier_stiffness;

		full_to_reduced(grad, gradv);

		// std::cout<<"gradv\n"<<gradv<<"\n--------------\n"<<std::endl;
	}

	void NLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &grad)
	{
		//scaling * (elastic_energy + body_energy) + intertia_energy + _barrier_stiffness * collision_energy;

		Eigen::MatrixXd full;
		if (x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;
		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, grad);

		if (is_time_dependent)
		{
			grad *= dt * dt / 2.0;
			grad += state.mass * full;
		}

		grad /= _barrier_stiffness;

		if (!disable_collision && state.args["has_collision"])
		{
			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

			ipc::Constraints constraint_set;
			ipc::construct_constraint_set(state.boundary_nodes_pos, displaced, state.boundary_edges, state.boundary_triangles, _dhat, constraint_set);
			grad += ipc::compute_barrier_potential_gradient(displaced, state.boundary_edges, state.boundary_triangles, constraint_set, _dhat);
			// const double ddd = ipc::compute_minimum_distance(displaced, state.boundary_edges, state.boundary_triangles, constraint_set);
			// polyfem::logger().trace("min_dist {}", ddd);
		}

		assert(grad.size() == full_size);
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian)
	{
		THessian tmp;
		hessian_full(x, tmp);

		std::vector<Eigen::Triplet<double>> entries;

		Eigen::VectorXi indices(full_size);

		int index = 0;
		size_t kk = 0;
		for (int i = 0; i < full_size; ++i)
		{
			if (kk < state.boundary_nodes.size() && state.boundary_nodes[kk] == i)
			{
				++kk;
				indices(i) = -1;
				continue;
			}

			indices(i) = index++;
		}
		assert(index == reduced_size);

		for (int k = 0; k < tmp.outerSize(); ++k)
		{
			if (indices(k) < 0)
			{
				continue;
			}

			for (THessian::InnerIterator it(tmp, k); it; ++it)
			{
				// std::cout<<it.row()<<" "<<it.col()<<" "<<k<<std::endl;
				assert(it.col() == k);
				if (indices(it.row()) < 0 || indices(it.col()) < 0)
				{
					continue;
				}

				assert(indices(it.row()) >= 0);
				assert(indices(it.col()) >= 0);

				entries.emplace_back(indices(it.row()), indices(it.col()), it.value());
			}
		}

		hessian.resize(reduced_size, reduced_size);
		hessian.setFromTriplets(entries.begin(), entries.end());
		hessian.makeCompressed();
	}

	void NLProblem::hessian_full(const TVector &x, THessian &hessian)
	{
		//scaling * (elastic_energy + body_energy) + intertia_energy + _barrier_stiffness * collision_energy;

		Eigen::MatrixXd full;
		if (x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;

		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		if (assembler.is_linear(rhs_assembler.formulation()))
		{
			compute_cached_stiffness();
			hessian = cached_stiffness;
		}
		else
			assembler.assemble_energy_hessian(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, project_to_psd, state.bases, gbases, full, hessian);
		if (is_time_dependent)
		{
			hessian *= dt * dt / 2;
			hessian += state.mass;
		}

		hessian /= _barrier_stiffness;

		if (!disable_collision && state.args["has_collision"])
		{
			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

			ipc::Constraints constraint_set;
			ipc::construct_constraint_set(state.boundary_nodes_pos, displaced, state.boundary_edges, state.boundary_triangles, _dhat, constraint_set);
			hessian += ipc::compute_barrier_potential_hessian(displaced, state.boundary_edges, state.boundary_triangles, constraint_set, _dhat, project_to_psd);
		}

		assert(hessian.rows() == full_size);
		assert(hessian.cols() == full_size);
		// Eigen::saveMarket(tmp, "tmp.mat");
		// exit(0);
	}

	void NLProblem::full_to_reduced(const Eigen::MatrixXd &full, TVector &reduced) const
	{
		full_to_reduced_aux(state, full_size, reduced_size, full, reduced);
	}

	void NLProblem::reduced_to_full(const TVector &reduced, Eigen::MatrixXd &full)
	{
		reduced_to_full_aux(state, full_size, reduced_size, reduced, current_rhs(), full);
	}
} // namespace polyfem
