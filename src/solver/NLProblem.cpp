#include <polyfem/NLProblem.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/Types.hpp>

#include <ipc.hpp>
#include <barrier/barrier.hpp>
#include <barrier/adaptive_stiffness.hpp>

#include <igl/write_triangle_mesh.h>

#include <unsupported/Eigen/SparseExtra>

static bool disable_collision = false;

namespace polyfem
{
	using namespace polysolve;

	NLProblem::NLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat)
		: state(state), assembler(AssemblerUtils::instance()), rhs_assembler(rhs_assembler),
		  full_size((assembler.is_mixed(state.formulation()) ? state.n_pressure_bases : 0) + state.n_bases * state.mesh->dimension()),
		  reduced_size(full_size - state.boundary_nodes.size()),
		  t(t), rhs_computed(false), is_time_dependent(state.problem->is_time_dependent())
	{
		assert(!assembler.is_mixed(state.formulation()));

		_dhat_squared = dhat * dhat;
		_barrier_stiffness = 50;
	}

	void NLProblem::init(const TVector &x)
	{
		if (disable_collision || !state.args["has_collision"])
			return;

		Eigen::MatrixXd full;
		if (x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;
		assert(full.size() == full_size);

		Eigen::MatrixXd grad;
		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, grad);
		Eigen::MatrixXd displaced;
		compute_displaced_points(full, displaced);
		double max_barrier_stiffness = 0;
		_barrier_stiffness = ipc::intial_barrier_stiffness(
			state.boundary_nodes_pos,
			displaced,
			state.boundary_edges, state.boundary_triangles,
			_dhat_squared,
			1,
			grad,
			max_barrier_stiffness);
		polyfem::logger().trace("adaptive stiffness {}", _barrier_stiffness);
	}

	void NLProblem::init_timestep(const TVector &x_prev, const TVector &v_prev, const double dt)
	{
		this->x_prev = x_prev;
		this->v_prev = v_prev;
		this->dt = dt;
	}

	void NLProblem::update_quantities(const double t, const TVector &x)
	{
		if (is_time_dependent){
			v_prev = (x - x_prev) / dt;
			x_prev = x;
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
			rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, state.rhs, t, _current_rhs);
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
				const TVector tmp = state.mass*(x_prev + dt * v_prev);

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

		// static int vvvv = 0;
		// if (is_valid && displaced0.cols() == 2)
		// {
		// 	{
		// 		std::ofstream out("test_" + std::to_string(vvvv) + "_0.obj");
		// 		for (int i = 0; i < state.boundary_nodes_pos.rows(); ++i)
		// 			out << "v " << displaced0(i, 0) << " " << displaced0(i, 1) << " 0\n";

		// 		for (int i = 0; i < state.boundary_edges.rows(); ++i)
		// 			out << "l " << state.boundary_edges(i, 0) + 1 << " " << state.boundary_edges(i, 1) + 1 << "\n";
		// 		out.close();
		// 	}

		// 	{
		// 		std::ofstream out("test_" + std::to_string(vvvv) + "_1.obj");
		// 		for (int i = 0; i < state.boundary_nodes_pos.rows(); ++i)
		// 			out << "v " << displaced1(i, 0) << " " << displaced1(i, 1) << " 0\n";

		// 		for (int i = 0; i < state.boundary_edges.rows(); ++i)
		// 			out << "l " << state.boundary_edges(i, 0) + 1 << " " << state.boundary_edges(i, 1) + 1 << "\n";
		// 		out.close();
		// 	}

		// 	vvvv++;
		// }
		// if (is_valid && displaced0.cols() == 3)
		// {
		// 	igl::write_triangle_mesh("test_" + std::to_string(vvvv) + "_0.obj", displaced0, state.boundary_triangles);
		// 	igl::write_triangle_mesh("test_" + std::to_string(vvvv) + "_1.obj", displaced1, state.boundary_triangles);
		// 	vvvv++;
		// }

		// std::cout<<"state.boundary_nodes_pos + displaced\n"<<full<<std::endl;
		// std::cout<<"state.boundary_nodes_pos + displaced\n"<<displaced<<std::endl;

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
		// std::cout << "is_valid " << is_valid << std::endl;
		// polyfem::logger().trace("best step {}", ipc::compute_collision_free_stepsize(displaced0, displaced1, state.boundary_edges, state.boundary_triangles));

		// static int vvvv = 0;
		// if (is_valid && displaced0.cols() == 2)
		// {
		// 	{
		// 		std::ofstream out("test_"+std::to_string(vvvv)+"_0.obj");
		// 		for (int i = 0; i < state.boundary_nodes_pos.rows(); ++i)
		// 			out << "v " << displaced0(i, 0) << " " << displaced0(i, 1) << " 0\n";

		// 		for (int i = 0; i < state.boundary_edges.rows(); ++i)
		// 			out << "l " << state.boundary_edges(i, 0) + 1 << " " << state.boundary_edges(i, 1) + 1 << "\n";
		// 		out.close();
		// 	}

		// 	{
		// 		std::ofstream out("test_"+std::to_string(vvvv)+"_1.obj");
		// 		for (int i = 0; i < state.boundary_nodes_pos.rows(); ++i)
		// 			out << "v " << displaced1(i, 0) << " " << displaced1(i, 1) << " 0\n";

		// 		for (int i = 0; i < state.boundary_edges.rows(); ++i)
		// 			out << "l " << state.boundary_edges(i, 0) + 1 << " " << state.boundary_edges(i, 1) + 1 << "\n";
		// 		out.close();
		// 	}

		// 	vvvv++;
		// }
		// if (is_valid && displaced0.cols() == 3)
		// {
		// 	igl::write_triangle_mesh("test_" + std::to_string(vvvv) + "_0.obj", displaced0, state.boundary_triangles);
		// 	igl::write_triangle_mesh("test_" + std::to_string(vvvv) + "_1.obj", displaced1, state.boundary_triangles);
		// 	vvvv++;
		// }

		// std::cout<<"state.boundary_nodes_pos + displaced\n"<<full<<std::endl;
		// std::cout<<"state.boundary_nodes_pos + displaced\n"<<displaced<<std::endl;

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
		const double body_energy = rhs_assembler.compute_energy(full, state.local_neumann_boundary, state.args["n_boundary_samples"], t);

		double intertia_energy = 0;
		double collision_energy = 0;
		double scaling = 1;

		if(is_time_dependent)
		{
			scaling = dt * dt / 2.0;
			const TVector tmp = full - (x_prev + dt * v_prev);

			intertia_energy = 0.5 * tmp.transpose() * state.mass * tmp;
		}

		if (!disable_collision && state.args["has_collision"])
		{
			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

			// std::ofstream out("test.obj");
			// for (int i = 0; i < state.boundary_nodes_pos.rows(); ++i)
			// 	out << "v " << displaced(i, 0) << " " << displaced(i, 1) << " 0\n";

			// for (int i = 0; i < state.boundary_edges.rows(); ++i)
			// 	out << "l " << state.boundary_edges(i, 0) + 1 << " " << state.boundary_edges(i, 1) + 1 << "\n";
			// out.close();

			// std::cout<<" + displaced\n"<<full<<std::endl;
			// std::cout<<" + displaced\n"<<displaced<<std::endl;

			ipc::Candidates constraint_set;
			ipc::construct_constraint_set(displaced, state.boundary_edges, state.boundary_triangles, _dhat_squared, constraint_set);
			collision_energy = ipc::compute_barrier_potential(state.boundary_nodes_pos, displaced, state.boundary_edges, state.boundary_triangles, constraint_set, _dhat_squared);

			// if(collision_energy > 0)
			// {
			// 	if(displaced.cols() == 3)
			// 		igl::write_triangle_mesh("test_mesh.obj", displaced, state.boundary_triangles);
			// 	else {
			// 		Eigen::MatrixXd asd(displaced.rows(), 3);
			// 		asd.col(0) = displaced.col(0);
			// 		asd.col(1) = displaced.col(1);
			// 		asd.col(2).setZero();
			// 		igl::write_triangle_mesh("test_mesh.obj", asd, state.boundary_triangles);
			// 	}
			// 	// exit(0);
			// }

			polyfem::logger().trace("collision_energy {}", collision_energy);
			// const double ddd = ipc::compute_minimum_distance(displaced, state.boundary_edges, state.boundary_triangles, constraint_set);
			// polyfem::logger().trace("min_dist {}", ddd);
			// polyfem::logger().trace("barrier {}", ipc::barrier(ddd, _dhat_squared));
		}

		return scaling * (elastic_energy + body_energy + _barrier_stiffness * collision_energy) + intertia_energy;
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

		if(is_time_dependent)
		{
			Eigen::MatrixXd full;
			if (x.size() == reduced_size)
				reduced_to_full(x, full);
			else
				full = x;
			assert(full.size() == full_size);

			grad *= dt * dt / 2.0;
			grad += state.mass * full;
		}

		grad -= current_rhs();

		full_to_reduced(grad, gradv);

		// std::cout<<"gradv\n"<<gradv<<"\n--------------\n"<<std::endl;
	}

	void NLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &grad)
	{
		Eigen::MatrixXd full;
		if (x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;
		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, grad);


		if (!disable_collision && state.args["has_collision"])
		{
			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

			ipc::Candidates constraint_set;
			ipc::construct_constraint_set(displaced, state.boundary_edges, state.boundary_triangles, _dhat_squared, constraint_set);
			grad += _barrier_stiffness * ipc::compute_barrier_potential_gradient(state.boundary_nodes_pos, displaced, state.boundary_edges, state.boundary_triangles, constraint_set, _dhat_squared);
			// polyfem::logger().trace("collision grad {}", ipc::compute_barrier_potential_gradient(state.boundary_nodes_pos, displaced, state.boundary_edges, state.boundary_triangles, constraint_set, _dhat_squared).norm() * _barrier_stiffness);
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
		Eigen::MatrixXd full;
		if (x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;

		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		if (assembler.is_linear(rhs_assembler.formulation())){
			compute_cached_stiffness();
			hessian = cached_stiffness;
		}
		else
			assembler.assemble_energy_hessian(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, hessian);
		if (is_time_dependent)
		{
			hessian *= dt * dt / 2;
			hessian += state.mass;
		}

		if (!disable_collision && state.args["has_collision"])
		{
			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

			ipc::Candidates constraint_set;
			ipc::construct_constraint_set(displaced, state.boundary_edges, state.boundary_triangles, _dhat_squared, constraint_set);
			hessian += _barrier_stiffness * ipc::compute_barrier_potential_hessian(state.boundary_nodes_pos, displaced, state.boundary_edges, state.boundary_triangles, constraint_set, _dhat_squared);
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
