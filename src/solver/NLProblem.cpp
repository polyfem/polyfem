#include <polyfem/NLProblem.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/Types.hpp>

#include <unsupported/Eigen/SparseExtra>

namespace polyfem
{
	using namespace polysolve;
	// NLProblem::NLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const int full_size, const int reduced_size)
	// : state(state), assembler(AssemblerUtils::instance()), rhs_assembler(rhs_assembler),
	// full_size(full_size), reduced_size(reduced_size), t(t), rhs_computed(false)
	// { }

NLProblem::NLProblem(State &state, const RhsAssembler &rhs_assembler, const double t)
	: state(state), assembler(AssemblerUtils::instance()), rhs_assembler(rhs_assembler),
	  full_size((assembler.is_mixed(state.formulation()) ? state.n_pressure_bases : 0) + state.n_bases * state.mesh->dimension()),
	  reduced_size(full_size - state.boundary_nodes.size()),
	  t(t), rhs_computed(false)
{ }

	const Eigen::MatrixXd &NLProblem::current_rhs()
	{
		if(!rhs_computed)
		{
			rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, state.rhs, t, _current_rhs);
			rhs_computed = true;

			if (assembler.is_mixed(state.formulation())){
				const int prev_size = _current_rhs.size();
				if(prev_size < full_size){
					_current_rhs.conservativeResize(prev_size + state.n_pressure_bases, _current_rhs.cols());
					_current_rhs.block(prev_size, 0, state.n_pressure_bases, _current_rhs.cols()).setZero();
				}
			}
			assert(_current_rhs.size() == full_size);
		}

		return _current_rhs;
	}

	double NLProblem::value(const TVector &x) {
		if(assembler.is_gradient_based(state.formulation()))
		{
			// Eigen::MatrixXd grad;
			TVector grad;
			gradient(x, grad);
			// grad -= current_rhs();
			// return grad.lpNorm<Eigen::Infinity>();
			return grad.norm();
		}

		Eigen::MatrixXd full;
		if(x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;
		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		const double elastic_energy = assembler.assemble_energy(rhs_assembler.formulation(), state.mesh->is_volume(), state.bases, gbases, full);
		const double body_energy 	= rhs_assembler.compute_energy(full, state.local_neumann_boundary, state.args["n_boundary_samples"], t);

		return elastic_energy + body_energy;
	}

	void NLProblem::compute_cached_stiffness()
	{
		if (cached_stiffness.size() == 0)
		{
			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

			StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;
			assembler.assemble_problem(state.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, velocity_stiffness);
			assembler.assemble_mixed_problem(state.formulation(), state.mesh->is_volume(), state.n_pressure_bases, state.n_bases, state.pressure_bases, state.bases, gbases, mixed_stiffness);
			assembler.assemble_pressure_problem(state.formulation(), state.mesh->is_volume(), state.n_pressure_bases, state.pressure_bases, gbases, pressure_stiffness);

			const int problem_dim = state.problem->is_scalar() ? 1 : state.mesh->dimension();

			AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, false, //assembler.is_fluid(state.formulation()),
												 velocity_stiffness, mixed_stiffness, pressure_stiffness,
												 cached_stiffness);
		}
	}

	void NLProblem::gradient(const TVector &x, TVector &gradv) {
		Eigen::MatrixXd grad;
		gradient_no_rhs(x, grad);
		grad -= current_rhs();

		full_to_reduced(grad, gradv);

		// std::cout<<"gradv\n"<<gradv<<"\n--------------\n"<<std::endl;
	}

	void NLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &grad)
	{
		Eigen::MatrixXd full;
		if(x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;
		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, grad);

		if(assembler.is_mixed(state.formulation()))
		{
			const int prev_size = grad.size();
			grad.conservativeResize(prev_size + state.n_pressure_bases, grad.cols());
			grad.block(prev_size, 0, state.n_pressure_bases, grad.cols()).setZero();

			compute_cached_stiffness();
			grad += cached_stiffness * full;
		}
		assert(grad.size() == full_size);
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian) {
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
		if(x.size() == reduced_size)
			reduced_to_full(x, full);
		else
			full = x;

		assert(full.size() == full_size);

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		assembler.assemble_energy_hessian(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, full, hessian);

		if (assembler.is_mixed(state.formulation()))
		{
			StiffnessMatrix velocity_stiffness = hessian, mixed_stiffness, pressure_stiffness;
			const int problem_dim = state.problem->is_scalar() ? 1 : state.mesh->dimension();

			AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, false, //assembler.is_fluid(state.formulation()),
												 velocity_stiffness, mixed_stiffness, pressure_stiffness,
												 hessian);

			compute_cached_stiffness();
			hessian += cached_stiffness;
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
}