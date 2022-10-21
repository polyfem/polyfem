#include "BodyForm.hpp"

#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

namespace polyfem::solver
{
	namespace {
		class LocalThreadVecStorage
		{
		public:
			Eigen::MatrixXd vec;
			assembler::ElementAssemblyValues vals, gvals;
			QuadratureVector da;

			LocalThreadVecStorage(const int size)
			{
				vec.resize(size, 1);
				vec.setZero();
			}
		};
	}
	BodyForm::BodyForm(const int ndof,
					   const int n_pressure_bases,
					   const std::vector<int> &boundary_nodes,
					   const std::vector<mesh::LocalBoundary> &local_boundary,
					   const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
					   const int n_boundary_samples,
					   const Eigen::MatrixXd &rhs,
					   const assembler::RhsAssembler &rhs_assembler,
					   const Density &density,
					   const bool apply_DBC,
					   const bool is_formulation_mixed,
					   const bool is_time_dependent)
		: ndof_(ndof),
		  n_pressure_bases_(n_pressure_bases),
		  boundary_nodes_(boundary_nodes),
		  local_boundary_(local_boundary),
		  local_neumann_boundary_(local_neumann_boundary),
		  n_boundary_samples_(n_boundary_samples),
		  rhs_(rhs),
		  rhs_assembler_(rhs_assembler),
		  density_(density),
		  apply_DBC_(apply_DBC),
		  is_formulation_mixed_(is_formulation_mixed)
	{
		t_ = 0;
		if (!is_time_dependent)
			update_current_rhs(Eigen::VectorXd());
	}

	double BodyForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return rhs_assembler_.compute_energy(x, local_neumann_boundary_, density_, n_boundary_samples_, t_);
	}

	void BodyForm::first_derivative_unweighted(const Eigen::VectorXd &, Eigen::VectorXd &gradv) const
	{
		// REMEMBER -!!!!!
		gradv = -current_rhs_;
	}

	void BodyForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		this->t_ = t;
		update_current_rhs(x);
	}

	void BodyForm::update_current_rhs(const Eigen::VectorXd &x)
	{
		rhs_assembler_.compute_energy_grad(
			local_boundary_, boundary_nodes_, density_,
			n_boundary_samples_, local_neumann_boundary_,
			rhs_, t_, current_rhs_);

		if (is_formulation_mixed_ && current_rhs_.size() < ndof_)
		{
			current_rhs_.conservativeResize(
				current_rhs_.rows() + n_pressure_bases_, current_rhs_.cols());
			current_rhs_.bottomRows(n_pressure_bases_).setZero();
		}

		// Apply Neumann boundary conditions
		rhs_assembler_.set_bc(
			std::vector<mesh::LocalBoundary>(), std::vector<int>(),
			n_boundary_samples_, local_neumann_boundary_,
			current_rhs_, x, t_);

		// Apply Dirichlet boundary conditions
		if (apply_DBC_)
			rhs_assembler_.set_bc(
				local_boundary_, boundary_nodes_,
				n_boundary_samples_, std::vector<mesh::LocalBoundary>(),
				current_rhs_, x, t_);
	}

	void BodyForm::force_shape_derivative(const int n_verts, const Eigen::MatrixXd &x, const Eigen::MatrixXd &adjoint, Eigen::VectorXd &term)
	{
		const auto &bases = rhs_assembler_.bases();
		const auto &gbases = rhs_assembler_.gbases();
		const int dim = rhs_assembler_.mesh().dimension();
		const int actual_dim = rhs_assembler_.problem().is_scalar() ? 1 : dim;

		const int n_elements = int(bases.size());
		term.setZero(n_verts * dim, 1);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			for (int e = start; e < end; ++e)
			{
				assembler::ElementAssemblyValues &vals = local_storage.vals;
				rhs_assembler_.ass_vals_cache().compute(e, rhs_assembler_.mesh().is_volume(), bases[e], gbases[e], vals);
				assembler::ElementAssemblyValues &gvals = local_storage.gvals;
				gvals.compute(e, rhs_assembler_.mesh().is_volume(), vals.quadrature.points, gbases[e], gbases[e]);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				local_storage.da = vals.det.array() * quadrature.weights.array();

				Eigen::MatrixXd p, grad_p;
				io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, adjoint, p, grad_p);

				Eigen::MatrixXd rhs_function;
				rhs_assembler_.problem().rhs(rhs_assembler_.assembler(), rhs_assembler_.formulation(), vals.val, 0, rhs_function);
				rhs_function *= -1;
				for (int q = 0; q < vals.val.rows(); q++)
				{
					const double rho = density_(quadrature.points.row(q), vals.val.row(q), e);
					rhs_function.row(q) *= rho;
				}

				for (int q = 0; q < local_storage.da.size(); ++q)
				{
					const double value = p.row(q).dot(rhs_function.row(q)) * local_storage.da(q);
					for (auto &v : gvals.basis_values)
					{
						local_storage.vec.block(v.global[0].index * dim, 0, dim, 1) += value * v.grad_t_m.row(q).transpose();
					}
				}
			}
		});

		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;
	}
} // namespace polyfem::solver
