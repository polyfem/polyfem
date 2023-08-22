#include "InertiaForm.hpp"

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

namespace polyfem::solver
{
	namespace {

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

		double dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }
	}
	
	InertiaForm::InertiaForm(const StiffnessMatrix &mass,
							 const time_integrator::ImplicitTimeIntegrator &time_integrator)
		: mass_(mass), time_integrator_(time_integrator)
	{
		assert(mass.size() != 0);
	}

	double InertiaForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd tmp = x - time_integrator_.x_tilde();
		// FIXME: DBC on x tilde
		const double prod = tmp.transpose() * mass_ * tmp;
		const double energy = 0.5 * prod;
		return energy;
	}

	void InertiaForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = mass_ * (x - time_integrator_.x_tilde());
	}

	void InertiaForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = mass_;
	}

	void InertiaForm::force_shape_derivative(
		bool is_volume,
		const int n_geom_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const assembler::Mass &assembler,
		const assembler::AssemblyValsCache &ass_vals_cache,
		const Eigen::MatrixXd &velocity, 
		const Eigen::MatrixXd &adjoint, 
		Eigen::VectorXd &term)
	{
		const int dim = is_volume ? 3 : 2;
		const int n_elements = int(bases.size());
		term.setZero(n_geom_bases * dim, 1);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

			for (int e = start; e < end; ++e)
			{
				assembler::ElementAssemblyValues &vals = local_storage.vals;
				ass_vals_cache.compute(e, is_volume, bases[e], geom_bases[e], vals);
				assembler::ElementAssemblyValues gvals;
				gvals.compute(e, is_volume, vals.quadrature.points, geom_bases[e], geom_bases[e]);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				local_storage.da = vals.det.array() * quadrature.weights.array();

				Eigen::MatrixXd vel, grad_vel, p, grad_p;
				io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);
				io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, velocity, vel, grad_vel);

				for (int q = 0; q < local_storage.da.size(); ++q)
				{
					const double rho = assembler.density()(quadrature.points.row(q), vals.val.row(q), e);
					const double value = rho * dot(p.row(q), vel.row(q)) * local_storage.da(q);
					for (const auto &v : gvals.basis_values)
					{
						local_storage.vec.block(v.global[0].index * dim, 0, dim, 1) += v.grad_t_m.row(q).transpose() * value;
					}
				}
			}
		});

		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;
	}
} // namespace polyfem::solver
