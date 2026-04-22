#include "InertiaForceDerivative.hpp"

#include <vector>
#include <Eigen/Core>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/quadrature/Quadrature.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	namespace
	{
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
	} // namespace

	void InertiaForceDerivative::force_shape_derivative(
		const InertiaForm &form,
		bool is_volume,
		const int n_geom_bases,
		const double t,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const assembler::Mass &assembler,
		const assembler::AssemblyValsCache &ass_vals_cache,
		const Eigen::MatrixXd &velocity,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &term)
	{
		(void)form;

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
					const double rho = assembler.density()(quadrature.points.row(q), vals.val.row(q), t, e);
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
