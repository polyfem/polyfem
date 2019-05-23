#include "MassMatrixAssembler.hpp"

#include <polyfem/Logger.hpp>

#ifdef POLYFEM_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>
#endif

namespace polyfem
{
	namespace
	{
		class LocalThreadMatStorage
		{
		public:
			std::vector< Eigen::Triplet<double> > entries;
			StiffnessMatrix tmp_mat;
			StiffnessMatrix mass_mat;

			LocalThreadMatStorage(const int buffer_size, const int mat_size)
			{
				entries.reserve(buffer_size);
				tmp_mat.resize(mat_size, mat_size);
				mass_mat.resize(mat_size, mat_size);
			}
		};
	}

	void MassMatrixAssembler::assemble(
		const bool is_volume,
		const int size,
		const int n_basis,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		StiffnessMatrix &mass) const
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * size);
		logger().debug("buffer_size {}", buffer_size);

		mass.resize(n_basis*size, n_basis*size);
		mass.setZero();

#ifdef POLYFEM_WITH_TBB
		typedef tbb::enumerable_thread_specific< LocalThreadMatStorage > LocalStorage;
		LocalStorage storages(LocalThreadMatStorage(buffer_size, mass.rows()));
#else
		LocalThreadMatStorage loc_storage(buffer_size, mass.rows());
#endif

		const int n_bases = int(bases.size());

#ifdef POLYFEM_WITH_TBB
		tbb::parallel_for( tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
		LocalStorage::reference loc_storage = storages.local();
		for (int e = r.begin(); e != r.end(); ++e) {
#else
		for(int e=0; e < n_bases; ++e) {
#endif
			ElementAssemblyValues vals;
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			const QuadratureVector da = vals.det.array() * quadrature.weights.array();
			const int n_loc_bases = int(vals.basis_values.size());

			for(int i = 0; i < n_loc_bases; ++i)
			{
				const auto &global_i = vals.basis_values[i].global;

				for(int j = 0; j <= i; ++j)
				{
					const auto &global_j = vals.basis_values[j].global;

					const double tmp = (vals.basis_values[i].val.array() * vals.basis_values[j].val.array() * da.array()).sum();
					if (std::abs(tmp) < 1e-30) { continue; }


					for(int n = 0; n < size; ++n)
					{
						//local matrix is diagonal
						const int m = n;
						// for(int m = 0; m < size; ++m)
						{
							const double local_value = tmp; //val(n*size+m);
							for(size_t ii = 0; ii < global_i.size(); ++ii)
							{
								const auto gi = global_i[ii].index*size+m;
								const auto wi = global_i[ii].val;

								for(size_t jj = 0; jj < global_j.size(); ++jj)
								{
									const auto gj = global_j[jj].index*size+n;
									const auto wj = global_j[jj].val;

									loc_storage.entries.emplace_back(gi, gj, local_value * wi * wj);
									if (j < i) {
										loc_storage.entries.emplace_back(gj, gi, local_value * wj * wi);
									}

									if(loc_storage.entries.size() >= 1e8)
									{
										loc_storage.tmp_mat.setFromTriplets(loc_storage.entries.begin(), loc_storage.entries.end());
										loc_storage.mass_mat += loc_storage.tmp_mat;
										loc_storage.mass_mat.makeCompressed();

										loc_storage.entries.clear();
										logger().debug("cleaning memory...");
									}
								}
							}
						}
					}

				// t1.stop();
				// if (!vals.has_parameterization) { std::cout << "-- t1: " << t1.getElapsedTime() << std::endl; }

				}

			}

			// timer.stop();
			// if (!vals.has_parameterization) { std::cout << "-- Timer: " << timer.getElapsedTime() << std::endl; }
#ifdef POLYFEM_WITH_TBB
		}});
#else
		}
#endif

#ifdef POLYFEM_WITH_TBB
		for (LocalStorage::iterator i = storages.begin(); i != storages.end();  ++i)
		{
			mass += i->mass_mat;
			i->tmp_mat.setFromTriplets(i->entries.begin(), i->entries.end());
			mass += i->tmp_mat;
		}
#else
		mass = loc_storage.mass_mat;
		loc_storage.tmp_mat.setFromTriplets(loc_storage.entries.begin(), loc_storage.entries.end());
		mass += loc_storage.tmp_mat;
#endif
		mass.makeCompressed();
	}
}