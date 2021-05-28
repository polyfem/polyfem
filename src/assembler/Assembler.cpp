#include <polyfem/Assembler.hpp>
#include <polyfem/par_for.hpp>

#include <polyfem/Laplacian.hpp>
#include <polyfem/Helmholtz.hpp>
#include <polyfem/Bilaplacian.hpp>

#include <polyfem/LinearElasticity.hpp>
#include <polyfem/HookeLinearElasticity.hpp>
#include <polyfem/SaintVenantElasticity.hpp>
#include <polyfem/NeoHookeanElasticity.hpp>
#include <polyfem/MultiModel.hpp>
// #include <polyfem/OgdenElasticity.hpp>

#include <polyfem/Stokes.hpp>
#include <polyfem/NavierStokes.hpp>
#include <polyfem/IncompressibleLinElast.hpp>

#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

#include <ipc/utils/eigen_ext.hpp>

#ifdef POLYFEM_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
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
			SpareMatrixCache cache;
			ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadMatStorage()
			{
			}

			LocalThreadMatStorage(const int buffer_size, const int rows, const int cols)
			{
				init(buffer_size, rows, cols);
			}

			LocalThreadMatStorage(const int buffer_size, const SpareMatrixCache &c)
			{
				init(buffer_size, c);
			}

			void init(const int buffer_size, const int rows, const int cols)
			{
				assert(rows == cols);
				cache.reserve(buffer_size);
				cache.init(rows);
			}

			void init(const int buffer_size, const SpareMatrixCache &c)
			{
				cache.reserve(buffer_size);
				cache.init(c);
			}
		};

		class LocalThreadVecStorage
		{
		public:
			Eigen::MatrixXd vec;
			ElementAssemblyValues vals;
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
			ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadScalarStorage()
			{
				val = 0;
			}
		};

#ifdef POLYFEM_WITH_TBB
		template <typename LTM>
		void merge_matrices(tbb::enumerable_thread_specific<LTM> &storages, SpareMatrixCache &mat)
		{
			for (auto &t : storages)
			{
				t.cache.prune();
				mat += t.cache;
			}
			// std::vector<LTM *> flat_view;
			// for (auto i = storages.begin(); i != storages.end(); ++i)
			// {
			// 	flat_view.emplace_back(&*i);
			// }

			// mat = tbb::parallel_reduce(
			// 	tbb::blocked_range<int>(0, flat_view.size()), mat,
			// 	[&](const tbb::blocked_range<int> &r, const SpareMatrixCache &m)
			// 	{
			// 		SpareMatrixCache tmp = m;
			// 		for (int e = r.begin(); e != r.end(); ++e)
			// 		{
			// 			const auto i = flat_view[e];
			// 			i->cache.prune();

			// 			tmp += i->cache;
			// 		}

			// 		return tmp;
			// 	},
			// 	[](const SpareMatrixCache &a, const SpareMatrixCache &b)
			// 	{
			// 		return a + b;
			// 	});
		}
#endif
	} // namespace

	template <class LocalAssembler>
	void Assembler<LocalAssembler>::assemble(
		const bool is_volume,
		const int n_basis,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		StiffnessMatrix &stiffness) const
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * local_assembler_.size());
		// #ifdef POLYFEM_WITH_TBB
		// 		buffer_size /= tbb::task_scheduler_init::default_num_threads();
		// #endif
		logger().debug("buffer_size {}", buffer_size);
		try
		{
			stiffness.resize(n_basis * local_assembler_.size(), n_basis * local_assembler_.size());
			stiffness.setZero();

#if defined(POLYFEM_WITH_CPP_THREADS)
			std::vector<LocalThreadMatStorage> storages(polyfem::get_n_threads());
#elif defined(POLYFEM_WITH_TBB)
			typedef tbb::enumerable_thread_specific<LocalThreadMatStorage> LocalStorage;
			LocalStorage storages(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));
#else
			LocalThreadMatStorage loc_storage(buffer_size, stiffness.rows(), stiffness.cols());
#endif

			const int n_bases = int(bases.size());
			igl::Timer timerg;
			timerg.start();
#if defined(POLYFEM_WITH_CPP_THREADS)
			polyfem::par_for(n_bases, [&](int start, int end, int t)
							 {
								 auto &loc_storage = storages[t];
								 loc_storage.init(buffer_size, stiffness.rows(), stiffness.cols());
								 for (int e = start; e < end; ++e)
								 {
#elif defined(POLYFEM_WITH_TBB)
			tbb::parallel_for(tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
				LocalStorage::reference loc_storage = storages.local();

				for (int e = r.begin(); e != r.end(); ++e)
				{
#else
			for (int e = 0; e < n_bases; ++e)
			{
#endif
									 ElementAssemblyValues &vals = loc_storage.vals;
									 // igl::Timer timer; timer.start();
									 // vals.compute(e, is_volume, bases[e], gbases[e]);
									 cache.compute(e, is_volume, bases[e], gbases[e], vals);

									 const Quadrature &quadrature = vals.quadrature;

									 assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
									 loc_storage.da = vals.det.array() * quadrature.weights.array();
									 const int n_loc_bases = int(vals.basis_values.size());

									 for (int i = 0; i < n_loc_bases; ++i)
									 {
										 // const AssemblyValues &values_i = vals.basis_values[i];
										 // const Eigen::MatrixXd &gradi = values_i.grad_t_m;
										 const auto &global_i = vals.basis_values[i].global;

										 for (int j = 0; j <= i; ++j)
										 {
											 // const AssemblyValues &values_j = vals.basis_values[j];
											 // const Eigen::MatrixXd &gradj = values_j.grad_t_m;
											 const auto &global_j = vals.basis_values[j].global;

											 const auto stiffness_val = local_assembler_.assemble(vals, i, j, loc_storage.da);
											 assert(stiffness_val.size() == local_assembler_.size() * local_assembler_.size());

											 // igl::Timer t1; t1.start();
											 for (int n = 0; n < local_assembler_.size(); ++n)
											 {
												 for (int m = 0; m < local_assembler_.size(); ++m)
												 {
													 const double local_value = stiffness_val(n * local_assembler_.size() + m);
													 if (std::abs(local_value) < 1e-30)
													 {
														 continue;
													 }

													 for (size_t ii = 0; ii < global_i.size(); ++ii)
													 {
														 const auto gi = global_i[ii].index * local_assembler_.size() + m;
														 const auto wi = global_i[ii].val;

														 for (size_t jj = 0; jj < global_j.size(); ++jj)
														 {
															 const auto gj = global_j[jj].index * local_assembler_.size() + n;
															 const auto wj = global_j[jj].val;

															 loc_storage.cache.add_value(gi, gj, local_value * wi * wj);
															 if (j < i)
															 {
																 loc_storage.cache.add_value(gj, gi, local_value * wj * wi);
															 }

															 if (loc_storage.cache.entries_size() >= 1e8)
															 {
																 loc_storage.cache.prune();
																 logger().debug("cleaning memory. Current storage: {}. mat nnz: {}", loc_storage.cache.capacity(), loc_storage.cache.non_zeros());
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
#if defined(POLYFEM_WITH_CPP_THREADS) || defined(POLYFEM_WITH_TBB)
								 }
#if defined(POLYFEM_WITH_CPP_THREADS)
								 loc_storage.cache.prune();
#endif
							 });
#else
				}
#endif
			timerg.stop();
			logger().debug("done separate assembly {}s...", timerg.getElapsedTime());

			timerg.start();
#if defined(POLYFEM_WITH_CPP_THREADS)
			for (auto &t : storages)
			{
				stiffness += t.cache.get_matrix(false);
			}
			stiffness.makeCompressed();
#elif defined(POLYFEM_WITH_TBB)
				SpareMatrixCache tmp_cache;
				merge_matrices(storages, tmp_cache);

				stiffness = tmp_cache.get_matrix(false);
#else
				loc_storage.cache.prune();
				stiffness = loc_storage.cache.get_matrix(false);
#endif

			timerg.stop();
			logger().debug("done merge assembly {}s...", timerg.getElapsedTime());
		}
		catch (std::bad_alloc &ba)
		{
			logger().error("bad alloc {}", ba.what());
			exit(0);
		}

		// stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		// stiffness.setFromTriplets(entries.begin(), entries.end());
	}

	template <class LocalAssembler>
	void MixedAssembler<LocalAssembler>::assemble(
		const bool is_volume,
		const int n_psi_basis,
		const int n_phi_basis,
		const std::vector<ElementBases> &psi_bases,
		const std::vector<ElementBases> &phi_bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &psi_cache,
		const AssemblyValsCache &phi_cache,
		StiffnessMatrix &stiffness) const
	{
		assert(phi_bases.size() == psi_bases.size());

		const int buffer_size = std::min(long(1e8), long(std::max(n_psi_basis, n_phi_basis)) * std::max(local_assembler_.rows(), local_assembler_.cols()));
		logger().debug("buffer_size {}", buffer_size);

		stiffness.resize(n_phi_basis * local_assembler_.rows(), n_psi_basis * local_assembler_.cols());
		stiffness.setZero();

#if defined(POLYFEM_WITH_CPP_THREADS)
		std::vector<LocalThreadMatStorage> storages(polyfem::get_n_threads());
#elif defined(POLYFEM_WITH_TBB)
			typedef tbb::enumerable_thread_specific<LocalThreadMatStorage> LocalStorage;
			LocalStorage storages(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));
#else
			LocalThreadMatStorage loc_storage(buffer_size, stiffness.rows(), stiffness.cols());
			ElementAssemblyValues psi_vals, phi_vals;
#endif

		const int n_bases = int(phi_bases.size());
		igl::Timer timerg;
		timerg.start();
#if defined(POLYFEM_WITH_CPP_THREADS)
		polyfem::par_for(n_bases, [&](int start, int end, int t)
						 {
							 auto &loc_storage = storages[t];
							 loc_storage.init(buffer_size, stiffness.rows(), stiffness.cols());
							 ElementAssemblyValues psi_vals, phi_vals;
							 for (int e = start; e < end; ++e)
							 {
#elif defined(POLYFEM_WITH_TBB)
		tbb::parallel_for(tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
				LocalStorage::reference loc_storage = storages.local();
				ElementAssemblyValues psi_vals, phi_vals;
				for (int e = r.begin(); e != r.end(); ++e)
				{
#else
			for (int e = 0; e < n_bases; ++e)
			{
#endif
								 // psi_vals.compute(e, is_volume, psi_bases[e], gbases[e]);
								 // phi_vals.compute(e, is_volume, phi_bases[e], gbases[e]);
								 psi_cache.compute(e, is_volume, psi_bases[e], gbases[e], psi_vals);
								 phi_cache.compute(e, is_volume, phi_bases[e], gbases[e], phi_vals);

								 const Quadrature &quadrature = phi_vals.quadrature;

								 assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
								 loc_storage.da = phi_vals.det.array() * quadrature.weights.array();
								 const int n_phi_loc_bases = int(phi_vals.basis_values.size());
								 const int n_psi_loc_bases = int(psi_vals.basis_values.size());

								 for (int i = 0; i < n_psi_loc_bases; ++i)
								 {
									 const auto &global_i = psi_vals.basis_values[i].global;

									 for (int j = 0; j < n_phi_loc_bases; ++j)
									 {
										 const auto &global_j = phi_vals.basis_values[j].global;

										 const auto stiffness_val = local_assembler_.assemble(psi_vals, phi_vals, i, j, loc_storage.da);
										 assert(stiffness_val.size() == local_assembler_.rows() * local_assembler_.cols());

										 // igl::Timer t1; t1.start();
										 for (int n = 0; n < local_assembler_.rows(); ++n)
										 {
											 for (int m = 0; m < local_assembler_.cols(); ++m)
											 {
												 const double local_value = stiffness_val(n * local_assembler_.cols() + m);
												 if (std::abs(local_value) < 1e-30)
												 {
													 continue;
												 }

												 for (size_t ii = 0; ii < global_i.size(); ++ii)
												 {
													 const auto gi = global_i[ii].index * local_assembler_.cols() + m;
													 const auto wi = global_i[ii].val;

													 for (size_t jj = 0; jj < global_j.size(); ++jj)
													 {
														 const auto gj = global_j[jj].index * local_assembler_.rows() + n;
														 const auto wj = global_j[jj].val;

														 loc_storage.cache.add_value(gj, gi, local_value * wi * wj);

														 if (loc_storage.cache.entries_size() >= 1e8)
														 {
															 loc_storage.cache.prune();
															 logger().debug("cleaning memory...");
														 }
													 }
												 }
											 }
										 }
									 }
								 }
#if defined(POLYFEM_WITH_CPP_THREADS) || defined(POLYFEM_WITH_TBB)
							 }
#if defined(POLYFEM_WITH_CPP_THREADS)
							 loc_storage.cache.prune();
#endif
						 });
#else
				}
#endif

		timerg.stop();
		logger().trace("done separate assembly {}s...", timerg.getElapsedTime());

		timerg.start();

#if defined(POLYFEM_WITH_CPP_THREADS)
		for (auto &t : storages)
		{
			stiffness += t.cache.get_matrix(false);
		}
		stiffness.makeCompressed();
#elif defined(POLYFEM_WITH_TBB)
				SpareMatrixCache tmp_cache;
				merge_matrices(storages, tmp_cache);
				stiffness = tmp_cache.get_matrix(false);

#else
				loc_storage.cache.prune();
				stiffness = loc_storage.cache.get_matrix(false);
#endif
		timerg.stop();
		logger().trace("done merge assembly {}s...", timerg.getElapsedTime());

		// stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		// stiffness.setFromTriplets(entries.begin(), entries.end());
	}

	template <class LocalAssembler>
	void NLAssembler<LocalAssembler>::assemble_grad(
		const bool is_volume,
		const int n_basis,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const Eigen::MatrixXd &displacement,
		Eigen::MatrixXd &rhs) const
	{
		rhs.resize(n_basis * local_assembler_.size(), 1);
		rhs.setZero();

#if defined(POLYFEM_WITH_CPP_THREADS)
		std::vector<LocalThreadVecStorage> storages(polyfem::get_n_threads(), rhs.size());
#elif defined(POLYFEM_WITH_TBB)
				typedef tbb::enumerable_thread_specific<LocalThreadVecStorage> LocalStorage;
				LocalStorage storages(LocalThreadVecStorage(rhs.size()));
#else
				LocalThreadVecStorage loc_storage(rhs.size());
#endif

		const int n_bases = int(bases.size());

#if defined(POLYFEM_WITH_CPP_THREADS)
		polyfem::par_for(n_bases, [&](int start, int end, int t)
						 {
							 auto &loc_storage = storages[t];
							 assert(loc_storage.vec.size() == rhs.size());
							 for (int e = start; e < end; ++e)
							 {
#elif defined(POLYFEM_WITH_TBB)
		tbb::parallel_for(tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
					LocalStorage::reference loc_storage = storages.local();
					for (int e = r.begin(); e != r.end(); ++e)
					{
#else
				for (int e = 0; e < n_bases; ++e)
				{
#endif
								 // igl::Timer timer; timer.start();

								 ElementAssemblyValues &vals = loc_storage.vals;
								 // vals.compute(e, is_volume, bases[e], gbases[e]);
								 cache.compute(e, is_volume, bases[e], gbases[e], vals);

								 const Quadrature &quadrature = vals.quadrature;

								 assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
								 loc_storage.da = vals.det.array() * quadrature.weights.array();
								 const int n_loc_bases = int(vals.basis_values.size());

								 const auto val = local_assembler_.assemble_grad(vals, displacement, loc_storage.da);
								 assert(val.size() == n_loc_bases * local_assembler_.size());

								 for (int j = 0; j < n_loc_bases; ++j)
								 {
									 const auto &global_j = vals.basis_values[j].global;

									 // igl::Timer t1; t1.start();
									 for (int m = 0; m < local_assembler_.size(); ++m)
									 {
										 const double local_value = val(j * local_assembler_.size() + m);
										 if (std::abs(local_value) < 1e-30)
										 {
											 continue;
										 }

										 for (size_t jj = 0; jj < global_j.size(); ++jj)
										 {
											 const auto gj = global_j[jj].index * local_assembler_.size() + m;
											 const auto wj = global_j[jj].val;

											 loc_storage.vec(gj) += local_value * wj;
										 }
									 }

									 // t1.stop();
									 // if (!vals.has_parameterization) { std::cout << "-- t1: " << t1.getElapsedTime() << std::endl; }
								 }

				// timer.stop();
				// if (!vals.has_parameterization) { std::cout << "-- Timer: " << timer.getElapsedTime() << std::endl; }

#if defined(POLYFEM_WITH_CPP_THREADS) || defined(POLYFEM_WITH_TBB)
							 }
						 });
#else
					}
#endif

#if defined(POLYFEM_WITH_CPP_THREADS)
		for (const auto &t : storages)
		{
			rhs += t.vec;
		}
#elif defined(POLYFEM_WITH_TBB)
					for (LocalStorage::iterator i = storages.begin(); i != storages.end(); ++i)
					{
						rhs += i->vec;
					}
#else
					rhs = loc_storage.vec;
#endif
	}

	template <class LocalAssembler>
	void NLAssembler<LocalAssembler>::assemble_hessian(
		const bool is_volume,
		const int n_basis,
		const bool project_to_psd,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const Eigen::MatrixXd &displacement,
		SpareMatrixCache &mat_cache,
		StiffnessMatrix &grad) const
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * local_assembler_.size());
		// std::cout<<"buffer_size "<<buffer_size<<std::endl;

		// grad.resize(n_basis * local_assembler_.size(), n_basis * local_assembler_.size());
		// grad.setZero();

		mat_cache.init(n_basis * local_assembler_.size());
		mat_cache.set_zero();

#if defined(POLYFEM_WITH_CPP_THREADS)
		std::vector<LocalThreadMatStorage> storages(polyfem::get_n_threads());
#elif defined(POLYFEM_WITH_TBB)
					typedef tbb::enumerable_thread_specific<LocalThreadMatStorage> LocalStorage;
					LocalStorage storages(LocalThreadMatStorage(buffer_size, mat_cache));
#else
					LocalThreadMatStorage loc_storage(buffer_size, mat_cache);
#endif

		const int n_bases = int(bases.size());
		igl::Timer timerg;
		timerg.start();

#if defined(POLYFEM_WITH_CPP_THREADS)
		polyfem::par_for(n_bases, [&](int start, int end, int t)
						 {
							 auto &loc_storage = storages[t];
							 loc_storage.init(buffer_size, mat_cache);
							 for (int e = start; e < end; ++e)
							 {
#elif defined(POLYFEM_WITH_TBB)
		tbb::parallel_for(tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
						LocalStorage::reference loc_storage = storages.local();
						for (int e = r.begin(); e != r.end(); ++e)
						{
#else
					for (int e = 0; e < n_bases; ++e)
					{
#endif
								 ElementAssemblyValues &vals = loc_storage.vals;
								 cache.compute(e, is_volume, bases[e], gbases[e], vals);

								 const Quadrature &quadrature = vals.quadrature;

								 assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
								 loc_storage.da = vals.det.array() * quadrature.weights.array();
								 const int n_loc_bases = int(vals.basis_values.size());

								 auto stiffness_val = local_assembler_.assemble_hessian(vals, displacement, loc_storage.da);
								 assert(stiffness_val.rows() == n_loc_bases * local_assembler_.size());
								 assert(stiffness_val.cols() == n_loc_bases * local_assembler_.size());

								 if (project_to_psd)
									 stiffness_val = ipc::project_to_psd(stiffness_val);

								 // bool has_nan = false;
								 // for(int k = 0; k < stiffness_val.size(); ++k)
								 // {
								 // 	if(std::isnan(stiffness_val(k)))
								 // 	{
								 // 		has_nan = true;
								 // 		break;
								 // 	}
								 // }

								 // if(has_nan)
								 // {
								 // 	loc_storage.entries.emplace_back(0, 0, std::nan(""));
								 // 	break;
								 // }

								 for (int i = 0; i < n_loc_bases; ++i)
								 {
									 const auto &global_i = vals.basis_values[i].global;

									 for (int j = 0; j < n_loc_bases; ++j)
									 // for(int j = 0; j <= i; ++j)
									 {
										 const auto &global_j = vals.basis_values[j].global;

										 for (int n = 0; n < local_assembler_.size(); ++n)
										 {
											 for (int m = 0; m < local_assembler_.size(); ++m)
											 {
												 const double local_value = stiffness_val(i * local_assembler_.size() + m, j * local_assembler_.size() + n);
												 //  if (std::abs(local_value) < 1e-30)
												 //  {
												 // 	 continue;
												 //  }

												 for (size_t ii = 0; ii < global_i.size(); ++ii)
												 {
													 const auto gi = global_i[ii].index * local_assembler_.size() + m;
													 const auto wi = global_i[ii].val;

													 for (size_t jj = 0; jj < global_j.size(); ++jj)
													 {
														 const auto gj = global_j[jj].index * local_assembler_.size() + n;
														 const auto wj = global_j[jj].val;

														 loc_storage.cache.add_value(gi, gj, local_value * wi * wj);
														 // if (j < i) {
														 // 	loc_storage.entries.emplace_back(gj, gi, local_value * wj * wi);
														 // }

														 if (loc_storage.cache.entries_size() >= 1e8)
														 {
															 loc_storage.cache.prune();
															 logger().debug("cleaning memory...");
														 }
													 }
												 }
											 }
										 }
									 }
								 }
#if defined(POLYFEM_WITH_CPP_THREADS) || defined(POLYFEM_WITH_TBB)
							 }
#if defined(POLYFEM_WITH_CPP_THREADS)
							 loc_storage.cache.prune();
#endif
						 });
#else
						}
#endif

		timerg.stop();
		logger().trace("done separate assembly {}s...", timerg.getElapsedTime());

		timerg.start();

#if defined(POLYFEM_WITH_CPP_THREADS)
		for (const auto &t : storages)
		{
			mat_cache += t.cache;
		}

#elif defined(POLYFEM_WITH_TBB)
						merge_matrices(storages, mat_cache);
#else
						loc_storage.cache.prune();
						mat_cache += loc_storage.cache;
#endif

		grad = mat_cache.get_matrix();

		timerg.stop();
		logger().trace("done merge assembly {}s...", timerg.getElapsedTime());
	}

	template <class LocalAssembler>
	double NLAssembler<LocalAssembler>::assemble(
		const bool is_volume,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const Eigen::MatrixXd &displacement) const
	{
#if defined(POLYFEM_WITH_CPP_THREADS)
		std::vector<LocalThreadScalarStorage> storages(polyfem::get_n_threads());
#elif defined(POLYFEM_WITH_TBB)
						typedef tbb::enumerable_thread_specific<LocalThreadScalarStorage> LocalStorage;
						LocalStorage storages((LocalThreadScalarStorage()));
#else
						LocalThreadScalarStorage loc_storage;
#endif
		const int n_bases = int(bases.size());

#if defined(POLYFEM_WITH_CPP_THREADS)
		polyfem::par_for(n_bases, [&](int start, int end, int t)
						 {
							 auto &loc_storage = storages[t];
							 for (int e = start; e < end; ++e)
							 {
#elif defined(POLYFEM_WITH_TBB)
		tbb::parallel_for(tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
							LocalStorage::reference loc_storage = storages.local();
							for (int e = r.begin(); e != r.end(); ++e)
							{
#else
						for (int e = 0; e < n_bases; ++e)
						{
#endif
								 // igl::Timer timer; timer.start();

								 ElementAssemblyValues &vals = loc_storage.vals;
								 cache.compute(e, is_volume, bases[e], gbases[e], vals);

								 const Quadrature &quadrature = vals.quadrature;

								 assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
								 loc_storage.da = vals.det.array() * quadrature.weights.array();

								 const double val = local_assembler_.compute_energy(vals, displacement, loc_storage.da);
								 loc_storage.val += val;
#if defined(POLYFEM_WITH_CPP_THREADS) || defined(POLYFEM_WITH_TBB)
							 }
						 });
#else
							}
#endif

#if defined(POLYFEM_WITH_CPP_THREADS)
		double res = 0;
		for (const auto &t : storages)
		{
			res += t.val;
		}

		return res;
#elif defined(POLYFEM_WITH_TBB)
							double res = 0;
							for (LocalStorage::const_iterator i = storages.begin(); i != storages.end(); ++i)
							{
								res += i->val;
							}

							return res;
#else
							return loc_storage.val;
#endif
	}

	//template instantiation
	template class Assembler<Laplacian>;
	template class Assembler<Helmholtz>;

	template class Assembler<BilaplacianMain>;
	template class MixedAssembler<BilaplacianMixed>;
	template class Assembler<BilaplacianAux>;

	template class Assembler<LinearElasticity>;
	template class NLAssembler<LinearElasticity>;
	template class Assembler<HookeLinearElasticity>;
	template class NLAssembler<SaintVenantElasticity>;
	template class NLAssembler<NeoHookeanElasticity>;
	template class NLAssembler<MultiModel>;
	// template class NLAssembler<OgdenElasticity>;

	template class Assembler<StokesVelocity>;
	template class MixedAssembler<StokesMixed>;
	template class Assembler<StokesPressure>;

	template class NLAssembler<NavierStokesVelocity<true>>;
	template class NLAssembler<NavierStokesVelocity<false>>;

	template class Assembler<IncompressibleLinearElasticityDispacement>;
	template class MixedAssembler<IncompressibleLinearElasticityMixed>;
	template class Assembler<IncompressibleLinearElasticityPressure>;
} // namespace polyfem
