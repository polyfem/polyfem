#include <polyfem/Assembler.hpp>

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
#include <polyfem/MaybeParallelFor.hpp>

#include <igl/Timer.h>

#include <ipc/utils/eigen_ext.hpp>

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
				// assert(rows == cols);
				cache.reserve(buffer_size);
				cache.init(rows, cols);
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

			auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));

			const int n_bases = int(bases.size());
			igl::Timer timerg;
			timerg.start();

			maybe_parallel_for(n_bases, [&](int start, int end) {
				LocalThreadMatStorage &local_storage = get_local_thread_storage(storage);

				for (int e = start; e < end; ++e)
				{
					ElementAssemblyValues &vals = local_storage.vals;
					// igl::Timer timer; timer.start();
					// vals.compute(e, is_volume, bases[e], gbases[e]);
					cache.compute(e, is_volume, bases[e], gbases[e], vals);

					const Quadrature &quadrature = vals.quadrature;

					assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
					local_storage.da = vals.det.array() * quadrature.weights.array();
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

							const auto stiffness_val = local_assembler_.assemble(vals, i, j, local_storage.da);
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

											local_storage.cache.add_value(gi, gj, local_value * wi * wj);
											if (j < i)
											{
												local_storage.cache.add_value(gj, gi, local_value * wj * wi);
											}

											if (local_storage.cache.entries_size() >= 1e8)
											{
												local_storage.cache.prune();
												logger().debug("cleaning memory. Current storage: {}. mat nnz: {}", local_storage.cache.capacity(), local_storage.cache.non_zeros());
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
				}
			});

			timerg.stop();
			logger().debug("done separate assembly {}s...", timerg.getElapsedTime());

			timerg.start();
			// Serially merge local storages
			for (LocalThreadMatStorage &local_storage : storage)
				stiffness += local_storage.cache.get_matrix(false); // will also prune
			stiffness.makeCompressed();
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

		auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));

		const int n_bases = int(phi_bases.size());
		igl::Timer timerg;
		timerg.start();

		maybe_parallel_for(n_bases, [&](int start, int end) {
			LocalThreadMatStorage &local_storage = get_local_thread_storage(storage);
			ElementAssemblyValues psi_vals, phi_vals;

			for (int e = start; e < end; ++e)
			{
				// psi_vals.compute(e, is_volume, psi_bases[e], gbases[e]);
				// phi_vals.compute(e, is_volume, phi_bases[e], gbases[e]);
				psi_cache.compute(e, is_volume, psi_bases[e], gbases[e], psi_vals);
				phi_cache.compute(e, is_volume, phi_bases[e], gbases[e], phi_vals);

				const Quadrature &quadrature = phi_vals.quadrature;

				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = phi_vals.det.array() * quadrature.weights.array();
				const int n_phi_loc_bases = int(phi_vals.basis_values.size());
				const int n_psi_loc_bases = int(psi_vals.basis_values.size());

				for (int i = 0; i < n_psi_loc_bases; ++i)
				{
					const auto &global_i = psi_vals.basis_values[i].global;

					for (int j = 0; j < n_phi_loc_bases; ++j)
					{
						const auto &global_j = phi_vals.basis_values[j].global;

						const auto stiffness_val = local_assembler_.assemble(psi_vals, phi_vals, i, j, local_storage.da);
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

										local_storage.cache.add_value(gj, gi, local_value * wi * wj);

										if (local_storage.cache.entries_size() >= 1e8)
										{
											local_storage.cache.prune();
											logger().debug("cleaning memory...");
										}
									}
								}
							}
						}
					}
				}
			}
		});

		timerg.stop();
		logger().trace("done separate assembly {}s...", timerg.getElapsedTime());

		timerg.start();
		// Serially merge local storages
		for (LocalThreadMatStorage &local_storage : storage)
			stiffness += local_storage.cache.get_matrix(false); // will also prune
		stiffness.makeCompressed();
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

		auto storage = create_thread_storage(LocalThreadVecStorage(rhs.size()));

		const int n_bases = int(bases.size());

		maybe_parallel_for(n_bases, [&](int start, int end) {
			LocalThreadVecStorage &local_storage = get_local_thread_storage(storage);

			for (int e = start; e < end; ++e)
			{
				// igl::Timer timer; timer.start();

				ElementAssemblyValues &vals = local_storage.vals;
				// vals.compute(e, is_volume, bases[e], gbases[e]);
				cache.compute(e, is_volume, bases[e], gbases[e], vals);

				const Quadrature &quadrature = vals.quadrature;

				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = vals.det.array() * quadrature.weights.array();
				const int n_loc_bases = int(vals.basis_values.size());

				const auto val = local_assembler_.assemble_grad(vals, displacement, local_storage.da);
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

							local_storage.vec(gj) += local_value * wj;
						}
					}

					// t1.stop();
					// if (!vals.has_parameterization) { std::cout << "-- t1: " << t1.getElapsedTime() << std::endl; }
				}

				// timer.stop();
				// if (!vals.has_parameterization) { std::cout << "-- Timer: " << timer.getElapsedTime() << std::endl; }
			}
		});

		// Serially merge local storages
		for (const LocalThreadVecStorage &local_storage : storage)
			rhs += local_storage.vec;
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

		auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, mat_cache));

		const int n_bases = int(bases.size());
		igl::Timer timerg;
		timerg.start();

		maybe_parallel_for(n_bases, [&](int start, int end) {
			LocalThreadMatStorage &local_storage = get_local_thread_storage(storage);

			for (int e = start; e < end; ++e)
			{
				ElementAssemblyValues &vals = local_storage.vals;
				cache.compute(e, is_volume, bases[e], gbases[e], vals);

				const Quadrature &quadrature = vals.quadrature;

				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = vals.det.array() * quadrature.weights.array();
				const int n_loc_bases = int(vals.basis_values.size());

				auto stiffness_val = local_assembler_.assemble_hessian(vals, displacement, local_storage.da);
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
				// 	local_storage.entries.emplace_back(0, 0, std::nan(""));
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

										local_storage.cache.add_value(gi, gj, local_value * wi * wj);
										// if (j < i) {
										// 	local_storage.entries.emplace_back(gj, gi, local_value * wj * wi);
										// }

										if (local_storage.cache.entries_size() >= 1e8)
										{
											local_storage.cache.prune();
											logger().debug("cleaning memory...");
										}
									}
								}
							}
						}
					}
				}
			}
		});

		timerg.stop();
		logger().trace("done separate assembly {}s...", timerg.getElapsedTime());

		timerg.start();

		// Serially merge local storages
		for (LocalThreadMatStorage &local_storage : storage)
		{
			local_storage.cache.prune();
			mat_cache += local_storage.cache;
		}
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
		auto storage = create_thread_storage(LocalThreadScalarStorage());
		const int n_bases = int(bases.size());

		maybe_parallel_for(n_bases, [&](int start, int end) {
			// igl::Timer timer; timer.start();
			LocalThreadScalarStorage &local_storage = get_local_thread_storage(storage);
			ElementAssemblyValues &vals = local_storage.vals;

			for (int e = start; e < end; ++e)
			{
				ElementAssemblyValues &vals = local_storage.vals;
				cache.compute(e, is_volume, bases[e], gbases[e], vals);

				const Quadrature &quadrature = vals.quadrature;

				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = vals.det.array() * quadrature.weights.array();

				const double val = local_assembler_.compute_energy(vals, displacement, local_storage.da);
				local_storage.val += val;
			}
		});

		double res = 0;
		// Serially merge local storages
		for (const LocalThreadScalarStorage &local_storage : storage)
			res += local_storage.val;
		return res;
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
