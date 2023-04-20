#include "Assembler.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <igl/Timer.h>

#include <ipc/utils/eigen_ext.hpp>

namespace polyfem::assembler
{
	using namespace basis;
	using namespace quadrature;
	using namespace utils;

	namespace
	{
		class LocalThreadMatStorage
		{
		public:
			SparseMatrixCache cache;
			ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadMatStorage()
			{
			}

			LocalThreadMatStorage(const int buffer_size, const int rows, const int cols)
			{
				init(buffer_size, rows, cols);
			}

			LocalThreadMatStorage(const int buffer_size, const SparseMatrixCache &c)
			{
				init(buffer_size, c);
			}

			void init(const int buffer_size, const int rows, const int cols)
			{
				// assert(rows == cols);
				cache.reserve(buffer_size);
				cache.init(rows, cols);
			}

			void init(const int buffer_size, const SparseMatrixCache &c)
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

	void Assembler::set_materials(const std::vector<int> &body_ids, const json &body_params, const Units &units)
	{
		if (!body_params.is_array())
		{
			this->add_multimaterial(0, body_params, units);
			return;
		}

		std::map<int, json> materials;
		for (int i = 0; i < body_params.size(); ++i)
		{
			json mat = body_params[i];
			json id = mat["id"];
			if (id.is_array())
			{
				for (int j = 0; j < id.size(); ++j)
					materials[id[j]] = mat;
			}
			else
			{
				const int mid = id;
				materials[mid] = mat;
			}
		}

		std::set<int> missing;

		std::map<int, int> body_element_count;
		std::vector<int> eid_to_eid_in_body(body_ids.size());
		for (int e = 0; e < body_ids.size(); ++e)
		{
			const int bid = body_ids[e];
			body_element_count.try_emplace(bid, 0);
			eid_to_eid_in_body[e] = body_element_count[bid]++;
		}

		for (int e = 0; e < body_ids.size(); ++e)
		{
			const int bid = body_ids[e];
			const auto it = materials.find(bid);
			if (it == materials.end())
			{
				missing.insert(bid);
				continue;
			}

			const json &tmp = it->second;
			this->add_multimaterial(e, tmp, units);
		}

		for (int bid : missing)
		{
			logger().warn("Missing material parameters for body {}", bid);
		}
	}

	LinearAssembler::LinearAssembler()
	{
	}

	void LinearAssembler::assemble(
		const bool is_volume,
		const int n_basis,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		StiffnessMatrix &stiffness,
		const bool is_mass) const
	{
		assert(size() > 0);

		const int max_triplets_size = int(1e7);
		const int buffer_size = std::min(long(max_triplets_size), long(n_basis) * size());
		// #ifdef POLYFEM_WITH_TBB
		// 		buffer_size /= tbb::task_scheduler_init::default_num_threads();
		// #endif
		logger().trace("buffer_size {}", buffer_size);
		try
		{
			stiffness.resize(n_basis * size(), n_basis * size());
			stiffness.setZero();

			auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));

			const int n_bases = int(bases.size());
			igl::Timer timerg;
			timerg.start();
			assert(cache.is_mass() == is_mass);

			maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
				LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);

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

							const auto stiffness_val = assemble(LinearAssemblerData(vals, i, j, local_storage.da));
							assert(stiffness_val.size() == size() * size());

							// igl::Timer t1; t1.start();
							for (int n = 0; n < size(); ++n)
							{
								for (int m = 0; m < size(); ++m)
								{
									const double local_value = stiffness_val(n * size() + m);
									if (std::abs(local_value) < 1e-30)
									{
										continue;
									}

									for (size_t ii = 0; ii < global_i.size(); ++ii)
									{
										const auto gi = global_i[ii].index * size() + m;
										const auto wi = global_i[ii].val;

										for (size_t jj = 0; jj < global_j.size(); ++jj)
										{
											const auto gj = global_j[jj].index * size() + n;
											const auto wj = global_j[jj].val;

											local_storage.cache.add_value(e, gi, gj, local_value * wi * wj);
											if (j < i)
											{
												local_storage.cache.add_value(e, gj, gi, local_value * wj * wi);
											}

											if (local_storage.cache.entries_size() >= max_triplets_size)
											{
												local_storage.cache.prune();
												logger().trace("cleaning memory. Current storage: {}. mat nnz: {}", local_storage.cache.capacity(), local_storage.cache.non_zeros());
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
			logger().trace("done separate assembly {}s...", timerg.getElapsedTime());

			// Assemble the stiffness matrix by concatenating the tuples in each local storage
			igl::Timer timer1, timer2, timer3;

			// Collect thread storages
			std::vector<LocalThreadMatStorage *> storages(storage.size());
			int index = 0;
			for (auto &local_storage : storage)
			{
				storages[index] = &local_storage;
				++index;
			}

			timerg.start();
			maybe_parallel_for(storages.size(), [&](int i) {
				auto *s = storages[i];
				s->cache.prune();
			});
			timerg.stop();
			logger().trace("done pruning triplets {}s...", timerg.getElapsedTime());

			// Prepares for parallel concatenation
			std::vector<int> offsets(storage.size());

			index = 0;
			int triplet_count = 0;
			for (auto &local_storage : storage)
			{
				offsets[index] = triplet_count;
				++index;
				triplet_count += local_storage.cache.entries().size();
				triplet_count += local_storage.cache.mat().nonZeros();
			}

			std::vector<Eigen::Triplet<double>> triplets;

			if (triplet_count >= triplets.max_size())
			{
				// Serial fallback version in case the vector of triplets cannot be allocated

				logger().warn("Cannot allocate space for triplets, switching to serial assembly.");

				timerg.start();
				// Serially merge local storages
				for (LocalThreadMatStorage &local_storage : storage)
					stiffness += local_storage.cache.get_matrix(false); // will also prune
				stiffness.makeCompressed();
				timerg.stop();

				logger().trace("Serial assembly time: {}s...", timerg.getElapsedTime());
			}
			else
			{
				timer1.start();
				triplets.resize(triplet_count);
				timer1.stop();

				logger().trace("done allocate triplets {}s...", timer1.getElapsedTime());
				logger().trace("Triplets Count: {}", triplet_count);

				timer2.start();
				// Parallel copy into triplets
				maybe_parallel_for(storages.size(), [&](int i) {
					const auto *s = storages[i];
					const int offset = offsets[i];
					for (int j = 0; j < s->cache.entries().size(); ++j)
					{
						triplets[offset + j] = s->cache.entries()[j];
					}
					if (s->cache.mat().nonZeros() > 0)
					{
						int count = 0;
						for (int k = 0; k < s->cache.mat().outerSize(); ++k)
						{
							for (Eigen::SparseMatrix<double>::InnerIterator it(s->cache.mat(), k); it; ++it)
							{
								assert(count < s->cache.mat().nonZeros());
								triplets[offset + s->cache.entries().size() + count++] = Eigen::Triplet<double>(it.row(), it.col(), it.value());
							}
						}
					}
				});

				timer2.stop();
				logger().trace("done concatenate triplets {}s...", timer2.getElapsedTime());

				timer3.start();
				// Sort and assemble
				stiffness.setFromTriplets(triplets.begin(), triplets.end());
				timer3.stop();

				logger().trace("done setFromTriplets assembly {}s...", timer3.getElapsedTime());
			}

			// exit(0);
		}
		catch (std::bad_alloc &ba)
		{
			logger().error("bad alloc {}", ba.what());
			exit(0);
		}

		// stiffness.resize(n_basis*size(), n_basis*size());
		// stiffness.setFromTriplets(entries.begin(), entries.end());
	}

	MixedAssembler::MixedAssembler()
	{
	}

	void MixedAssembler::assemble(
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
		assert(size() > 0);
		assert(phi_bases.size() == psi_bases.size());

		const int max_triplets_size = int(1e7);
		const int buffer_size = std::min(long(max_triplets_size), long(std::max(n_psi_basis, n_phi_basis)) * std::max(rows(), cols()));
		logger().debug("buffer_size {}", buffer_size);

		stiffness.resize(n_phi_basis * rows(), n_psi_basis * cols());
		stiffness.setZero();

		auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));

		const int n_bases = int(phi_bases.size());
		igl::Timer timerg;
		timerg.start();

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);
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

						const auto stiffness_val = assemble(MixedAssemblerData(psi_vals, phi_vals, i, j, local_storage.da));
						assert(stiffness_val.size() == rows() * cols());

						// igl::Timer t1; t1.start();
						for (int n = 0; n < rows(); ++n)
						{
							for (int m = 0; m < cols(); ++m)
							{
								const double local_value = stiffness_val(n * cols() + m);
								if (std::abs(local_value) < 1e-30)
								{
									continue;
								}

								for (size_t ii = 0; ii < global_i.size(); ++ii)
								{
									const auto gi = global_i[ii].index * cols() + m;
									const auto wi = global_i[ii].val;

									for (size_t jj = 0; jj < global_j.size(); ++jj)
									{
										const auto gj = global_j[jj].index * rows() + n;
										const auto wj = global_j[jj].val;

										local_storage.cache.add_value(e, gj, gi, local_value * wi * wj);

										if (local_storage.cache.entries_size() >= max_triplets_size)
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

		// stiffness.resize(n_basis*size(), n_basis*size());
		// stiffness.setFromTriplets(entries.begin(), entries.end());
	}

	double NLAssembler::assemble_energy(
		const bool is_volume,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const double dt,
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &displacement_prev) const
	{
		auto storage = create_thread_storage(LocalThreadScalarStorage());
		const int n_bases = int(bases.size());

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadScalarStorage &local_storage = get_local_thread_storage(storage, thread_id);
			ElementAssemblyValues &vals = local_storage.vals;

			for (int e = start; e < end; ++e)
			{
				cache.compute(e, is_volume, bases[e], gbases[e], vals);

				const Quadrature &quadrature = vals.quadrature;

				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = vals.det.array() * quadrature.weights.array();

				const double val = compute_energy(NonLinearAssemblerData(vals, dt, displacement, displacement_prev, local_storage.da));
				local_storage.val += val;
			}
		});

		double res = 0;
		// Serially merge local storages
		for (const LocalThreadScalarStorage &local_storage : storage)
			res += local_storage.val;
		return res;
	}

	void NLAssembler::assemble_gradient(
		const bool is_volume,
		const int n_basis,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const double dt,
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &displacement_prev,
		Eigen::MatrixXd &rhs) const
	{
		rhs.resize(n_basis * size(), 1);
		rhs.setZero();

		auto storage = create_thread_storage(LocalThreadVecStorage(rhs.size()));

		const int n_bases = int(bases.size());

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = get_local_thread_storage(storage, thread_id);

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

				const auto val = assemble_gradient(NonLinearAssemblerData(vals, dt, displacement, displacement_prev, local_storage.da));
				assert(val.size() == n_loc_bases * size());

				for (int j = 0; j < n_loc_bases; ++j)
				{
					const auto &global_j = vals.basis_values[j].global;

					// igl::Timer t1; t1.start();
					for (int m = 0; m < size(); ++m)
					{
						const double local_value = val(j * size() + m);
						if (std::abs(local_value) < 1e-30)
						{
							continue;
						}

						for (size_t jj = 0; jj < global_j.size(); ++jj)
						{
							const auto gj = global_j[jj].index * size() + m;
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

	void NLAssembler::assemble_hessian(
		const bool is_volume,
		const int n_basis,
		const bool project_to_psd,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const double dt,
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &displacement_prev,
		SparseMatrixCache &mat_cache,
		StiffnessMatrix &grad) const
	{
		const int max_triplets_size = int(1e7);
		const int buffer_size = std::min(long(max_triplets_size), long(n_basis) * size());
		// std::cout<<"buffer_size "<<buffer_size<<std::endl;

		// grad.resize(n_basis * size(), n_basis * size());
		// grad.setZero();

		mat_cache.init(n_basis * size());
		mat_cache.set_zero();

		auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, mat_cache));

		const int n_bases = int(bases.size());
		igl::Timer timerg;
		timerg.start();

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);

			for (int e = start; e < end; ++e)
			{
				ElementAssemblyValues &vals = local_storage.vals;
				cache.compute(e, is_volume, bases[e], gbases[e], vals);

				const Quadrature &quadrature = vals.quadrature;

				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = vals.det.array() * quadrature.weights.array();
				const int n_loc_bases = int(vals.basis_values.size());

				auto stiffness_val = assemble_hessian(NonLinearAssemblerData(vals, dt, displacement, displacement_prev, local_storage.da));
				assert(stiffness_val.rows() == n_loc_bases * size());
				assert(stiffness_val.cols() == n_loc_bases * size());

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

						for (int n = 0; n < size(); ++n)
						{
							for (int m = 0; m < size(); ++m)
							{
								const double local_value = stiffness_val(i * size() + m, j * size() + n);
								//  if (std::abs(local_value) < 1e-30)
								//  {
								// 	 continue;
								//  }

								for (size_t ii = 0; ii < global_i.size(); ++ii)
								{
									const auto gi = global_i[ii].index * size() + m;
									const auto wi = global_i[ii].val;

									for (size_t jj = 0; jj < global_j.size(); ++jj)
									{
										const auto gj = global_j[jj].index * size() + n;
										const auto wj = global_j[jj].val;

										local_storage.cache.add_value(e, gi, gj, local_value * wi * wj);
										// if (j < i) {
										// 	local_storage.entries.emplace_back(gj, gi, local_value * wj * wi);
										// }

										if (local_storage.cache.entries_size() >= max_triplets_size)
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

} // namespace polyfem::assembler
