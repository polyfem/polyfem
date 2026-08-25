#include "Assembler.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <igl/Timer.h>

#include <ipc/utils/eigen_ext.hpp>

#include <algorithm>
#include <cmath>

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
			std::unique_ptr<MatrixCache> cache = nullptr;
			ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadMatStorage() = delete;

			LocalThreadMatStorage(const int buffer_size, const int rows, const int cols)
			{
				init(buffer_size, rows, cols);
			}

			LocalThreadMatStorage(const int buffer_size, const MatrixCache &c)
			{
				init(buffer_size, c);
			}

			LocalThreadMatStorage(const LocalThreadMatStorage &other)
				: cache(other.cache->copy()), vals(other.vals), da(other.da)
			{
			}

			LocalThreadMatStorage &operator=(const LocalThreadMatStorage &other)
			{
				assert(other.cache != nullptr);
				cache = other.cache->copy();
				vals = other.vals;
				da = other.da;
				return *this;
			}

			void init(const int buffer_size, const int rows, const int cols)
			{
				// assert(rows == cols);
				// cache = std::make_unique<DenseMatrixCache>();
				cache = std::make_unique<SparseMatrixCache>();
				cache->reserve(buffer_size);
				cache->init(rows, cols);
			}

			void init(const int buffer_size, const MatrixCache &c)
			{
				if (cache == nullptr)
					cache = c.copy();
				cache->reserve(buffer_size);
				cache->init(c);
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

		void split_mixed_solution(
			const MixedNLAssembler::SolutionSplitter &split_solution,
			const Eigen::MatrixXd &x,
			const int phi_ndof,
			const int psi_ndof,
			Eigen::MatrixXd &x_phi,
			Eigen::MatrixXd &x_psi)
		{
			assert(split_solution);

			split_solution(x, x_phi, x_psi);

			assert(x_phi.rows() == phi_ndof);
			assert(x_phi.cols() == 1);
			assert(x_psi.rows() == psi_ndof);
			assert(x_psi.cols() == 1);
		}

		void split_mixed_previous_solution(
			const MixedNLAssembler::SolutionSplitter &split_solution,
			const Eigen::MatrixXd &x,
			const int phi_ndof,
			const int psi_ndof,
			Eigen::MatrixXd &x_phi,
			Eigen::MatrixXd &x_psi)
		{
			if (x.size() == 0)
			{
				x_phi.resize(0, 1);
				x_psi.resize(0, 1);
				return;
			}

			split_mixed_solution(split_solution, x, phi_ndof, psi_ndof, x_phi, x_psi);
		}

		void compute_mixed_element_values(
			const int el_index,
			const bool is_volume,
			const ElementBases &psi_basis,
			const ElementBases &phi_basis,
			const ElementBases &gbasis,
			const AssemblyValsCache &psi_cache,
			const AssemblyValsCache &phi_cache,
			ElementAssemblyValues &psi_vals,
			ElementAssemblyValues &phi_vals)
		{
			psi_cache.compute(el_index, is_volume, psi_basis, gbasis, psi_vals);
			phi_cache.compute(el_index, is_volume, phi_basis, gbasis, phi_vals);

			const auto same_quadrature = [&]() {
				if (psi_vals.quadrature.weights.size() != phi_vals.quadrature.weights.size())
					return false;
				if (psi_vals.quadrature.points.rows() != phi_vals.quadrature.points.rows()
					|| psi_vals.quadrature.points.cols() != phi_vals.quadrature.points.cols())
					return false;
				return (psi_vals.quadrature.points - phi_vals.quadrature.points).cwiseAbs().maxCoeff() < 1e-14;
			};

			if (same_quadrature())
				return;

			if (phi_vals.quadrature.weights.size() >= psi_vals.quadrature.weights.size())
			{
				const Quadrature quadrature = phi_vals.quadrature;
				psi_vals.compute(el_index, is_volume, quadrature.points, psi_basis, gbasis);
				psi_vals.quadrature = quadrature;
			}
			else
			{
				const Quadrature quadrature = psi_vals.quadrature;
				phi_vals.compute(el_index, is_volume, quadrature.points, phi_basis, gbasis);
				phi_vals.quadrature = quadrature;
			}

			assert(psi_vals.quadrature.weights.size() == phi_vals.quadrature.weights.size());
			assert(psi_vals.quadrature.points.rows() == phi_vals.quadrature.points.rows());
		}
	} // namespace

	void Assembler::set_materials(const std::vector<int> &body_ids, const json &body_params, const Units &units, const std::string &root_path)
	{
		if (!body_params.is_array())
		{
			this->add_multimaterial(0, body_params, units, root_path);
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
			this->add_multimaterial(e, tmp, units, root_path);
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
		const double t,
		StiffnessMatrix &stiffness,
		const bool is_mass) const
	{
		assert(size() > 0);

		const long int max_triplets_size = long(1e7);
		const long int buffer_size = std::min(long(max_triplets_size), long(n_basis) * size());
		// #ifdef POLYFEM_WITH_TBB
		// 		buffer_size /= tbb::task_scheduler_init::default_num_threads();
		// #endif
		// logger().trace("buffer_size {}", buffer_size);
		try
		{
			stiffness.resize(n_basis * size(), n_basis * size());
			stiffness.setZero();

			auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));

			const int n_bases = int(bases.size());
			igl::Timer timer;
			timer.start();
			assert(cache.is_mass() == is_mass);

			// (potentially parallel) loop over elements
			// Note that n_bases is the number of elements since ach ElementBases object stores
			// all local basis functions on a given element
			maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
				LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					ElementAssemblyValues &vals = local_storage.vals;
					// igl::Timer timer; timer.start();
					// vals.compute(e, is_volume, bases[e], gbases[e]);

					// compute geometric mapping
					// evaluate and store basis functions/their gradients at quadrature points
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

						// loop over other bases up to the current one, taking advantage of symmetry
						for (int j = 0; j <= i; ++j)
						{
							// const AssemblyValues &values_j = vals.basis_values[j];
							// const Eigen::MatrixXd &gradj = values_j.grad_t_m;
							const auto &global_j = vals.basis_values[j].global;

							// compute local entry in stiffness matrix
							const auto stiffness_val = assemble(LinearAssemblerData(vals, t, i, j, local_storage.da));
							assert(stiffness_val.size() == size() * size());

							// igl::Timer t1; t1.start();
							// loop over dimensions of the problem
							for (int n = 0; n < size(); ++n)
							{
								for (int m = 0; m < size(); ++m)
								{
									const double local_value = stiffness_val(n * size() + m);

									// loop over the global nodes corresponding to local element (useful for non-conforming cases)
									for (size_t ii = 0; ii < global_i.size(); ++ii)
									{
										const auto gi = global_i[ii].index * size() + m;
										const auto wi = global_i[ii].val;

										for (size_t jj = 0; jj < global_j.size(); ++jj)
										{
											const auto gj = global_j[jj].index * size() + n;
											const auto wj = global_j[jj].val;

											// add local value to the global matrix (weighted by corresponding nodes)
											local_storage.cache->add_value(e, gi, gj, local_value * wi * wj);
											if (j < i)
											{
												local_storage.cache->add_value(e, gj, gi, local_value * wj * wi);
											}

											if (local_storage.cache->entries_size() >= max_triplets_size)
											{
												local_storage.cache->prune();
												logger().trace("cleaning memory. Current storage: {}. mat nnz: {}", local_storage.cache->capacity(), local_storage.cache->non_zeros());
											}
										}
									}
								}
							}

							// t1.stop();
							// if (!vals.has_parameterization) { logger().trace("-- t1: " {}, t1.getElapsedTime()); }
						}
					}

					// timer.stop();
					// if (!vals.has_parameterization) { logger().trace("-- Timer: {}", timer.getElapsedTime()); }
				}
			});

			timer.stop();
			logger().trace("done separate assembly {}s...", timer.getElapsedTime());

			// Assemble the stiffness matrix by concatenating the tuples in each local storage

			// Collect thread storages
			std::vector<LocalThreadMatStorage *> storages(storage.size());
			long int index = 0;
			for (auto &local_storage : storage)
			{
				storages[index++] = &local_storage;
			}

			timer.start();
			maybe_parallel_for(storages.size(), [&](int i) {
				storages[i]->cache->prune();
			});
			timer.stop();
			logger().trace("done pruning triplets {}s...", timer.getElapsedTime());

			// Prepares for parallel concatenation
			std::vector<long int> offsets(storage.size());

			index = 0;
			long int triplet_count = 0;
			for (auto &local_storage : storage)
			{
				offsets[index++] = triplet_count;
				triplet_count += local_storage.cache->triplet_count();
			}

			std::vector<Eigen::Triplet<double>> triplets;

			assert(storages.size() >= 1);
			if (storages[0]->cache->is_dense())
			{
				timer.start();
				// Serially merge local storages
				Eigen::MatrixXd tmp(stiffness);
				for (const LocalThreadMatStorage &local_storage : storage)
					tmp += dynamic_cast<const DenseMatrixCache &>(*local_storage.cache).mat();
				stiffness = tmp.sparseView();
				stiffness.makeCompressed();
				timer.stop();

				logger().trace("Serial assembly time: {}s...", timer.getElapsedTime());
			}
			else if (triplet_count >= triplets.max_size())
			{
				// Serial fallback version in case the vector of triplets cannot be allocated

				logger().warn("Cannot allocate space for triplets, switching to serial assembly.");

				timer.start();
				// Serially merge local storages
				for (LocalThreadMatStorage &local_storage : storage)
					stiffness += local_storage.cache->get_matrix(false); // will also prune
				stiffness.makeCompressed();
				timer.stop();

				logger().trace("Serial assembly time: {}s...", timer.getElapsedTime());
			}
			else
			{
				timer.start();
				triplets.resize(triplet_count);
				timer.stop();

				logger().trace("done allocate triplets {}s...", timer.getElapsedTime());
				logger().trace("Triplets Count: {}", triplet_count);

				timer.start();
				// Parallel copy into triplets
				maybe_parallel_for(storages.size(), [&](int i) {
					const SparseMatrixCache &cache = dynamic_cast<const SparseMatrixCache &>(*storages[i]->cache);
					long int offset = offsets[i];

					std::copy(cache.entries().begin(), cache.entries().end(), triplets.begin() + offset);
					offset += cache.entries().size();

					if (cache.mat().nonZeros() > 0)
					{
						long int count = 0;
						for (int k = 0; k < cache.mat().outerSize(); ++k)
						{
							for (Eigen::SparseMatrix<double>::InnerIterator it(cache.mat(), k); it; ++it)
							{
								assert(count < cache.mat().nonZeros());
								triplets[offset + count++] = Eigen::Triplet<double>(it.row(), it.col(), it.value());
							}
						}
					}
				});

				timer.stop();
				logger().trace("done concatenate triplets {}s...", timer.getElapsedTime());

				timer.start();
				// Sort and assemble
				stiffness.setFromTriplets(triplets.begin(), triplets.end());
				timer.stop();

				logger().trace("done setFromTriplets assembly {}s...", timer.getElapsedTime());
			}
		}
		catch (std::bad_alloc &ba)
		{
			log_and_throw_error("bad alloc {}", ba.what());
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
		const double t,
		StiffnessMatrix &stiffness) const
	{
		assert(size() > 0);
		assert(phi_bases.size() == psi_bases.size());

		const int max_triplets_size = int(1e7);
		const int buffer_size = std::min(long(max_triplets_size), long(std::max(n_psi_basis, n_phi_basis)) * std::max(rows(), cols()));
		// logger().debug("buffer_size {}", buffer_size);

		stiffness.resize(n_phi_basis * rows(), n_psi_basis * cols());
		stiffness.setZero();

		auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));

		const int n_bases = int(phi_bases.size());
		igl::Timer timer;
		timer.start();

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

						const auto stiffness_val = assemble(MixedAssemblerData(psi_vals, phi_vals, t, i, j, local_storage.da));
						assert(stiffness_val.size() == rows() * cols());

						// igl::Timer t1; t1.start();
						for (int n = 0; n < rows(); ++n)
						{
							for (int m = 0; m < cols(); ++m)
							{
								const double local_value = stiffness_val(n * cols() + m);

								for (size_t ii = 0; ii < global_i.size(); ++ii)
								{
									const auto gi = global_i[ii].index * cols() + m;
									const auto wi = global_i[ii].val;

									for (size_t jj = 0; jj < global_j.size(); ++jj)
									{
										const auto gj = global_j[jj].index * rows() + n;
										const auto wj = global_j[jj].val;

										local_storage.cache->add_value(e, gj, gi, local_value * wi * wj);

										if (local_storage.cache->entries_size() >= max_triplets_size)
										{
											local_storage.cache->prune();
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

		timer.stop();
		logger().trace("done separate assembly {}s...", timer.getElapsedTime());

		timer.start();
		// Serially merge local storages
		for (LocalThreadMatStorage &local_storage : storage)
			stiffness += local_storage.cache->get_matrix(false); // will also prune
		stiffness.makeCompressed();
		timer.stop();
		logger().trace("done merge assembly {}s...", timer.getElapsedTime());

		// stiffness.resize(n_basis*size(), n_basis*size());
		// stiffness.setFromTriplets(entries.begin(), entries.end());
	}

	double MixedNLAssembler::assemble_energy(
		const bool is_volume,
		const int n_psi_basis,
		const int n_phi_basis,
		const std::vector<ElementBases> &psi_bases,
		const std::vector<ElementBases> &phi_bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &psi_cache,
		const AssemblyValsCache &phi_cache,
		const double t,
		const double dt,
		const Eigen::MatrixXd &x,
		const Eigen::MatrixXd &x_prev,
		const SolutionSplitter &split_solution) const
	{
		assert(size() > 0);
		assert(rows() > 0);
		assert(cols() > 0);
		assert(phi_bases.size() == psi_bases.size());
		assert(phi_bases.size() == gbases.size());

		const int phi_ndof = n_phi_basis * rows();
		const int psi_ndof = n_psi_basis * cols();
		Eigen::MatrixXd x_phi, x_psi;
		Eigen::MatrixXd x_phi_prev, x_psi_prev;
		split_mixed_solution(split_solution, x, phi_ndof, psi_ndof, x_phi, x_psi);
		split_mixed_previous_solution(split_solution, x_prev, phi_ndof, psi_ndof, x_phi_prev, x_psi_prev);

		auto storage = create_thread_storage(LocalThreadScalarStorage());
		const int n_bases = int(phi_bases.size());

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadScalarStorage &local_storage = get_local_thread_storage(storage, thread_id);
			ElementAssemblyValues &phi_vals = local_storage.vals;
			ElementAssemblyValues psi_vals;

			for (int e = start; e < end; ++e)
			{
				compute_mixed_element_values(
					e, is_volume,
					psi_bases[e], phi_bases[e], gbases[e],
					psi_cache, phi_cache,
					psi_vals, phi_vals);

				const Quadrature &quadrature = phi_vals.quadrature;
				assert(psi_vals.quadrature.weights.size() == quadrature.weights.size());
				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = phi_vals.det.array() * quadrature.weights.array();

				local_storage.val += compute_energy(
					MixedNonLinearAssemblerData(psi_vals, phi_vals, t, dt, x_phi, x_psi, x_phi_prev, x_psi_prev, local_storage.da));
			}
		});

		double res = 0;
		for (const LocalThreadScalarStorage &local_storage : storage)
			res += local_storage.val;
		return res;
	}

	Eigen::VectorXd MixedNLAssembler::assemble_energy_per_element(
		const bool is_volume,
		const int n_psi_basis,
		const int n_phi_basis,
		const std::vector<ElementBases> &psi_bases,
		const std::vector<ElementBases> &phi_bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &psi_cache,
		const AssemblyValsCache &phi_cache,
		const double t,
		const double dt,
		const Eigen::MatrixXd &x,
		const Eigen::MatrixXd &x_prev,
		const SolutionSplitter &split_solution) const
	{
		assert(size() > 0);
		assert(rows() > 0);
		assert(cols() > 0);
		assert(phi_bases.size() == psi_bases.size());
		assert(phi_bases.size() == gbases.size());

		const int phi_ndof = n_phi_basis * rows();
		const int psi_ndof = n_psi_basis * cols();
		Eigen::MatrixXd x_phi, x_psi;
		Eigen::MatrixXd x_phi_prev, x_psi_prev;
		split_mixed_solution(split_solution, x, phi_ndof, psi_ndof, x_phi, x_psi);
		split_mixed_previous_solution(split_solution, x_prev, phi_ndof, psi_ndof, x_phi_prev, x_psi_prev);

		auto storage = create_thread_storage(LocalThreadScalarStorage());
		const int n_bases = int(phi_bases.size());
		Eigen::VectorXd out(phi_bases.size());

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadScalarStorage &local_storage = get_local_thread_storage(storage, thread_id);
			ElementAssemblyValues &phi_vals = local_storage.vals;
			ElementAssemblyValues psi_vals;

			for (int e = start; e < end; ++e)
			{
				compute_mixed_element_values(
					e, is_volume,
					psi_bases[e], phi_bases[e], gbases[e],
					psi_cache, phi_cache,
					psi_vals, phi_vals);

				const Quadrature &quadrature = phi_vals.quadrature;
				assert(psi_vals.quadrature.weights.size() == quadrature.weights.size());
				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = phi_vals.det.array() * quadrature.weights.array();

				out(e) = compute_energy(
					MixedNonLinearAssemblerData(psi_vals, phi_vals, t, dt, x_phi, x_psi, x_phi_prev, x_psi_prev, local_storage.da));
			}
		});

#ifndef NDEBUG
		const double assemble_val = assemble_energy(
			is_volume, n_psi_basis, n_phi_basis,
			psi_bases, phi_bases, gbases,
			psi_cache, phi_cache, t, dt, x, x_prev, split_solution);
		assert(std::abs(assemble_val - out.sum()) < std::max(1e-10 * std::abs(assemble_val), 1e-10));
#endif

		return out;
	}

	void MixedNLAssembler::assemble_gradient(
		const bool is_volume,
		const int n_psi_basis,
		const int n_phi_basis,
		const std::vector<ElementBases> &psi_bases,
		const std::vector<ElementBases> &phi_bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &psi_cache,
		const AssemblyValsCache &phi_cache,
		const double t,
		const double dt,
		const Eigen::MatrixXd &x,
		const Eigen::MatrixXd &x_prev,
		const SolutionSplitter &split_solution,
		Eigen::MatrixXd &grad) const
	{
		assert(size() > 0);
		assert(rows() > 0);
		assert(cols() > 0);
		assert(phi_bases.size() == psi_bases.size());
		assert(phi_bases.size() == gbases.size());

		const int phi_ndof = n_phi_basis * rows();
		const int psi_ndof = n_psi_basis * cols();
		Eigen::MatrixXd x_phi, x_psi;
		Eigen::MatrixXd x_phi_prev, x_psi_prev;
		split_mixed_solution(split_solution, x, phi_ndof, psi_ndof, x_phi, x_psi);
		split_mixed_previous_solution(split_solution, x_prev, phi_ndof, psi_ndof, x_phi_prev, x_psi_prev);

		grad.resize(phi_ndof + psi_ndof, 1);
		grad.setZero();

		auto storage = create_thread_storage(LocalThreadVecStorage(grad.size()));
		const int n_bases = int(phi_bases.size());

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadVecStorage &local_storage = get_local_thread_storage(storage, thread_id);
			ElementAssemblyValues &phi_vals = local_storage.vals;
			ElementAssemblyValues psi_vals;

			for (int e = start; e < end; ++e)
			{
				compute_mixed_element_values(
					e, is_volume,
					psi_bases[e], phi_bases[e], gbases[e],
					psi_cache, phi_cache,
					psi_vals, phi_vals);

				const Quadrature &quadrature = phi_vals.quadrature;
				assert(psi_vals.quadrature.weights.size() == quadrature.weights.size());
				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = phi_vals.det.array() * quadrature.weights.array();

				const Eigen::VectorXd local_grad = compute_gradient(
					MixedNonLinearAssemblerData(psi_vals, phi_vals, t, dt, x_phi, x_psi, x_phi_prev, x_psi_prev, local_storage.da));

				const int n_phi_loc_bases = int(phi_vals.basis_values.size());
				const int n_psi_loc_bases = int(psi_vals.basis_values.size());
				assert(local_grad.size() == n_phi_loc_bases * rows() + n_psi_loc_bases * cols());

				for (int i = 0; i < n_phi_loc_bases; ++i)
				{
					const auto &global_i = phi_vals.basis_values[i].global;
					for (int d = 0; d < rows(); ++d)
					{
						const double local_value = local_grad(i * rows() + d);
						for (const auto &global : global_i)
							local_storage.vec(global.index * rows() + d) += local_value * global.val;
					}
				}

				const int local_psi_offset = n_phi_loc_bases * rows();
				for (int i = 0; i < n_psi_loc_bases; ++i)
				{
					const auto &global_i = psi_vals.basis_values[i].global;
					for (int d = 0; d < cols(); ++d)
					{
						const double local_value = local_grad(local_psi_offset + i * cols() + d);
						for (const auto &global : global_i)
							local_storage.vec(phi_ndof + global.index * cols() + d) += local_value * global.val;
					}
				}
			}
		});

		for (const LocalThreadVecStorage &local_storage : storage)
			grad += local_storage.vec;
	}

	void MixedNLAssembler::assemble_hessian(
		const bool is_volume,
		const int n_psi_basis,
		const int n_phi_basis,
		const bool project_to_psd,
		const std::vector<ElementBases> &psi_bases,
		const std::vector<ElementBases> &phi_bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &psi_cache,
		const AssemblyValsCache &phi_cache,
		const double t,
		const double dt,
		const Eigen::MatrixXd &x,
		const Eigen::MatrixXd &x_prev,
		const SolutionSplitter &split_solution,
		MatrixCache &mat_cache,
		StiffnessMatrix &hessian) const
	{
		assert(size() > 0);
		assert(rows() > 0);
		assert(cols() > 0);
		assert(phi_bases.size() == psi_bases.size());
		assert(phi_bases.size() == gbases.size());

		const int phi_ndof = n_phi_basis * rows();
		const int psi_ndof = n_psi_basis * cols();
		const int total_ndof = phi_ndof + psi_ndof;
		Eigen::MatrixXd x_phi, x_psi;
		Eigen::MatrixXd x_phi_prev, x_psi_prev;
		split_mixed_solution(split_solution, x, phi_ndof, psi_ndof, x_phi, x_psi);
		split_mixed_previous_solution(split_solution, x_prev, phi_ndof, psi_ndof, x_phi_prev, x_psi_prev);

		const int max_triplets_size = int(1e7);
		const int buffer_size = std::min(max_triplets_size, total_ndof);

		mat_cache.init(total_ndof);
		mat_cache.set_zero();

		auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, mat_cache));
		const int n_bases = int(phi_bases.size());

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);
			ElementAssemblyValues &phi_vals = local_storage.vals;
			ElementAssemblyValues psi_vals;

			for (int e = start; e < end; ++e)
			{
				compute_mixed_element_values(
					e, is_volume,
					psi_bases[e], phi_bases[e], gbases[e],
					psi_cache, phi_cache,
					psi_vals, phi_vals);

				const Quadrature &quadrature = phi_vals.quadrature;
				assert(psi_vals.quadrature.weights.size() == quadrature.weights.size());
				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = phi_vals.det.array() * quadrature.weights.array();

				Eigen::MatrixXd local_hessian = compute_hessian(
					MixedNonLinearAssemblerData(psi_vals, phi_vals, t, dt, x_phi, x_psi, x_phi_prev, x_psi_prev, local_storage.da));

				const int n_phi_loc_bases = int(phi_vals.basis_values.size());
				const int n_psi_loc_bases = int(psi_vals.basis_values.size());
				const int local_size = n_phi_loc_bases * rows() + n_psi_loc_bases * cols();
				assert(local_hessian.rows() == local_size);
				assert(local_hessian.cols() == local_size);

				if (project_to_psd)
					local_hessian = ipc::project_to_psd(local_hessian);

				const auto local_index_data = [&](
												  const int local_index,
												  const std::vector<basis::Local2Global> *&global,
												  int &component,
												  int &component_count,
												  int &global_offset) {
					assert(local_index >= 0);
					assert(local_index < local_size);

					const int phi_local_size = n_phi_loc_bases * rows();
					if (local_index < phi_local_size)
					{
						const int basis_index = local_index / rows();
						global = &phi_vals.basis_values[basis_index].global;
						component = local_index % rows();
						component_count = rows();
						global_offset = 0;
						return;
					}

					const int psi_local_index = local_index - phi_local_size;
					const int basis_index = psi_local_index / cols();
					global = &psi_vals.basis_values[basis_index].global;
					component = psi_local_index % cols();
					component_count = cols();
					global_offset = phi_ndof;
				};

				for (int i = 0; i < local_size; ++i)
				{
					const std::vector<basis::Local2Global> *global_i = nullptr;
					int component_i = -1;
					int component_count_i = -1;
					int global_offset_i = -1;
					local_index_data(i, global_i, component_i, component_count_i, global_offset_i);

					for (int j = 0; j < local_size; ++j)
					{
						const double local_value = local_hessian(i, j);
						if (local_value == 0)
							continue;

						const std::vector<basis::Local2Global> *global_j = nullptr;
						int component_j = -1;
						int component_count_j = -1;
						int global_offset_j = -1;
						local_index_data(j, global_j, component_j, component_count_j, global_offset_j);

						for (const auto &gi : *global_i)
						{
							const int row = global_offset_i + gi.index * component_count_i + component_i;
							for (const auto &gj : *global_j)
							{
								const int col = global_offset_j + gj.index * component_count_j + component_j;
								local_storage.cache->add_value(e, row, col, local_value * gi.val * gj.val);
							}
						}

						if (local_storage.cache->entries_size() >= max_triplets_size)
						{
							local_storage.cache->prune();
							logger().debug("cleaning memory...");
						}
					}
				}
			}
		});

		for (LocalThreadMatStorage &local_storage : storage)
		{
			local_storage.cache->prune();
			mat_cache += *local_storage.cache;
		}
		hessian = mat_cache.get_matrix();
	}

	double NLAssembler::assemble_energy(
		const bool is_volume,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const double t,
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

				const double val = compute_energy(NonLinearAssemblerData(vals, t, dt, displacement, displacement_prev, local_storage.da));
				local_storage.val += val;
			}
		});

		double res = 0;
		// Serially merge local storages
		for (const LocalThreadScalarStorage &local_storage : storage)
			res += local_storage.val;
		return res;
	}

	Eigen::VectorXd NLAssembler::assemble_energy_per_element(
		const bool is_volume,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const double t,
		const double dt,
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &displacement_prev) const
	{
		auto storage = create_thread_storage(LocalThreadScalarStorage());
		const int n_bases = int(bases.size());
		Eigen::VectorXd out(bases.size());

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadScalarStorage &local_storage = get_local_thread_storage(storage, thread_id);
			ElementAssemblyValues &vals = local_storage.vals;

			for (int e = start; e < end; ++e)
			{
				cache.compute(e, is_volume, bases[e], gbases[e], vals);

				const Quadrature &quadrature = vals.quadrature;

				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = vals.det.array() * quadrature.weights.array();

				const double val = compute_energy(NonLinearAssemblerData(vals, t, dt, displacement, displacement_prev, local_storage.da));
				out[e] = val;
			}
		});

#ifndef NDEBUG
		const double assemble_val = assemble_energy(
			is_volume, bases, gbases, cache, t, dt, displacement, displacement_prev);
		assert(std::abs(assemble_val - out.sum()) < std::max(1e-10 * assemble_val, 1e-10));
#endif

		return out;
	}

	void NLAssembler::assemble_gradient(
		const bool is_volume,
		const int n_basis,
		const std::vector<ElementBases> &bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const double t,
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

				const auto val = assemble_gradient(NonLinearAssemblerData(vals, t, dt, displacement, displacement_prev, local_storage.da));
				assert(val.size() == n_loc_bases * size());

				for (int j = 0; j < n_loc_bases; ++j)
				{
					const auto &global_j = vals.basis_values[j].global;

					// igl::Timer t1; t1.start();
					for (int m = 0; m < size(); ++m)
					{
						const double local_value = val(j * size() + m);

						for (size_t jj = 0; jj < global_j.size(); ++jj)
						{
							const auto gj = global_j[jj].index * size() + m;
							const auto wj = global_j[jj].val;

							local_storage.vec(gj) += local_value * wj;
						}
					}

					// t1.stop();
					// if (!vals.has_parameterization) { logger().trace("-- t1: ", t1.getElapsedTime()); }
				}

				// timer.stop();
				// if (!vals.has_parameterization) { logger().trace("-- Timer: ", timer.getElapsedTime()); }
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
		const double t,
		const double dt,
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &displacement_prev,
		MatrixCache &mat_cache,
		StiffnessMatrix &hess) const
	{
		const int max_triplets_size = int(1e7);
		const int buffer_size = std::min(long(max_triplets_size), long(n_basis) * size());
		// logger().trace("buffer_size {}", buffer_size);

		// hess.resize(n_basis * size(), n_basis * size());
		// hess.setZero();

		mat_cache.init(n_basis * size());
		mat_cache.set_zero();

		auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, mat_cache));

		const int n_bases = int(bases.size());
		igl::Timer timer;
		timer.start();

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

				auto stiffness_val = assemble_hessian(NonLinearAssemblerData(vals, t, dt, displacement, displacement_prev, local_storage.da));
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

								for (size_t ii = 0; ii < global_i.size(); ++ii)
								{
									const auto gi = global_i[ii].index * size() + m;
									const auto wi = global_i[ii].val;

									for (size_t jj = 0; jj < global_j.size(); ++jj)
									{
										const auto gj = global_j[jj].index * size() + n;
										const auto wj = global_j[jj].val;

										local_storage.cache->add_value(e, gi, gj, local_value * wi * wj);
										// if (j < i) {
										// 	local_storage.entries.emplace_back(gj, gi, local_value * wj * wi);
										// }

										if (local_storage.cache->entries_size() >= max_triplets_size)
										{
											local_storage.cache->prune();
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

		timer.stop();
		logger().trace("done separate assembly {}s...", timer.getElapsedTime());

		timer.start();

		// Serially merge local storages
		for (LocalThreadMatStorage &local_storage : storage)
		{
			local_storage.cache->prune();
			mat_cache += *local_storage.cache;
		}
		hess = mat_cache.get_matrix();

		timer.stop();
		logger().trace("done merge assembly {}s...", timer.getElapsedTime());
	}

	void ElasticityAssembler::set_use_robust_jacobian()
	{
#ifdef POLYFEM_WITH_MISO
		use_robust_jacobian = true;
#else
		logger().error("Enable Bezier library to use the robust Jacobian check!");
#endif
	}
} // namespace polyfem::assembler
