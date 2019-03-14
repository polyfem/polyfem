#include <polyfem/Assembler.hpp>

#include <polyfem/Laplacian.hpp>
#include <polyfem/Helmholtz.hpp>
#include <polyfem/Bilaplacian.hpp>

#include <polyfem/LinearElasticity.hpp>
#include <polyfem/HookeLinearElasticity.hpp>
#include <polyfem/SaintVenantElasticity.hpp>
#include <polyfem/NeoHookeanElasticity.hpp>
// #include <polyfem/OgdenElasticity.hpp>

#include <polyfem/Stokes.hpp>
#include <polyfem/IncompressibleLinElast.hpp>

#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

#ifdef USE_TBB
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
			Eigen::SparseMatrix<double> tmp_mat;
			Eigen::SparseMatrix<double> stiffness;
            ElementAssemblyValues vals;
            QuadratureVector da;

			LocalThreadMatStorage(const int buffer_size, const int rows, const int cols)
			{
				entries.reserve(buffer_size);
				tmp_mat.resize(rows, cols);
				stiffness.resize(rows, cols);
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
	}

	template<class LocalAssembler>
	void Assembler<LocalAssembler>::assemble(
		const bool is_volume,
		const int n_basis,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		Eigen::SparseMatrix<double> &stiffness) const
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * local_assembler_.size());
// #ifdef USE_TBB
// 		buffer_size /= tbb::task_scheduler_init::default_num_threads();
// #endif
		logger().debug("buffer_size {}", buffer_size);
		try{
		stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		stiffness.setZero();

#ifdef USE_TBB
		typedef tbb::enumerable_thread_specific< LocalThreadMatStorage > LocalStorage;
		LocalStorage storages(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));
#else
		LocalThreadMatStorage loc_storage(buffer_size, stiffness.rows(), stiffness.cols());
#endif

		const int n_bases = int(bases.size());

#ifdef USE_TBB
		tbb::parallel_for( tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
		LocalStorage::reference loc_storage = storages.local();
		loc_storage.entries.reserve(buffer_size);
		for (int e = r.begin(); e != r.end(); ++e) {
#else
		for(int e=0; e < n_bases; ++e) {
#endif
            ElementAssemblyValues &vals = loc_storage.vals;
			igl::Timer timer; timer.start();
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			loc_storage.da = vals.det.array() * quadrature.weights.array();
			const int n_loc_bases = int(vals.basis_values.size());

			for(int i = 0; i < n_loc_bases; ++i)
			{
				// const AssemblyValues &values_i = vals.basis_values[i];
				// const Eigen::MatrixXd &gradi = values_i.grad_t_m;
				const auto &global_i = vals.basis_values[i].global;

				for(int j = 0; j <= i; ++j)
				{
					// const AssemblyValues &values_j = vals.basis_values[j];
					// const Eigen::MatrixXd &gradj = values_j.grad_t_m;
					const auto &global_j = vals.basis_values[j].global;

					const auto stiffness_val = local_assembler_.assemble(vals, i, j, loc_storage.da);
					assert(stiffness_val.size() == local_assembler_.size() * local_assembler_.size());

					igl::Timer t1; t1.start();
					for(int n = 0; n < local_assembler_.size(); ++n)
					{
						for(int m = 0; m < local_assembler_.size(); ++m)
						{
							const double local_value = stiffness_val(n*local_assembler_.size()+m);
							if (std::abs(local_value) < 1e-30) { continue; }

							for(size_t ii = 0; ii < global_i.size(); ++ii)
							{
								const auto gi = global_i[ii].index*local_assembler_.size()+m;
								const auto wi = global_i[ii].val;

								for(size_t jj = 0; jj < global_j.size(); ++jj)
								{
									const auto gj = global_j[jj].index*local_assembler_.size()+n;
									const auto wj = global_j[jj].val;

									loc_storage.entries.emplace_back(gi, gj, local_value * wi * wj);
									if (j < i) {
										loc_storage.entries.emplace_back(gj, gi, local_value * wj * wi);
									}

									if(loc_storage.entries.size() >= 1e8)
									{
										loc_storage.tmp_mat.setFromTriplets(loc_storage.entries.begin(), loc_storage.entries.end());
										loc_storage.stiffness += loc_storage.tmp_mat;

										loc_storage.tmp_mat.setZero();
										loc_storage.tmp_mat.data().squeeze();

										loc_storage.stiffness.makeCompressed();

										loc_storage.entries.clear();
										logger().debug("cleaning memory. Current storage: {}. mat nnz: {}", loc_storage.entries.capacity(), loc_storage.stiffness.nonZeros());
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
#ifdef USE_TBB
		}});
#else
		}
#endif
		logger().debug("done separate assembly...");

#ifdef USE_TBB
		for (LocalStorage::iterator i = storages.begin(); i != storages.end();  ++i)
		{
			logger().debug("local stiffness: {}, entries: {}", i->stiffness.nonZeros(), i->entries.size());
			stiffness += i->stiffness;

			i->stiffness.resize(0,0);
			i->stiffness.data().squeeze();

			i->tmp_mat.setFromTriplets(i->entries.begin(), i->entries.end());

			i->entries.clear();
			i->entries.shrink_to_fit();


			stiffness += i->tmp_mat;

			i->tmp_mat.resize(0,0);
			i->tmp_mat.data().squeeze();

			stiffness.makeCompressed();
		}
#else
		stiffness = loc_storage.stiffness;
		loc_storage.tmp_mat.setFromTriplets(loc_storage.entries.begin(), loc_storage.entries.end());
		stiffness += loc_storage.tmp_mat;
		stiffness.makeCompressed();
#endif
		} catch(std::bad_alloc &ba)
    	{
    		logger().error("bad alloc {}", ba.what());
    		exit(0);
    	}

		// stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		// stiffness.setFromTriplets(entries.begin(), entries.end());
	}


	template<class LocalAssembler>
	void MixedAssembler<LocalAssembler>::assemble(
		const bool is_volume,
		const int n_psi_basis,
		const int n_phi_basis,
		const std::vector< ElementBases > &psi_bases,
		const std::vector< ElementBases > &phi_bases,
		const std::vector< ElementBases > &gbases,
		Eigen::SparseMatrix<double> &stiffness) const
	{
		assert(phi_bases.size() == psi_bases.size());

		const int buffer_size = std::min(long(1e8), long(std::max(n_psi_basis, n_phi_basis)) * std::max(local_assembler_.rows(), local_assembler_.cols()));
		logger().debug("buffer_size {}", buffer_size);

		stiffness.resize(n_phi_basis*local_assembler_.rows(), n_psi_basis*local_assembler_.cols());
		stiffness.setZero();

#ifdef USE_TBB
		typedef tbb::enumerable_thread_specific< LocalThreadMatStorage > LocalStorage;
		LocalStorage storages(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));
#else
		LocalThreadMatStorage loc_storage(buffer_size, stiffness.rows(), stiffness.cols());
        ElementAssemblyValues psi_vals, phi_vals;
#endif

		const int n_bases = int(phi_bases.size());

#ifdef USE_TBB
		tbb::parallel_for( tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
		LocalStorage::reference loc_storage = storages.local();
        ElementAssemblyValues psi_vals, phi_vals;
		for (int e = r.begin(); e != r.end(); ++e) {
#else
		for(int e=0; e < n_bases; ++e) {
#endif
			igl::Timer timer; timer.start();
			psi_vals.compute(e, is_volume, psi_bases[e], gbases[e]);
			phi_vals.compute(e, is_volume, phi_bases[e], gbases[e]);

			const Quadrature &quadrature = phi_vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			loc_storage.da = phi_vals.det.array() * quadrature.weights.array();
			const int n_phi_loc_bases = int(phi_vals.basis_values.size());
			const int n_psi_loc_bases = int(psi_vals.basis_values.size());

			for(int i = 0; i < n_psi_loc_bases; ++i)
			{
				const auto &global_i = psi_vals.basis_values[i].global;

				for(int j = 0; j < n_phi_loc_bases; ++j)
				{
					const auto &global_j = phi_vals.basis_values[j].global;

					const auto stiffness_val = local_assembler_.assemble(psi_vals, phi_vals, i, j, loc_storage.da);
					assert(stiffness_val.size() == local_assembler_.rows() * local_assembler_.cols());

					// igl::Timer t1; t1.start();
					for(int n = 0; n < local_assembler_.rows(); ++n)
					{
						for(int m = 0; m < local_assembler_.cols(); ++m)
						{
							const double local_value = stiffness_val(n*local_assembler_.cols() + m);
							if (std::abs(local_value) < 1e-30) { continue; }

							for(size_t ii = 0; ii < global_i.size(); ++ii)
							{
								const auto gi = global_i[ii].index*local_assembler_.cols()+m;
								const auto wi = global_i[ii].val;

								for(size_t jj = 0; jj < global_j.size(); ++jj)
								{
									const auto gj = global_j[jj].index*local_assembler_.rows()+n;
									const auto wj = global_j[jj].val;

									loc_storage.entries.emplace_back(gj, gi, local_value * wi * wj);

									if(loc_storage.entries.size() >= 1e8)
									{
										loc_storage.tmp_mat.setFromTriplets(loc_storage.entries.begin(), loc_storage.entries.end());
										loc_storage.stiffness += loc_storage.tmp_mat;
										loc_storage.stiffness.makeCompressed();

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
#ifdef USE_TBB
		}});
#else
		}
#endif


#ifdef USE_TBB
		for (LocalStorage::iterator i = storages.begin(); i != storages.end();  ++i)
		{
			stiffness += i->stiffness;
			i->tmp_mat.setFromTriplets(i->entries.begin(), i->entries.end());
			stiffness += i->tmp_mat;
		}
#else
		stiffness = loc_storage.stiffness;
		loc_storage.tmp_mat.setFromTriplets(loc_storage.entries.begin(), loc_storage.entries.end());
		stiffness += loc_storage.tmp_mat;
#endif
		stiffness.makeCompressed();

		// stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		// stiffness.setFromTriplets(entries.begin(), entries.end());
	}


	template<class LocalAssembler>
	void NLAssembler<LocalAssembler>::assemble_grad(
		const bool is_volume,
		const int n_basis,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		const Eigen::MatrixXd &displacement,
		Eigen::MatrixXd &rhs) const
	{
		rhs.resize(n_basis*local_assembler_.size(), 1);
		rhs.setZero();

#ifdef USE_TBB
		typedef tbb::enumerable_thread_specific< LocalThreadVecStorage > LocalStorage;
		LocalStorage storages(LocalThreadVecStorage(rhs.size()));
#else
		LocalThreadVecStorage loc_storage(rhs.size());
#endif


		const int n_bases = int(bases.size());

#ifdef USE_TBB
		tbb::parallel_for( tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
		LocalStorage::reference loc_storage = storages.local();
		for (int e = r.begin(); e != r.end(); ++e) {
#else
		for(int e=0; e < n_bases; ++e) {
#endif
			// igl::Timer timer; timer.start();

			ElementAssemblyValues &vals = loc_storage.vals;
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			loc_storage.da = vals.det.array() * quadrature.weights.array();
			const int n_loc_bases = int(vals.basis_values.size());
			const auto val = local_assembler_.assemble(vals, displacement, loc_storage.da);
			assert(val.size() == n_loc_bases*local_assembler_.size());

			for(int j = 0; j < n_loc_bases; ++j)
			{
				const auto &global_j = vals.basis_values[j].global;

				// igl::Timer t1; t1.start();
				for(int m = 0; m < local_assembler_.size(); ++m)
				{
					const double local_value = val(j*local_assembler_.size() + m);
					if (std::abs(local_value) < 1e-30) { continue; }

					for(size_t jj = 0; jj < global_j.size(); ++jj)
					{
						const auto gj = global_j[jj].index*local_assembler_.size() + m;
						const auto wj = global_j[jj].val;

						loc_storage.vec(gj) += local_value * wj;
					}
				}

				// t1.stop();
				// if (!vals.has_parameterization) { std::cout << "-- t1: " << t1.getElapsedTime() << std::endl; }
			}

			// timer.stop();
			// if (!vals.has_parameterization) { std::cout << "-- Timer: " << timer.getElapsedTime() << std::endl; }

#ifdef USE_TBB
		}});
#else
		}
#endif

#ifdef USE_TBB
	for (LocalStorage::iterator i = storages.begin(); i != storages.end();  ++i)
	{
		rhs += i->vec;
	}
#else
		rhs = loc_storage.vec;
#endif
	}


	template<class LocalAssembler>
	void NLAssembler<LocalAssembler>::assemble_hessian(
		const bool is_volume,
		const int n_basis,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		const Eigen::MatrixXd &displacement,
		Eigen::SparseMatrix<double> &grad) const
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * local_assembler_.size());
		// std::cout<<"buffer_size "<<buffer_size<<std::endl;

		grad.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		grad.setZero();

#ifdef USE_TBB
		typedef tbb::enumerable_thread_specific< LocalThreadMatStorage > LocalStorage;
		LocalStorage storages(LocalThreadMatStorage(buffer_size, grad.rows(), grad.cols()));
#else
		LocalThreadMatStorage loc_storage(buffer_size, grad.rows(), grad.cols());
#endif

		const int n_bases = int(bases.size());

#ifdef USE_TBB
		tbb::parallel_for( tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
		LocalStorage::reference loc_storage = storages.local();
		for (int e = r.begin(); e != r.end(); ++e) {
#else
		for(int e=0; e < n_bases; ++e) {
#endif
			ElementAssemblyValues &vals = loc_storage.vals;
			// igl::Timer timer; timer.start();
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			loc_storage.da = vals.det.array() * quadrature.weights.array();
			const int n_loc_bases = int(vals.basis_values.size());

			const auto stiffness_val = local_assembler_.assemble_grad(vals, displacement, loc_storage.da);
			assert(stiffness_val.rows() == n_loc_bases * local_assembler_.size());
			assert(stiffness_val.cols() == n_loc_bases * local_assembler_.size());

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


				// igl::Timer t1; t1.start();
			for(int i = 0; i < n_loc_bases; ++i)
			{
				const auto &global_i = vals.basis_values[i].global;

				// for(int j = 0; j < n_loc_bases; ++j)
				for(int j = 0; j <= i; ++j)
				{
					const auto &global_j = vals.basis_values[j].global;

					for(int n = 0; n < local_assembler_.size(); ++n)
					{
						for(int m = 0; m < local_assembler_.size(); ++m)
						{
							const double local_value = stiffness_val(i*local_assembler_.size() + m, j*local_assembler_.size() + n);
							if (std::abs(local_value) < 1e-30) { continue; }

							for(size_t ii = 0; ii < global_i.size(); ++ii)
							{
								const auto gi = global_i[ii].index*local_assembler_.size() + m;
								const auto wi = global_i[ii].val;

								for(size_t jj = 0; jj < global_j.size(); ++jj)
								{
									const auto gj = global_j[jj].index*local_assembler_.size() + n;
									const auto wj = global_j[jj].val;

									loc_storage.entries.emplace_back(gi, gj, local_value * wi * wj);
									if (j < i) {
										loc_storage.entries.emplace_back(gj, gi, local_value * wj * wi);
									}

									if(loc_storage.entries.size() >= 1e8)
									{
										loc_storage.tmp_mat.setFromTriplets(loc_storage.entries.begin(), loc_storage.entries.end());
										loc_storage.stiffness += loc_storage.tmp_mat;
										loc_storage.stiffness.makeCompressed();

										loc_storage.entries.clear();
										logger().debug("cleaning memory...");
									}
								}
							}
						}
					}
				}
				// t1.stop();
				// if (!vals.has_parameterization) { std::cout << "-- t1: " << t1.getElapsedTime() << std::endl; }
			}

			// timer.stop();
			// if (!vals.has_parameterization) { std::cout << "-- Timer: " << timer.getElapsedTime() << std::endl; }
#ifdef USE_TBB
		}});
#else
		}
#endif


#ifdef USE_TBB
	for (LocalStorage::iterator i = storages.begin(); i != storages.end();  ++i)
	{
		grad += i->stiffness;
		i->tmp_mat.setFromTriplets(i->entries.begin(), i->entries.end());
		grad += i->tmp_mat;
	}
#else
		grad = loc_storage.stiffness;
		loc_storage.tmp_mat.setFromTriplets(loc_storage.entries.begin(), loc_storage.entries.end());
		grad += loc_storage.tmp_mat;
#endif
		grad.makeCompressed();
	}

	template<class LocalAssembler>
	double NLAssembler<LocalAssembler>::assemble(
		const bool is_volume,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		const Eigen::MatrixXd &displacement) const
	{
#ifdef USE_TBB
		typedef tbb::enumerable_thread_specific< LocalThreadScalarStorage > LocalStorage;
		LocalStorage storages((LocalThreadScalarStorage()));
#else
		LocalThreadScalarStorage loc_storage;
#endif
		const int n_bases = int(bases.size());

#ifdef USE_TBB
		tbb::parallel_for( tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
		LocalStorage::reference loc_storage = storages.local();
		for (int e = r.begin(); e != r.end(); ++e) {
#else
		for(int e=0; e < n_bases; ++e) {
#endif
			// igl::Timer timer; timer.start();

			ElementAssemblyValues &vals = loc_storage.vals;
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			loc_storage.da = vals.det.array() * quadrature.weights.array();

			const double val = local_assembler_.compute_energy(vals, displacement, loc_storage.da);
			loc_storage.val += val;
#ifdef USE_TBB
		}});
#else
		}
#endif


#ifdef USE_TBB
	double res = 0;
	for (LocalStorage::iterator i = storages.begin(); i != storages.end();  ++i)
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
	template class Assembler<HookeLinearElasticity>;
	template class NLAssembler<SaintVenantElasticity>;
	template class NLAssembler<NeoHookeanElasticity>;
	// template class NLAssembler<OgdenElasticity>;

	template class Assembler<StokesVelocity>;
	template class MixedAssembler<StokesMixed>;
	template class Assembler<StokesPressure>;

	template class Assembler<IncompressibleLinearElasticityDispacement>;
	template class MixedAssembler<IncompressibleLinearElasticityMixed>;
	template class Assembler<IncompressibleLinearElasticityPressure>;
}
