#include <polyfem/Assembler.hpp>

#include <polyfem/Laplacian.hpp>
#include <polyfem/Helmholtz.hpp>

#include <polyfem/LinearElasticity.hpp>
#include <polyfem/HookeLinearElasticity.hpp>
#include <polyfem/SaintVenantElasticity.hpp>
#include <polyfem/NeoHookeanElasticity.hpp>
#include <polyfem/OgdenElasticity.hpp>

#include <igl/Timer.h>

#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/enumerable_thread_specific.h>


class LocalThreadMatStorage
{
public:
	std::vector< Eigen::Triplet<double> > entries;
	Eigen::SparseMatrix<double> tmp_mat;
	Eigen::SparseMatrix<double> stiffness;

	LocalThreadMatStorage(const int buffer_size, const int mat_size)
	{
		entries.reserve(buffer_size);
		tmp_mat.resize(mat_size, mat_size);
		stiffness.resize(mat_size, mat_size);
	}
};

class LocalThreadVecStorage
{
public:
	Eigen::MatrixXd vec;

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

	LocalThreadScalarStorage()
	{
		val = 0;
	}
};

namespace polyfem
{
	template<class LocalAssembler>
	void Assembler<LocalAssembler>::assemble(
		const bool is_volume,
		const int n_basis,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		Eigen::SparseMatrix<double> &stiffness)
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * local_assembler_.size());
		std::cout<<"buffer_size "<<buffer_size<<std::endl;

		stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		stiffness.setZero();

#ifdef USE_TBB
		typedef tbb::enumerable_thread_specific< LocalThreadMatStorage > LocalStorage;
		LocalStorage storages(LocalThreadMatStorage(buffer_size, stiffness.rows()));
#else
		LocalThreadMatStorage loc_storage(buffer_size, stiffness.rows());
#endif

		const int n_bases = int(bases.size());

#ifdef USE_TBB
		tbb::parallel_for( tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
		LocalStorage::reference loc_storage = storages.local();
		for (int e = r.begin(); e != r.end(); ++e) {
#else
		for(int e=0; e < n_bases; ++e) {
#endif
			ElementAssemblyValues vals;
			igl::Timer timer; timer.start();
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			const QuadratureVector da = vals.det.array() * quadrature.weights.array();
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

					const auto stiffness_val = local_assembler_.assemble(vals, i, j, da);
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
										loc_storage.stiffness.makeCompressed();

										loc_storage.entries.clear();
										std::cout<<"cleaning memory..."<<std::endl;
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
	void Assembler<LocalAssembler>::assemble_mass_matrix(
		const bool is_volume,
		const int n_basis,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		Eigen::SparseMatrix<double> &mass)
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * local_assembler_.size());
		std::cout<<"buffer_size "<<buffer_size<<std::endl;

		assert(local_assembler_.size() == 1);
		mass.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		mass.setZero();

#ifdef USE_TBB
		typedef tbb::enumerable_thread_specific< LocalThreadMatStorage > LocalStorage;
		LocalStorage storages(LocalThreadMatStorage(buffer_size, mass.rows()));
#else
		LocalThreadMatStorage loc_storage(buffer_size, mass.rows());
#endif

		const int n_bases = int(bases.size());

#ifdef USE_TBB
		tbb::parallel_for( tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
		LocalStorage::reference loc_storage = storages.local();
		for (int e = r.begin(); e != r.end(); ++e) {
#else
		for(int e=0; e < n_bases; ++e) {
#endif
			ElementAssemblyValues vals;
			igl::Timer timer; timer.start();
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			const QuadratureVector da = vals.det.array() * quadrature.weights.array();
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

					// const auto stiffness_val = local_assembler_.assemble(vals, i, j, da);
					// assert(stiffness_val.size() == local_assembler_.size() * local_assembler_.size());
					const double val = (vals.basis_values[i].val.array() * vals.basis_values[j].val.array() * da.array()).sum();

					igl::Timer t1; t1.start();
					for(int n = 0; n < local_assembler_.size(); ++n)
					{
						for(int m = 0; m < local_assembler_.size(); ++m)
						{
							const double local_value = val; //(n*local_assembler_.size()+m);
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
										loc_storage.stiffness.makeCompressed();

										loc_storage.entries.clear();
										std::cout<<"cleaning memory..."<<std::endl;
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
			mass += i->stiffness;
			i->tmp_mat.setFromTriplets(i->entries.begin(), i->entries.end());
			mass += i->tmp_mat;
		}
#else
		mass = loc_storage.stiffness;
		loc_storage.tmp_mat.setFromTriplets(loc_storage.entries.begin(), loc_storage.entries.end());
		mass += loc_storage.tmp_mat;
#endif
		mass.makeCompressed();

		// stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		// stiffness.setFromTriplets(entries.begin(), entries.end());
	}


	template<class LocalAssembler>
	void NLAssembler<LocalAssembler>::assemble(
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

			ElementAssemblyValues vals;
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			const QuadratureVector da = vals.det.array() * quadrature.weights.array();
			const int n_loc_bases = int(vals.basis_values.size());
			const auto val = local_assembler_.assemble(vals, displacement, da);
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
	void NLAssembler<LocalAssembler>::assemble_grad(
		const bool is_volume,
		const int n_basis,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		const Eigen::MatrixXd &displacement,
		Eigen::SparseMatrix<double> &grad)
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * local_assembler_.size());
		// std::cout<<"buffer_size "<<buffer_size<<std::endl;

		grad.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		grad.setZero();

#ifdef USE_TBB
		typedef tbb::enumerable_thread_specific< LocalThreadMatStorage > LocalStorage;
		LocalStorage storages(LocalThreadMatStorage(buffer_size, grad.rows()));
#else
		LocalThreadMatStorage loc_storage(buffer_size, grad.rows());
#endif

		const int n_bases = int(bases.size());

#ifdef USE_TBB
		tbb::parallel_for( tbb::blocked_range<int>(0, n_bases), [&](const tbb::blocked_range<int> &r) {
		LocalStorage::reference loc_storage = storages.local();
		for (int e = r.begin(); e != r.end(); ++e) {
#else
		for(int e=0; e < n_bases; ++e) {
#endif
			ElementAssemblyValues vals;
			// igl::Timer timer; timer.start();
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			const QuadratureVector da = vals.det.array() * quadrature.weights.array();
			const int n_loc_bases = int(vals.basis_values.size());

			const auto stiffness_val = local_assembler_.assemble_grad(vals, displacement, da);
			assert(stiffness_val.rows() == n_loc_bases * local_assembler_.size());
			assert(stiffness_val.cols() == n_loc_bases * local_assembler_.size());


				// igl::Timer t1; t1.start();
			for(int i = 0; i < n_loc_bases; ++i)
			{
				const auto &global_i = vals.basis_values[i].global;

				for(int j = 0; j < n_loc_bases; ++j)
				{
					const auto &global_j = vals.basis_values[j].global;

					for(int n = 0; n < local_assembler_.size(); ++n)
					{
						for(int m = 0; m < local_assembler_.size(); ++m)
						{
							const double local_value = stiffness_val(i*local_assembler_.size() + m, j*local_assembler_.size() + n);
							// if (!use_sparse_cached && std::abs(local_value) < 1e-30) { continue; }


							for(size_t ii = 0; ii < global_i.size(); ++ii)
							{
								const auto gi = global_i[ii].index*local_assembler_.size() + m;
								const auto wi = global_i[ii].val;

								for(size_t jj = 0; jj < global_j.size(); ++jj)
								{
									const auto gj = global_j[jj].index*local_assembler_.size() + n;
									const auto wj = global_j[jj].val;

									// std::cout<< gi <<"," <<gj<<" -> "<<local_value<<std::endl;

									loc_storage.entries.emplace_back(gi, gj, local_value * wi * wj);

									if(loc_storage.entries.size() >= 1e8)
									{
										loc_storage.tmp_mat.setFromTriplets(loc_storage.entries.begin(), loc_storage.entries.end());
										loc_storage.stiffness += loc_storage.tmp_mat;
										loc_storage.stiffness.makeCompressed();

										loc_storage.entries.clear();
										std::cout<<"cleaning memory..."<<std::endl;
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
	double NLAssembler<LocalAssembler>::compute_energy(
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

			ElementAssemblyValues vals;
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
			const QuadratureVector da = vals.det.array() * quadrature.weights.array();

			const double val = local_assembler_.compute_energy(vals, displacement, da);
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
	template class Assembler<LinearElasticity>;
	template class Assembler<HookeLinearElasticity>;
	template class NLAssembler<SaintVenantElasticity>;
	template class NLAssembler<NeoHookeanElasticity>;
	template class NLAssembler<OgdenElasticity>;
}
