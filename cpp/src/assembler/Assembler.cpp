#include "Assembler.hpp"

#include "Laplacian.hpp"
#include "LinearElasticity.hpp"
#include "HookeLinearElasticity.hpp"
#include "SaintVenantElasticity.hpp"

namespace poly_fem
{
	template<class LocalAssembler>
	void Assembler<LocalAssembler>::assemble(
		const bool is_volume,
		const int n_basis,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		Eigen::SparseMatrix<double> &stiffness) const
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * local_assembler_.size());

		std::vector< Eigen::Triplet<double> > entries;
		entries.reserve(buffer_size);
		std::cout<<"buffer_size "<<buffer_size<<std::endl;

		Eigen::MatrixXd local_val;
		stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		stiffness.setZero();

		Eigen::SparseMatrix<double> tmp(stiffness.rows(), stiffness.cols());

		const int n_bases = int(bases.size());
		for(int e=0; e < n_bases; ++e)
		{
			ElementAssemblyValues vals;
			igl::Timer timer; timer.start();
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();
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
									
									entries.emplace_back(gi, gj, local_value * wi * wj);
									if (j < i) {
										entries.emplace_back(gj, gi, local_value * wj * wi);
									}

									if(entries.size() >= 1e8)
									{
										tmp.setFromTriplets(entries.begin(), entries.end());
										stiffness += tmp;
										stiffness.makeCompressed();

										entries.clear();
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

		}

		tmp.setFromTriplets(entries.begin(), entries.end());
		stiffness += tmp;
		stiffness.makeCompressed();

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
		Eigen::MatrixXd local_val;
		rhs.resize(n_basis*local_assembler_.size(), 1);
		rhs.setZero();

		const int n_bases = int(bases.size());
		for(int e=0; e < n_bases; ++e)
		{
			// igl::Timer timer; timer.start();

			ElementAssemblyValues vals;
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();
			const int n_loc_bases = int(vals.basis_values.size());

			for(int j = 0; j < n_loc_bases; ++j)
			{
				const auto &global_j = vals.basis_values[j].global;
				const auto val = local_assembler_.assemble(vals, j, displacement, da);

				assert(val.size() == local_assembler_.size());

				// igl::Timer t1; t1.start();
				for(int m = 0; m < local_assembler_.size(); ++m)
				{
					const double local_value = val(m);
					if (std::abs(local_value) < 1e-30) { continue; }

					for(size_t jj = 0; jj < global_j.size(); ++jj)
					{
						const auto gj = global_j[jj].index*local_assembler_.size();
						const auto wj = global_j[jj].val;

						rhs(gj) += local_value * wj;
					}
				}

				// t1.stop();
				// if (!vals.has_parameterization) { std::cout << "-- t1: " << t1.getElapsedTime() << std::endl; }
			}

			// timer.stop();
			// if (!vals.has_parameterization) { std::cout << "-- Timer: " << timer.getElapsedTime() << std::endl; }

		}
	}


	template<class LocalAssembler>
	void NLAssembler<LocalAssembler>::assemble_grad(
		const bool is_volume,
		const int n_basis,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		const Eigen::MatrixXd &displacement,
		Eigen::SparseMatrix<double> &grad) const
	{
		const int buffer_size = std::min(long(1e8), long(n_basis) * local_assembler_.size());

		std::vector< Eigen::Triplet<double> > entries;
		entries.reserve(buffer_size);
		std::cout<<"buffer_size "<<buffer_size<<std::endl;

		Eigen::MatrixXd local_val;
		grad.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
		grad.setZero();

		Eigen::SparseMatrix<double> tmp(grad.rows(), grad.cols());

		const int n_bases = int(bases.size());
		for(int e = 0; e < n_bases; ++e)
		{
			// igl::Timer timer; timer.start();

			ElementAssemblyValues vals;
			vals.compute(e, is_volume, bases[e], gbases[e]);

			const Quadrature &quadrature = vals.quadrature;

			const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();
			const int n_loc_bases = int(vals.basis_values.size());

			for(int j = 0; j < n_loc_bases; ++j)
			{
				const auto &global_j = vals.basis_values[j].global;

				const auto stiffness_val = local_assembler_.assemble_grad(vals, j, displacement, da);
				assert(stiffness_val.size() == local_assembler_.size() * local_assembler_.size());

				// igl::Timer t1; t1.start();
				for(int n = 0; n < local_assembler_.size(); ++n)
				{
					for(int m = 0; m < local_assembler_.size(); ++m)
					{
						const double local_value = stiffness_val(n*local_assembler_.size()+m);
						if (std::abs(local_value) < 1e-30) { continue; }

						for(size_t jj = 0; jj < global_j.size(); ++jj)
						{
							const auto gj = global_j[jj].index*local_assembler_.size()+n;
							const auto wj = global_j[jj].val;

							entries.emplace_back(m, gj, local_value * wj);

							if(entries.size() >= 1e8)
							{
								tmp.setFromTriplets(entries.begin(), entries.end());
								grad += tmp;
								grad.makeCompressed();

								entries.clear();
								std::cout<<"cleaning memory..."<<std::endl;
							}
						}
					}
				}
				// t1.stop();
				// if (!vals.has_parameterization) { std::cout << "-- t1: " << t1.getElapsedTime() << std::endl; }
			}

			// timer.stop();
			// if (!vals.has_parameterization) { std::cout << "-- Timer: " << timer.getElapsedTime() << std::endl; }
		}

		tmp.setFromTriplets(entries.begin(), entries.end());
		grad += tmp;
		grad.makeCompressed();
	}

	template class Assembler<Laplacian>;
	template class Assembler<LinearElasticity>;
	template class Assembler<HookeLinearElasticity>;
	template class NLAssembler<SaintVenantElasticity>;
}