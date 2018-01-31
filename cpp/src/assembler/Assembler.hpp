#ifndef ASSEMBLER_HPP
#define ASSEMBLER_HPP

#include "ElementAssemblyValues.hpp"

#include <igl/Timer.h>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>

namespace poly_fem
{
	template<class LocalAssembler>
	class Assembler
	{
	public:
		void assemble(
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
				// const ElementAssemblyValues &vals  = values[e];
				// const ElementAssemblyValues &gvals = geom_values[e];

				ElementAssemblyValues vals;
				igl::Timer timer; timer.start();
				vals.compute(e, is_volume, bases[e], gbases[e]);

				const Quadrature &quadrature = vals.quadrature;

				const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();
				const int n_loc_bases = int(vals.basis_values.size());

				// if(n_loc_bases == 3)
				// {
				// 	std::cout<<"gvals.det "<<gvals.det<<std::endl;
				// 	std::cout<<"quadrature.weights "<<quadrature.weights<<std::endl;
				// }

				for(int i = 0; i < n_loc_bases; ++i)
				{
					const AssemblyValues &values_i = vals.basis_values[i];
					const Eigen::MatrixXd &gradi = values_i.grad_t_m;

					for(int j = 0; j <= i; ++j)
					{
						const AssemblyValues &values_j = vals.basis_values[j];

						const Eigen::MatrixXd &gradj = values_j.grad_t_m;

						const auto stiffness_val = local_assembler_.assemble(gradi, gradj, da);

						// const auto stiffness_val = local_val.array().colwise().sum();
						assert(stiffness_val.size() == local_assembler_.size() * local_assembler_.size());

						igl::Timer t1; t1.start();
						for(int n = 0; n < local_assembler_.size(); ++n)
						{
							for(int m = 0; m < local_assembler_.size(); ++m)
							{
								const double local_value = stiffness_val(n*local_assembler_.size()+m);
								if (std::abs(local_value) < 1e-30) { continue; }
								for(size_t ii = 0; ii < values_i.global.size(); ++ii)
								{
									const auto gi = values_i.global[ii].index*local_assembler_.size()+m;
									const auto wi = values_i.global[ii].val;
									for(size_t jj = 0; jj < values_j.global.size(); ++jj)
									{
										const auto gj = values_j.global[jj].index*local_assembler_.size()+n;
										const auto wj = values_j.global[jj].val;
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
											//entries.reserve(buffer_size); // not needed (a std::vector never frees memory)
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

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};
}

#endif //ASSEMBLER_HPP
