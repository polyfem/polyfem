#ifndef ASSEMBLER_HPP
#define ASSEMBLER_HPP

#include "ElementAssemblyValues.hpp"
#include "Problem.hpp"
#include "Basis.hpp"
#include "LocalBoundary.hpp"
#include "QuadBoundarySampler.hpp"

#include <Eigen/Sparse>
#include <vector>
#include <iostream>

#include "UIState.hpp"

namespace poly_fem
{
	template<class LocalAssembler>
	class Assembler
	{
	public:
		void assemble(
			const int n_basis,
			const std::vector< ElementAssemblyValues > &values,
			const std::vector< ElementAssemblyValues > &geom_values,
			Eigen::SparseMatrix<double, Eigen::RowMajor> &stiffness) const
		{
			const int buffer_size = n_basis * local_assembler_.size();

			std::vector< Eigen::Triplet<double> > entries;
			entries.reserve(buffer_size);

			Eigen::MatrixXd local_val;
			stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
			stiffness.setZero();

			Eigen::SparseMatrix<double, Eigen::RowMajor> tmp(stiffness.rows(), stiffness.cols());

			const int n_values = int(values.size());
			for(int e=0; e < n_values; ++e)
			{
				const ElementAssemblyValues &vals  = values[e];
				const ElementAssemblyValues &gvals = geom_values[e];

				const Quadrature &quadrature = vals.quadrature;

				const Eigen::MatrixXd da = gvals.det.array() * quadrature.weights.array();
				const int n_loc_bases = int(vals.basis_values.size());

				// if(n_loc_bases == 3)
				// {
				// 	std::cout<<"gvals.det "<<gvals.det<<std::endl;
				// 	std::cout<<"quadrature.weights "<<quadrature.weights<<std::endl;
				// }

				for(int i = 0; i < n_loc_bases; ++i)
				{
					const AssemblyValues &values_i = vals.basis_values[i];

					// const Eigen::MatrixXd &vali  = values_i.val;
					const Eigen::MatrixXd &gradi = values_i.grad_t_m;

					// std::cout<<vali<<"\n\n"<<std::endl;
					// if(n_loc_bases == 3)
					// 	std::cout<<"gradi "<<gradi<<"\n\n"<<std::endl;

					for(int j = 0; j < n_loc_bases; ++j)
					{
						const AssemblyValues &values_j = vals.basis_values[j];

						// const Eigen::MatrixXd &valj  = values_j.val;
						const Eigen::MatrixXd &gradj = values_j.grad_t_m;
						// if(n_loc_bases == 3)
						// 	std::cout<<"gradj "<<gradj<<"\n\n"<<std::endl;


						local_assembler_.assemble(gradi, gradj, da, local_val);

						const auto stiffness_val = local_val.array().colwise().sum();
						assert(stiffness_val.size() == local_assembler_.size() * local_assembler_.size());
						// if(n_loc_bases == 3)
						// if(values_i.global[0].index == 0 && values_j.global[0].index == 0){
						// 	// std::cout<<e<<" "<<i<<" "<<j<<" "<<values_i.global[0].index<<" "<<values_j.global[0].index<<"\n--------------\n"<<gradi<<"\n"<<gradj<<std::endl;
						// 			// std::cout<<(gradi.array() * gradj.array()).rowwise().sum()<<" "<<da<<std::endl;
						// 	std::cout<<e<<" "<<values_i.global[0].index*local_assembler_.size() <<" "<< values_j.global[0].index*local_assembler_.size() <<" "<< stiffness_val <<std::endl;

						// }
						// exit(0);
						for(std::size_t ii = 0; ii < values_i.global.size(); ++ii)
						{
							for(std::size_t jj = 0; jj < values_j.global.size(); ++jj)
							{
								for(int m = 0; m < local_assembler_.size(); ++m)
								{
									for(int n = 0; n < local_assembler_.size(); ++n)
									{
										entries.emplace_back(
											values_i.global[ii].index*local_assembler_.size()+m,
											values_j.global[jj].index*local_assembler_.size()+n,
											stiffness_val(n*local_assembler_.size()+m) * values_i.global[ii].val * values_j.global[jj].val);
										// std::cout<<e<<" "<<values_i.global[ii].index*local_assembler_.size()+m <<" "<< values_j.global[jj].index*local_assembler_.size()+n <<" "<< stiffness_val(n*local_assembler_.size()+m) * values_i.global[ii].val * values_j.global[jj].val <<std::endl;
									}
								}
							}
						}
					}
				}

				if(entries.size() > 1e6)
				{
					tmp.setFromTriplets(entries.begin(), entries.end());
					stiffness += tmp;
				}
			}

			// stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
			// stiffness.setFromTriplets(entries.begin(), entries.end());
		}

		void set_identity(const std::vector<int> &boundary_nodes, Eigen::SparseMatrix<double, Eigen::RowMajor> &stiffness) const
		{
			for(std::size_t i = 0; i < boundary_nodes.size(); ++i)
			{
				const int index = boundary_nodes[i];
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(stiffness, index); it; ++it)
				{
					if(it.row() == it.col())
						it.valueRef() = 1;
					else
						it.valueRef() = 0;
				}
			}
		}

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};
}

#endif //ASSEMBLER_HPP
