#ifndef ASSEMBLER_HPP
#define ASSEMBLER_HPP

#include "ElementAssemblyValues.hpp"
#include "Problem.hpp"

#include <Eigen/Sparse>
#include <vector>

namespace poly_fem
{
	template<class LocalAssembler>
	class Assembler
	{
	public:
		void assemble(const int n_basis, const std::vector< ElementAssemblyValues > &values, const std::vector< ElementAssemblyValues > &geom_values, Eigen::SparseMatrix<double, Eigen::RowMajor> &stiffness) const
		{
			const int buffer_size = n_basis * 10 * local_assembler_.size();

			std::vector< Eigen::Triplet<double> > entries;
			entries.reserve(buffer_size);

			Eigen::MatrixXd local_val;

			const int n_values = int(values.size());
			for(int e=0; e < n_values; ++e)
			{
				const ElementAssemblyValues &vals  = values[e];
				const ElementAssemblyValues &gvals = geom_values[e];

				const Quadrature &quadrature = vals.quadrature;
				const int n_loc_bases = int(vals.basis_values.size());

				for(int i = 0; i < n_loc_bases; ++i)
				{
					const AssemblyValues &values_i = vals.basis_values[i];

					const Eigen::MatrixXd &vali  = values_i.val;
					const Eigen::MatrixXd &gradi = values_i.grad_t_m;

					for(int j = 0; j < n_loc_bases; ++j)
					{
						const AssemblyValues &values_j = vals.basis_values[j];

						const Eigen::MatrixXd &valj  = values_j.val;
						const Eigen::MatrixXd &gradj = values_j.grad_t_m;

						local_assembler_.assemble(gradi, gradj, local_val);
						const auto stiffness_val = (  local_val.array() * gvals.det.array() * quadrature.weights.array() ).colwise().sum();
						assert(stiffness_val.rows() == local_assembler_.size());
						assert(stiffness_val.cols() == local_assembler_.size());

						for(int m = 0; m < local_assembler_.size(); ++m)
						{
							for(int n = 0; n < local_assembler_.size(); ++n)
							{
								entries.push_back(Eigen::Triplet<double>(values_i.global_index*local_assembler_.size()+m, values_j.global_index*local_assembler_.size()+n, stiffness_val(m,n)));
							}
						}
					}
				}
			}

			stiffness.resize(n_basis*local_assembler_.size(), n_basis*local_assembler_.size());
			stiffness.setFromTriplets(entries.begin(), entries.end());
		}

		void set_identity(const std::vector<int> &bounday_nodes, Eigen::SparseMatrix<double, Eigen::RowMajor> &stiffness) const
		{
			for(std::size_t i = 0; i < bounday_nodes.size(); ++i)
			{
				const int index = bounday_nodes[i];
				for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(stiffness, index); it; ++it)
				{
					if(it.row() == it.col())
						it.valueRef() = 1;
					else
						it.valueRef() = 0;
				}
			}
		}

		void rhs(const int n_basis, const std::vector< ElementAssemblyValues > &values, const std::vector< ElementAssemblyValues > &geom_values, const Problem &problem, Eigen::MatrixXd &rhs) const
		{
			rhs = Eigen::MatrixXd::Zero(n_basis, 1);
			Eigen::MatrixXd rhs_fun;

			const int n_values = int(values.size());
			for(int e = 0; e < n_values; ++e)
			{
				const ElementAssemblyValues &vals  = values[e];
				const ElementAssemblyValues &gvals = geom_values[e];

				problem.rhs(gvals.val, rhs_fun);

				// std::cout<<e<<"\n"<<gvals.val<<"\n"<<rhs_fun<<"\n\n"<<std::endl;

				rhs_fun = rhs_fun.array() * gvals.det.array() * vals.quadrature.weights.array();

				// std::cout<<"after:\n"<<rhs_fun<<std::endl;

				const int n_loc_bases = int(vals.basis_values.size());
				for(int i = 0; i < n_loc_bases; ++i)
				{
					const AssemblyValues &v = vals.basis_values[i];

					const double rhs_value = (rhs_fun.array() * v.val.array()).sum();
					// std::cout<<i<<" "<<rhs_value<<std::endl;
					rhs(v.global_index) +=  rhs_value;
				}
			}
		}

		void bc(const Eigen::MatrixXd &pts, const std::vector<int> &bounday_nodes, const Problem &problem, Eigen::MatrixXd &rhs) const
		{
			Eigen::MatrixXd val;
			for(std::size_t i = 0; i < bounday_nodes.size(); ++i)
			{
				const int index = bounday_nodes[i];
				problem.bc(pts.row(index), val);

				rhs(index) = val(0,0);
			}
		}

	private:
		LocalAssembler local_assembler_;
	};
}

#endif //ASSEMBLER_HPP