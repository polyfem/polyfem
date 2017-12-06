#ifndef ASSEMBLER_HPP
#define ASSEMBLER_HPP

#include "Mesh.hpp"

#include "ElementAssemblyValues.hpp"
#include "Problem.hpp"
#include "Basis.hpp"

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

					// std::cout<<vali<<"\n\n"<<std::endl;
					// std::cout<<gradi<<"\n\n"<<std::endl;

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

		void bc(const std::vector< std::vector<Basis> > &bases, const Mesh &mesh, const std::vector<int> &bounday_nodes, const int resolution,  const Problem &problem, Eigen::MatrixXd &rhs) const
		{
			const int n_el=int(bases.size());

			Eigen::MatrixXd samples, tmp, rhs_fun;

			int index = 0;
			std::vector<int> indices; indices.reserve(n_el*10);
			std::map<int, int> global_index_to_col;

			for(int e = 0; e < n_el; ++e)
			{
				const std::vector<Basis> &bs = bases[e];
				const int n_local_bases = int(bs.size());

				for(int j = 0; j < n_local_bases; ++j)
				{
					const Basis &b=bs[j];

					if(std::find(bounday_nodes.begin(), bounday_nodes.end(), b.global_index()) != bounday_nodes.end()) //pt found
					{
						if(global_index_to_col.find( b.global_index() ) == global_index_to_col.end())
						{
							global_index_to_col[b.global_index()] = index++;
							indices.push_back(b.global_index());
						}
					}
				}
			}

			Eigen::MatrixXd global_mat = Eigen::MatrixXd::Zero(n_el*4*resolution, indices.size());
			Eigen::MatrixXd global_rhs = Eigen::MatrixXd::Zero(n_el*4*resolution, 1);

			index = 0;

			int global_counter = 0;

			for(int e = 0; e < n_el; ++e)
			{
				bool has_samples = sample_boundary(e, mesh, resolution, samples);

				if(!has_samples)
					continue;

				const std::vector<Basis> &bs = bases[e];
				const int n_local_bases = int(bs.size());

				Eigen::MatrixXd mapped = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());

				for(int j = 0; j < n_local_bases; ++j)
				{
					const Basis &b=bs[j];

					b.basis(samples, tmp);
					if(std::find(bounday_nodes.begin(), bounday_nodes.end(), b.global_index()) != bounday_nodes.end()) //pt found
						global_mat.block(global_counter, global_index_to_col[b.global_index()], tmp.size(), 1) = tmp;

					for (long k = 0; k < tmp.rows(); ++k){
						mapped.row(k) += tmp(k,0) * b.node();
					}
				}

				// std::cout<<samples<<"\n"<<std::endl;

				problem.bc(mapped, rhs_fun);
				global_rhs.block(global_counter, 0, rhs_fun.size(), 1) = rhs_fun;
				global_counter += rhs_fun.size();
			}

			const Eigen::MatrixXd global_mat_small = global_mat.block(0, 0, global_counter, global_mat.cols());
			const Eigen::MatrixXd global_rhs_small = global_rhs.block(0, 0, global_counter, global_rhs.cols());

			Eigen::MatrixXd coeffs = global_mat_small.colPivHouseholderQr().solve(global_rhs_small);

			// std::cout<<global_mat<<"\n"<<std::endl;
			// std::cout<<coeffs<<"\n"<<std::endl;
			// std::cout<<global_rhs<<"\n\n\n"<<std::endl;
			for(long i = 0; i < coeffs.size(); ++i){
				// problem.bc(mesh.pts.row(indices[i]), rhs_fun);

				// std::cout<<indices[i]<<" "<<coeffs(i)<<" vs " <<rhs_fun<<std::endl;
				rhs(indices[i]) = coeffs(i);
			}
		}

	private:
		LocalAssembler local_assembler_;

		bool sample_boundary(const int el_index, const Mesh &mesh, const int resolution, Eigen::MatrixXd &samples) const
		{
			auto el = mesh.els.row(el_index);

			const int n_x = mesh.n_x;
			const int n_y = mesh.n_y;

			const bool has_left = el(0) % (n_x + 1) != 0;
			const bool has_right = el(2) % (n_x + 1) != n_x;

			const bool has_bottom = el(0) / (n_x + 1) != 0;
			const bool has_top = el(2) / (n_x + 1) != n_y;

			int n = 0;
			if(!has_left) n+=resolution;
			if(!has_right) n+=resolution;
			if(!has_bottom) n+=resolution;
			if(!has_top) n+=resolution;

			if(n <= 0) return false;

			const Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(resolution, 0, 1);

			samples.resize(n, 2);
			n = 0;
			if(!has_left){
				samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
				samples.block(n, 1, resolution, 1) = t;

				n += resolution;
			}

			if(!has_bottom){
				samples.block(n, 0, resolution, 1) = t;
				samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);

				n += resolution;
			}

			if(!has_right){
				samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
				samples.block(n, 1, resolution, 1) = t;

				n += resolution;
			}

			if(!has_top){
				samples.block(n, 0, resolution, 1) = t;
				samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);

				n += resolution;
			}

			assert(long(n) == samples.rows());

			return true;
		}
	};
}

#endif //ASSEMBLER_HPP