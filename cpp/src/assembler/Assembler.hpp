#ifndef ASSEMBLER_HPP
#define ASSEMBLER_HPP

#include "Mesh.hpp"

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
		//TODO refactor this and set identity? and maybe rhs to another file
		void compute_assembly_values(const bool is_volume, const std::vector< ElementBases > &bases, std::vector< ElementAssemblyValues > &values)
		{
			values.resize(bases.size());

			for(std::size_t i = 0; i < bases.size(); ++i)
			{
				const Quadrature &quadrature = bases[i].quadrature;
				const ElementBases &bs = bases[i];
				ElementAssemblyValues &vals = values[i];
				vals.basis_values.resize(bs.bases.size());
				vals.quadrature = quadrature;

				Eigen::MatrixXd mval = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

				Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
				Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
				Eigen::MatrixXd dzmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

				const int n_local_bases = int(bs.bases.size());
				for(int j = 0; j < n_local_bases; ++j)
				{
					const Basis &b=bs.bases[j];
					AssemblyValues &val = vals.basis_values[j];

					val.global_index = b.global_index();


					b.basis(quadrature.points, val.val);
					b.grad(quadrature.points, val.grad);

					if(!bs.has_parameterization) continue;

					for (long k = 0; k < val.val.rows(); ++k){
						mval.row(k) += val.val(k,0)    * b.node();

						dxmv.row(k) += val.grad(k,0) * b.node();
						dymv.row(k) += val.grad(k,1) * b.node();
						if(is_volume)
							dzmv.row(k) += val.grad(k,2) * b.node();
					}
				}

				if(!bs.has_parameterization)
					vals.finalize_global_element(quadrature.points);
				else
				{
					if(is_volume)
						vals.finalize(mval, dxmv, dymv, dzmv);
					else
						vals.finalize(mval, dxmv, dymv);
				}
			}
		}



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
						// 	std::cout<<e<<" "<<i<<" "<<j<<" "<<stiffness_val<<std::endl;
						// exit(0);
						for(int m = 0; m < local_assembler_.size(); ++m)
						{
							for(int n = 0; n < local_assembler_.size(); ++n)
							{
								entries.push_back(Eigen::Triplet<double>(values_i.global_index*local_assembler_.size()+m, values_j.global_index*local_assembler_.size()+n, stiffness_val(m*local_assembler_.size()+n)));
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
			rhs = Eigen::MatrixXd::Zero(n_basis * local_assembler_.size(), 1);
			Eigen::MatrixXd rhs_fun;

			const int n_values = int(values.size());
			for(int e = 0; e < n_values; ++e)
			{
				const ElementAssemblyValues &vals  = values[e];
				const ElementAssemblyValues &gvals = geom_values[e];

				problem.rhs(gvals.val, rhs_fun);

				// std::cout<<e<<"\n"<<gvals.val<<"\n"<<rhs_fun<<"\n\n"<<std::endl;

				for(int d = 0; d < local_assembler_.size(); ++d)
					rhs_fun.col(d) = rhs_fun.col(d).array() * gvals.det.array() * vals.quadrature.weights.array();

				// std::cout<<"after:\n"<<rhs_fun<<std::endl;

				const int n_loc_bases = int(vals.basis_values.size());
				for(int i = 0; i < n_loc_bases; ++i)
				{
					const AssemblyValues &v = vals.basis_values[i];

					for(int d = 0; d < local_assembler_.size(); ++d)
					{
						const double rhs_value = (rhs_fun.col(d).array() * v.val.array()).sum();
						// std::cout<<i<<" "<<rhs_value<<std::endl;
						rhs(v.global_index*local_assembler_.size()+d) +=  rhs_value;
					}
				}
			}
		}

		void bc(const std::vector< ElementBases > &bases, const Mesh &mesh, const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution,  const Problem &problem, Eigen::MatrixXd &rhs) const
		{
			const int n_el=int(bases.size());

			Eigen::MatrixXd samples, tmp, rhs_fun;

			int index = 0;
			std::vector<int> indices; indices.reserve(n_el*10);
			std::map<int, int> global_index_to_col;

			long total_size = 0;

			for(int e = 0; e < n_el; ++e)
			{
				bool has_samples = sample_boundary(mesh.is_volume(), local_boundary[e], resolution, true, samples);

				if(!has_samples)
					continue;

				const ElementBases &bs = bases[e];
				const int n_local_bases = int(bs.bases.size());

				total_size += samples.rows();

				for(int j = 0; j < n_local_bases; ++j)
				{
					const Basis &b=bs.bases[j];

					if(std::find(bounday_nodes.begin(), bounday_nodes.end(), local_assembler_.size() * b.global_index()) != bounday_nodes.end()) //pt found
					{
						if(global_index_to_col.find( b.global_index() ) == global_index_to_col.end())
						{
							global_index_to_col[b.global_index()] = index++;
							indices.push_back(b.global_index());
						}
					}
				}
			}

			// Eigen::MatrixXd global_mat = Eigen::MatrixXd::Zero(total_size, indices.size());
			Eigen::MatrixXd global_rhs = Eigen::MatrixXd::Zero(total_size, local_assembler_.size());

			const long buffer_size = total_size * long(indices.size());
			std::vector< Eigen::Triplet<double> > entries, entries_t;
			entries.reserve(buffer_size);
			entries_t.reserve(buffer_size);

			index = 0;

			int global_counter = 0;

			for(int e = 0; e < n_el; ++e)
			{
				bool has_samples = sample_boundary(mesh.is_volume(), local_boundary[e], resolution, false, samples);

				if(!has_samples)
					continue;

				const ElementBases &bs = bases[e];
				const int n_local_bases = int(bs.bases.size());

				Eigen::MatrixXd mapped = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());

				for(int j = 0; j < n_local_bases; ++j)
				{
					const Basis &b=bs.bases[j];

					b.basis(samples, tmp);

					// if(std::find(bounday_nodes.begin(), bounday_nodes.end(), b.global_index()) != bounday_nodes.end()) //pt found
					auto item = global_index_to_col.find(b.global_index());
					if(item != global_index_to_col.end()){
						for(int k = 0; k < int(tmp.size()); ++k)
						{
							entries.push_back(Eigen::Triplet<double>(global_counter+k, item->second, tmp(k)));
							entries_t.push_back(Eigen::Triplet<double>(item->second, global_counter+k, tmp(k)));
						}
						// global_mat.block(global_counter, item->second, tmp.size(), 1) = tmp;
					}

					for (long k = 0; k < tmp.rows(); ++k){
						mapped.row(k) += tmp(k,0) * b.node();
					}
				}

				// std::cout<<samples<<"\n"<<std::endl;

				// igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
				// viewer.data.add_points(mapped, Eigen::MatrixXd::Constant(mapped.rows(), 3, e/(n_el -1.)));

				problem.bc(mapped, rhs_fun);
				global_rhs.block(global_counter, 0, rhs_fun.rows(), rhs_fun.cols()) = rhs_fun;
				global_counter += rhs_fun.rows();
			}

			assert(global_counter == total_size);

			Eigen::SparseMatrix<double> mat(int(total_size), int(indices.size()));
			mat.setFromTriplets(entries.begin(), entries.end());

			Eigen::SparseMatrix<double> mat_t(int(indices.size()), int(total_size));
			mat_t.setFromTriplets(entries_t.begin(), entries_t.end());

			Eigen::SparseMatrix<double> A = mat_t * mat;
			Eigen::MatrixXd b = mat_t * global_rhs;



			Eigen::MatrixXd coeffs;
			// if(A.rows() > 2000)
			{
				// Eigen::BiCGSTAB< Eigen::SparseMatrix<double> > solver;
				// coeffs = solver.compute(A).solve(b);
				Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
				coeffs = solver.compute(A).solve(b);
			}
			// else
				// coeffs = A.ldlt().solve(b);

			// std::cout<<global_mat<<"\n"<<std::endl;
			// std::cout<<coeffs<<"\n"<<std::endl;
			// std::cout<<global_rhs<<"\n\n\n"<<std::endl;
			for(long i = 0; i < coeffs.rows(); ++i){
				// problem.bc(mesh.pts.row(indices[i]), rhs_fun);

				// std::cout<<indices[i]<<" "<<coeffs(i)<<" vs " <<rhs_fun<<std::endl;
				for(int d = 0; d < local_assembler_.size(); ++d){
					rhs(indices[i]*local_assembler_.size()+d) = coeffs(i, d);
				}
			}
		}

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;

		bool sample_boundary(const bool is_volume, const LocalBoundary &local_boundary, const int resolution_one_d, const bool skip_computation, Eigen::MatrixXd &samples) const
		{
			if(is_volume)
			{
				assert(false);
			}
			else
			{
				const int resolution = resolution_one_d;

				// std::cout<<local_boundary.flags()<<std::endl;

				const bool is_right_boundary  = local_boundary.is_right_boundary();
				const bool is_bottom_boundary = local_boundary.is_bottom_boundary();
				const bool is_left_boundary   = local_boundary.is_left_boundary();
				const bool is_top_boundary    = local_boundary.is_top_boundary();

				return QuadBoundarySampler::sample(is_right_boundary, is_bottom_boundary, is_left_boundary, is_top_boundary, resolution_one_d, skip_computation, samples);
			}

			return true;
		}
	};
}

#endif //ASSEMBLER_HPP