#ifndef ASSEMBLER_HPP
#define ASSEMBLER_HPP

#include "Mesh.hpp"

#include "ElementAssemblyValues.hpp"
#include "Problem.hpp"
#include "Basis.hpp"
#include "LocalBoundary.hpp"

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
		template<class Quadrature>
		void compute_assembly_values(const bool is_volume, const Quadrature &quadrature, const std::vector< std::vector<Basis> > &bases, std::vector< ElementAssemblyValues > &values)
		{
			values.resize(bases.size());

			for(std::size_t i = 0; i < bases.size(); ++i)
			{
				const std::vector<Basis> &bs = bases[i];
				ElementAssemblyValues &vals = values[i];
				vals.basis_values.resize(bs.size());
				vals.quadrature = quadrature;

				Eigen::MatrixXd mval = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

				Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
				Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());
				Eigen::MatrixXd dzmv = Eigen::MatrixXd::Zero(quadrature.points.rows(), quadrature.points.cols());

				const int n_local_bases = int(bs.size());
				for(int j = 0; j < n_local_bases; ++j)
				{
					const Basis &b=bs[j];
					AssemblyValues &val = vals.basis_values[j];

					val.global_index = b.global_index();


					b.basis(quadrature.points, val.val);
					b.grad(quadrature.points, val.grad);

					for (long k = 0; k < val.val.rows(); ++k){
						mval.row(k) += val.val(k,0)    * b.node();

						dxmv.row(k) += val.grad(k,0) * b.node();
						dymv.row(k) += val.grad(k,1) * b.node();
						if(is_volume)
							dzmv.row(k) += val.grad(k,2) * b.node();
					}
				}

				if(is_volume)
					vals.finalize(mval, dxmv, dymv, dzmv);
				else
					vals.finalize(mval, dxmv, dymv);
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

						local_assembler_.assemble(gradi, gradj, da, local_val);

						const auto stiffness_val = local_val.array().colwise().sum();
						assert(stiffness_val.size() == local_assembler_.size() * local_assembler_.size());

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

		void bc(const std::vector< std::vector<Basis> > &bases, const Mesh &mesh, const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution,  const Problem &problem, Eigen::MatrixXd &rhs) const
		{
			const int n_el=int(bases.size());

			Eigen::MatrixXd samples, tmp, rhs_fun;

			int index = 0;
			std::vector<int> indices; indices.reserve(n_el*10);
			std::map<int, int> global_index_to_col;

			long total_size = 0;

			for(int e = 0; e < n_el; ++e)
			{
				bool has_samples = sample_boundary(e, mesh, local_boundary[e], resolution, true, samples);

				if(!has_samples)
					continue;

				const std::vector<Basis> &bs = bases[e];
				const int n_local_bases = int(bs.size());

				total_size += samples.rows();

				for(int j = 0; j < n_local_bases; ++j)
				{
					const Basis &b=bs[j];

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
				bool has_samples = sample_boundary(e, mesh, local_boundary[e], resolution, false, samples);

				if(!has_samples)
					continue;

				const std::vector<Basis> &bs = bases[e];
				const int n_local_bases = int(bs.size());

				Eigen::MatrixXd mapped = Eigen::MatrixXd::Zero(samples.rows(), samples.cols());

				for(int j = 0; j < n_local_bases; ++j)
				{
					const Basis &b=bs[j];

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

				igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
				viewer.data.add_points(mapped, Eigen::MatrixXd::Constant(mapped.rows(), 3, e/(n_el -1.)));

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

		bool sample_boundary(const int el_index, const Mesh &mesh, const LocalBoundary &local_boundary, const int resolution_one_d, const bool skip_computation, Eigen::MatrixXd &samples) const
		{
			if(mesh.is_volume())
			{
				assert(false);
				// const int resolution = resolution_one_d *resolution_one_d;

				// const int n_x = mesh.n_x;
				// const int n_y = mesh.n_y;
				// const int n_z = mesh.n_z;

				// const bool has_left = (el(0) % ((n_x + 1)*(n_y + 1))) % (n_x + 1) != 0;
				// const bool has_right = (el(2) % ((n_x + 1)*(n_y + 1))) % (n_x + 1) != n_x;

				// const bool has_top = (el(0) % ((n_x + 1)*(n_y + 1))) / (n_x + 1) != 0;
				// const bool has_bottom = (el(2) % ((n_x + 1)*(n_y + 1))) / (n_x + 1) != n_y;

				// const bool has_front = el(4) < (n_x + 1) * (n_y + 1) * n_z;
				// const bool has_back = el(0) >= (n_x + 1) * (n_y + 1);

				// int n = 0;
				// if(!has_left) n+=resolution;
				// if(!has_right) n+=resolution;
				// if(!has_bottom) n+=resolution;
				// if(!has_top) n+=resolution;
				// if(!has_front) n+=resolution;
				// if(!has_back) n+=resolution;

				// if(n <= 0) return false;

				// const Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(resolution_one_d, 0, 1);

				// Eigen::MatrixXd tx(resolution, 1);
				// Eigen::MatrixXd ty(resolution, 1);

				// for(int i = 0; i < resolution_one_d; ++i)
				// {
				// 	for(int j = 0; j < resolution_one_d; ++j)
				// 	{
				// 		tx(i * resolution_one_d + j) = t(i);
				// 		ty(i * resolution_one_d + j) = t(j);
				// 	}
				// }

				// samples.resize(n, 3);
				// n = 0;

				// if(!has_left){
				// 	samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
				// 	samples.block(n, 1, resolution, 1) = tx;
				// 	samples.block(n, 2, resolution, 1) = ty;

				// 	n += resolution;
				// }
				// if(!has_right){
				// 	samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
				// 	samples.block(n, 1, resolution, 1) = tx;
				// 	samples.block(n, 2, resolution, 1) = ty;

				// 	n += resolution;
				// }



				// if(!has_bottom){
				// 	samples.block(n, 0, resolution, 1) = tx;
				// 	samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
				// 	samples.block(n, 2, resolution, 1) = ty;

				// 	n += resolution;
				// }
				// if(!has_top){
				// 	samples.block(n, 0, resolution, 1) = tx;
				// 	samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
				// 	samples.block(n, 2, resolution, 1) = ty;

				// 	n += resolution;
				// }


				// if(!has_front){
				// 	samples.block(n, 0, resolution, 1) = tx;
				// 	samples.block(n, 1, resolution, 1) = ty;
				// 	samples.block(n, 2, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);

				// 	n += resolution;
				// }
				// if(!has_back){
				// 	samples.block(n, 0, resolution, 1) = tx;
				// 	samples.block(n, 1, resolution, 1) = ty;
				// 	samples.block(n, 2, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);

				// 	n += resolution;
				// }



				// // std::cout<<samples<<std::endl;
				// // igl::viewer::Viewer &viewer = State::state().viewer;
				// // viewer.data.add_points(samples, Eigen::MatrixXd::Zero(samples.rows(), 3));
				// // viewer.launch();

				// assert(long(n) == samples.rows());
			}
			else
			{
				const int resolution = resolution_one_d;

				// std::cout<<local_boundary.flags()<<std::endl;

				const bool is_right_boundary  = local_boundary.is_right_boundary();
				const bool is_bottom_boundary = local_boundary.is_bottom_boundary();
				const bool is_left_boundary   = local_boundary.is_left_boundary();
				const bool is_top_boundary    = local_boundary.is_top_boundary();

				int n = 0;
				if(is_right_boundary) n+=resolution;
				if(is_left_boundary) n+=resolution;
				if(is_top_boundary) n+=resolution;
				if(is_bottom_boundary) n+=resolution;

				if(n <= 0) return false;

				samples.resize(n, 2);
				if(skip_computation) return true;

				const Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(resolution, 0, 1);

				n = 0;
				if(is_right_boundary){
					samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);
					samples.block(n, 1, resolution, 1) = t;

					n += resolution;
				}

				if(is_top_boundary){
					samples.block(n, 0, resolution, 1) = t;
					samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Zero(resolution, 1);

					n += resolution;
				}

				if(is_left_boundary){
					samples.block(n, 0, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);
					samples.block(n, 1, resolution, 1) = t;

					n += resolution;
				}

				if(is_bottom_boundary){
					samples.block(n, 0, resolution, 1) = t;
					samples.block(n, 1, resolution, 1) = Eigen::MatrixXd::Ones(resolution, 1);

					n += resolution;
				}

				assert(long(n) == samples.rows());
			}

			return true;
		}
	};
}

#endif //ASSEMBLER_HPP