#include "RhsAssembler.hpp"

#include "QuadBoundarySampler.hpp"

#include <iostream>

#include "UIState.hpp"

namespace poly_fem
{

	void RhsAssembler::assemble(const int n_basis, const int size, const std::vector< ElementAssemblyValues > &values, const std::vector< ElementAssemblyValues > &geom_values, const Problem &problem, Eigen::MatrixXd &rhs) const
	{
		rhs = Eigen::MatrixXd::Zero(n_basis * size, 1);
		Eigen::MatrixXd rhs_fun;

		const int n_values = int(values.size());
		for(int e = 0; e < n_values; ++e)
		{
			const ElementAssemblyValues &vals  = values[e];
			const ElementAssemblyValues &gvals = geom_values[e];

			problem.rhs(gvals.val, rhs_fun);

				// std::cout<<e<<"\n"<<gvals.val<<"\n"<<rhs_fun<<"\n\n"<<std::endl;

			for(int d = 0; d < size; ++d)
				rhs_fun.col(d) = rhs_fun.col(d).array() * gvals.det.array() * vals.quadrature.weights.array();

				// std::cout<<"after:\n"<<rhs_fun<<std::endl;

			const int n_loc_bases = int(vals.basis_values.size());
			for(int i = 0; i < n_loc_bases; ++i)
			{
				const AssemblyValues &v = vals.basis_values[i];

				for(int d = 0; d < size; ++d)
				{
					const double rhs_value = (rhs_fun.col(d).array() * v.val.array()).sum();
					for(std::size_t ii = 0; ii < v.global.size(); ++ii)
						rhs(v.global[ii].index*size+d) +=  rhs_value * v.global[ii].val;
				}
			}
		}
	}

	void RhsAssembler::set_bc(const int size, const std::vector< ElementBases > &bases, const std::vector< ElementBases > &geom_bases, const bool is_volume, const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution,  const Problem &problem, Eigen::MatrixXd &rhs) const
	{
		const int n_el=int(bases.size());

		Eigen::MatrixXd samples, tmp, gtmp, rhs_fun;

		int index = 0;
		std::vector<int> indices; indices.reserve(n_el*10);
		std::map<int, int> global_index_to_col;

		long total_size = 0;

		for(int e = 0; e < n_el; ++e)
		{
			bool has_samples = sample_boundary(is_volume, local_boundary[e], resolution, true, samples);

			if(!has_samples)
				continue;

			const ElementBases &bs = bases[e];
			const int n_local_bases = int(bs.bases.size());

			total_size += samples.rows();

			for(int j = 0; j < n_local_bases; ++j)
			{
				const Basis &b=bs.bases[j];

				for(std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					//pt found
					if(std::find(bounday_nodes.begin(), bounday_nodes.end(), size * b.global()[ii].index) != bounday_nodes.end())
					{
						if(global_index_to_col.find( b.global()[ii].index ) == global_index_to_col.end())
						{
							global_index_to_col[b.global()[ii].index] = index++;
							indices.push_back(b.global()[ii].index);
						}
					}
				}
			}
		}

			// Eigen::MatrixXd global_mat = Eigen::MatrixXd::Zero(total_size, indices.size());
		Eigen::MatrixXd global_rhs = Eigen::MatrixXd::Zero(total_size, size);

		const long buffer_size = total_size * long(indices.size());
		std::vector< Eigen::Triplet<double> > entries, entries_t;
		entries.reserve(buffer_size);
		entries_t.reserve(buffer_size);

		index = 0;

		int global_counter = 0;

		for(int e = 0; e < n_el; ++e)
		{
			bool has_samples = sample_boundary(is_volume, local_boundary[e], resolution, false, samples);

			if(!has_samples)
				continue;

			const ElementBases &bs = bases[e];
			const ElementBases &gbs = geom_bases[e];
			const int n_local_bases = int(bs.bases.size());

			Eigen::MatrixXd mapped;
			gbs.eval_geom_mapping(samples, mapped);

			for(int j = 0; j < n_local_bases; ++j)
			{
				const Basis &b=bs.bases[j];

				b.basis(samples, tmp);
				for(std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					auto item = global_index_to_col.find(b.global()[ii].index);
					if(item != global_index_to_col.end()){
						for(int k = 0; k < int(tmp.size()); ++k)
						{
							entries.push_back(Eigen::Triplet<double>(global_counter+k, item->second, tmp(k)));
							entries_t.push_back(Eigen::Triplet<double>(item->second, global_counter+k, tmp(k)));
						}
						// global_mat.block(global_counter, item->second, tmp.size(), 1) = tmp;
					}
				}
			}

				// std::cout<<samples<<"\n"<<std::endl;

				// igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
				// viewer.data.add_points(mapped, Eigen::MatrixXd::Constant(mapped.rows(), 3, 0));

				// std::cout<<mapped<<std::endl;

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
			for(int d = 0; d < size; ++d){
				rhs(indices[i]*size+d) = coeffs(i, d);
			}
		}
	}


	bool RhsAssembler::sample_boundary(const bool is_volume, const LocalBoundary &local_boundary, const int resolution_one_d, const bool skip_computation, Eigen::MatrixXd &samples) const
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

}
