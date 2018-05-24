#include "RhsAssembler.hpp"

#include "BoundarySampler.hpp"

#include "LinearSolver.hpp"

#include <Eigen/Sparse>

#include <iostream>
#include <map>
#include <memory>

#include "UIState.hpp"

namespace poly_fem
{
	RhsAssembler::RhsAssembler(const Mesh &mesh, const int n_basis, const int size, const std::vector< ElementBases > &bases, const std::vector< ElementBases > &gbases, const std::string &formulation, const Problem &problem)
	: mesh_(mesh), n_basis_(n_basis), size_(size), bases_(bases), gbases_(gbases), formulation_(formulation), problem_(problem)
	{ }

	void RhsAssembler::assemble(Eigen::MatrixXd &rhs) const
	{
		rhs = Eigen::MatrixXd::Zero(n_basis_ * size_, 1);
		Eigen::MatrixXd rhs_fun;

		const int n_elements = int(bases_.size());
		for(int e = 0; e < n_elements; ++e)
		{
			ElementAssemblyValues vals;
			vals.compute(e, mesh_.is_volume(), bases_[e], gbases_[e]);

			const Quadrature &quadrature = vals.quadrature;


			problem_.rhs(formulation_, vals.val, rhs_fun);

			for(int d = 0; d < size_; ++d)
				rhs_fun.col(d) = rhs_fun.col(d).array() * vals.det.array() * quadrature.weights.array();

			const int n_loc_bases_ = int(vals.basis_values.size());
			for(int i = 0; i < n_loc_bases_; ++i)
			{
				const AssemblyValues &v = vals.basis_values[i];

				for(int d = 0; d < size_; ++d)
				{
					const double rhs_value = (rhs_fun.col(d).array() * v.val.array()).sum();
					for(std::size_t ii = 0; ii < v.global.size(); ++ii)
						rhs(v.global[ii].index*size_+d) +=  rhs_value * v.global[ii].val;
				}
			}
		}
	}

	void RhsAssembler::set_bc(const std::vector< LocalBoundary > &local_boundary, const std::vector<int> &bounday_nodes, const int resolution, const std::vector< LocalBoundary > &local_neumann_boundary, Eigen::MatrixXd &rhs) const
	{
		const int n_el=int(bases_.size());

		Eigen::MatrixXd samples, tmp, gtmp, rhs_fun;
		Eigen::VectorXi global_primitive_ids;

		int index = 0;
		std::vector<int> indices; indices.reserve(n_el*10);
		std::map<int, int> global_index_to_col;

		long total_size = 0;

		for(const auto &lb : local_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = sample_boundary(lb, resolution, true, samples, global_primitive_ids);

			if(!has_samples)
				continue;

			const ElementBases &bs = bases_[e];
			const int n_local_bases = int(bs.bases.size());

			total_size += samples.rows();

			for(int j = 0; j < n_local_bases; ++j)
			{
				const Basis &b=bs.bases[j];

				for(std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					//pt found
					if(std::find(bounday_nodes.begin(), bounday_nodes.end(), size_ * b.global()[ii].index) != bounday_nodes.end())
					{
						if(global_index_to_col.find( b.global()[ii].index ) == global_index_to_col.end())
						{
							global_index_to_col[b.global()[ii].index] = index++;
							indices.push_back(b.global()[ii].index);
							assert(indices.size() == index);
						}
					}
				}
			}
		}

		// Eigen::MatrixXd global_mat = Eigen::MatrixXd::Zero(total_size, indices.size());
		Eigen::MatrixXd global_rhs = Eigen::MatrixXd::Zero(total_size, size_);

		const long buffer_size = total_size * long(indices.size());
		std::vector< Eigen::Triplet<double> > entries, entries_t;
		// entries.reserve(buffer_size);
		// entries_t.reserve(buffer_size);

		index = 0;

		int global_counter = 0;

		for(const auto &lb : local_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = sample_boundary(lb, resolution, false, samples, global_primitive_ids);

			if(!has_samples)
				continue;

			const ElementBases &bs = bases_[e];
			const ElementBases &gbs = gbases_[e];
			const int n_local_bases = int(bs.bases.size());

			Eigen::MatrixXd mapped;
			gbs.eval_geom_mapping(samples, mapped);

			bs.evaluate_bases(samples, tmp);
			for(int j = 0; j < n_local_bases; ++j)
			{
				const Basis &b=bs.bases[j];

				for(std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					auto item = global_index_to_col.find(b.global()[ii].index);
					if(item != global_index_to_col.end()){
						for(int k = 0; k < int(tmp.rows()); ++k)
						{
							entries.push_back(Eigen::Triplet<double>(global_counter+k, item->second, tmp(k, j) * b.global()[ii].val));
							entries_t.push_back(Eigen::Triplet<double>(item->second, global_counter+k, tmp(k, j) * b.global()[ii].val));
						}
						// global_mat.block(global_counter, item->second, tmp.size(), 1) = tmp;
					}
				}
			}

			problem_.bc(mesh_, global_primitive_ids, mapped, rhs_fun);
			global_rhs.block(global_counter, 0, rhs_fun.rows(), rhs_fun.cols()) = rhs_fun;
			global_counter += rhs_fun.rows();

			// igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
			// viewer.data.add_points(mapped, Eigen::MatrixXd::Constant(1, 3, 0));

			//Eigen::MatrixXd asd(mapped.rows(), 3);
			//asd.col(0)=mapped.col(0);
			//asd.col(1)=mapped.col(1);
			//asd.col(2)=rhs_fun;
			//viewer.data.add_points(asd, Eigen::MatrixXd::Constant(1, 3, 0));
		}

		assert(global_counter == total_size);

		{
			Eigen::SparseMatrix<double> mat(int(total_size), int(indices.size()));
			mat.setFromTriplets(entries.begin(), entries.end());

			Eigen::SparseMatrix<double> mat_t(int(indices.size()), int(total_size));
			mat_t.setFromTriplets(entries_t.begin(), entries_t.end());

			Eigen::SparseMatrix<double> A = mat_t * mat;
			Eigen::MatrixXd b = mat_t * global_rhs;


			Eigen::MatrixXd coeffs(b.rows(), b.cols());

			json params = {
			{"mtype", -2}, // matrix type for Pardiso (2 = SPD)
			// {"max_iter", 0}, // for iterative solvers
			// {"tolerance", 1e-9}, // for iterative solvers
			};

			// auto solver = LinearSolver::create("", "");
			auto solver = LinearSolver::create(LinearSolver::defaultSolver(), LinearSolver::defaultPrecond());
			solver->setParameters(params);
			solver->analyzePattern(A);
			solver->factorize(A);
			for(long i = 0; i < b.cols(); ++i){
				solver->solve(b.col(i), coeffs.col(i));
			}
			std::cout<<"RHS solve error "<< (A*coeffs-b).norm()<<std::endl;

			for(long i = 0; i < coeffs.rows(); ++i){
				for(int d = 0; d < size_; ++d){
					rhs(indices[i]*size_+d) = coeffs(i, d);
				}
			}
		}



		//Neumann
		Eigen::MatrixXd points;
		Eigen::VectorXd weights;
		for(const auto &lb : local_neumann_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = boundary_quadrature(lb, resolution/3, true, points, weights, global_primitive_ids);

			if(!has_samples)
				continue;

			const ElementBases &bs = bases_[e];
			for(int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh_);

				for(long n = 0; n < nodes.size(); ++n){
					const auto &b = bs.bases[nodes(n)];
					for(size_t g = 0; g < b.global().size(); ++g){
						for(int d = 0; d < size_; ++d){
							rhs(b.global()[g].index*size_+d) = 0;
						}
					}
				}
			}
		}

		for(const auto &lb : local_neumann_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = boundary_quadrature(lb, resolution/3, false, points, weights, global_primitive_ids);

			if(!has_samples)
				continue;

			const ElementBases &gbs = gbases_[e];
			const ElementBases &bs = bases_[e];
			Eigen::MatrixXd mapped;
			gbs.eval_geom_mapping(points, mapped);
			problem_.neumann_bc(mesh_, global_primitive_ids, mapped, rhs_fun);


			ElementAssemblyValues vals;
			vals.compute(e, mesh_.is_volume(), points, bases_[e], gbases_[e]);
			problem_.neumann_bc(mesh_, global_primitive_ids, vals.val, rhs_fun);
			// problem_.neumann_bc(mesh_, global_primitive_ids, mapped, rhs_fun);

			for(int d = 0; d < size_; ++d)
				rhs_fun.col(d) = rhs_fun.col(d).array() * weights.array();

			// const int n_loc_bases_ = int(vals.basis_values.size());
			// for(int i = 0; i < n_loc_bases_; ++i)
			// {
			// 	const AssemblyValues &v = vals.basis_values[i];

			// 	for(int d = 0; d < size_; ++d)
			// 	{
			// 		const double rhs_value = (rhs_fun.col(d).array() * v.val.array()).sum();
			// 		for(std::size_t ii = 0; ii < v.global.size(); ++ii)
			// 			rhs(v.global[ii].index*size_+d) +=  rhs_value * v.global[ii].val;
			// 	}
			// }

			for(int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, mesh_);

				for(long n = 0; n < nodes.size(); ++n)
				{
					// const auto &b = bs.bases[nodes(n)];
					const AssemblyValues &v = vals.basis_values[nodes(n)];
					for(int d = 0; d < size_; ++d)
					{
						const double rhs_value = (rhs_fun.col(d).array() * v.val.array()).sum();

						for(size_t g = 0; g < v.global.size(); ++g)
						{
							rhs(v.global[g].index*size_+d) +=  rhs_value * v.global[g].val;
						}
					}
				}
			}
		}
	}

	double RhsAssembler::compute_energy(const Eigen::MatrixXd &displacement) const
	{
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> local_displacement(size_);

		double res = 0;
		Eigen::MatrixXd forces;

		const int n_bases = int(bases_.size());
		for(int e = 0; e < n_bases; ++e)
		{
			ElementAssemblyValues vals;
			vals.compute(e, mesh_.is_volume(), bases_[e], gbases_[e]);

			const Quadrature &quadrature = vals.quadrature;
			const Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();


			problem_.rhs(formulation_, vals.val, forces);
			assert(forces.rows() == da.size());
			assert(forces.cols() == size_);

			for(long p = 0; p < da.size(); ++p)
			{
				local_displacement.setZero();

				for(size_t i = 0; i < vals.basis_values.size(); ++i)
				{
					const auto &bs = vals.basis_values[i];
					assert(bs.val.size() == da.size());
					const double b_val = bs.val(p);

					for(int d = 0; d < size_; ++d)
					{
						for(std::size_t ii = 0; ii < bs.global.size(); ++ii)
						{
							local_displacement(d) += (bs.global[ii].val * b_val) * displacement(bs.global[ii].index*size_ + d);
						}
					}
				}

				for(int d = 0; d < size_; ++d)
					res += forces(p, d) * local_displacement(d) * da(p);
			}
		}

		return res;
	}

	bool RhsAssembler::boundary_quadrature(const LocalBoundary &local_boundary, const int order, const bool skip_computation, Eigen::MatrixXd &points, Eigen::VectorXd &weights, Eigen::VectorXi &global_primitive_ids) const
	{
		points.resize(0, 0);
		weights.resize(0);
		global_primitive_ids.resize(0);

		for(int i = 0; i < local_boundary.size(); ++i)
		{
			const int gid = local_boundary.global_primitive_id(i);
			Eigen::MatrixXd tmp_p;
			Eigen::VectorXd tmp_w;
			switch(local_boundary.type())
			{
				case BoundaryType::TriLine:	 BoundarySampler::quadrature_for_tri_edge(local_boundary[i], order, tmp_p, tmp_w);  tmp_w *= mesh_.edge_length(gid); break;
				case BoundaryType::QuadLine: BoundarySampler::quadrature_for_quad_edge(local_boundary[i], order, tmp_p, tmp_w); tmp_w *= mesh_.edge_length(gid); break;
				case BoundaryType::Quad: 	 BoundarySampler::quadrature_for_quad_face(local_boundary[i], order, tmp_p, tmp_w); tmp_w *= mesh_.quad_area(gid); break;
				case BoundaryType::Tri: 	 BoundarySampler::quadrature_for_tri_face(local_boundary[i], order, tmp_p, tmp_w);  tmp_w *= mesh_.tri_area(gid); break;
				case BoundaryType::Invalid:  assert(false); break;
				default: assert(false);
			}

			points.conservativeResize(points.rows() + tmp_p.rows(), tmp_p.cols());
			points.bottomRows(tmp_p.rows()) = tmp_p;

			weights.conservativeResize(weights.rows() + tmp_w.rows(), tmp_w.cols());
			weights.bottomRows(tmp_w.rows()) = tmp_w;

			global_primitive_ids.conservativeResize(global_primitive_ids.rows() + tmp_p.rows());
			global_primitive_ids.bottomRows(tmp_p.rows()).setConstant(gid);
		}

		assert(points.rows() == global_primitive_ids.size());
		assert(weights.size() == global_primitive_ids.size());

		return true;
	}


	bool RhsAssembler::sample_boundary(const LocalBoundary &local_boundary, const int n_samples, const bool skip_computation, Eigen::MatrixXd &samples, Eigen::VectorXi &global_primitive_ids) const
	{
		samples.resize(0, 0);
		global_primitive_ids.resize(0);

		for(int i = 0; i < local_boundary.size(); ++i)
		{
			Eigen::MatrixXd tmp;
			switch(local_boundary.type())
			{
				case BoundaryType::TriLine:	 BoundarySampler::sample_parametric_tri_edge(local_boundary[i], n_samples, tmp); break;
				case BoundaryType::QuadLine: BoundarySampler::sample_parametric_quad_edge(local_boundary[i], n_samples, tmp); break;
				case BoundaryType::Quad: 	 BoundarySampler::sample_parametric_quad_face(local_boundary[i], n_samples, tmp); break;
				case BoundaryType::Tri: 	 BoundarySampler::sample_parametric_tri_face(local_boundary[i], n_samples, tmp); break;
				case BoundaryType::Invalid:  assert(false); break;
				default: assert(false);
			}

			samples.conservativeResize(samples.rows() + tmp.rows(), tmp.cols());
			global_primitive_ids.conservativeResize(global_primitive_ids.rows() + tmp.rows());
			samples.bottomRows(tmp.rows()) = tmp;
			global_primitive_ids.bottomRows(tmp.rows()).setConstant(local_boundary.global_primitive_id(i));
		}

		assert(samples.rows() == global_primitive_ids.size());


		return true;
	}

}
