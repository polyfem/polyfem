#include "State.hpp"

#include "Mesh2D.hpp"
#include "Mesh3D.hpp"

#include "QuadBasis2d.hpp"
#include "TriBasis2d.hpp"

#include "HexBasis3d.hpp"
#include "TetBasis3d.hpp"

#include "SplineBasis2d.hpp"
#include "SplineBasis3d.hpp"

#include "QuadBoundarySampler.hpp"
#include "HexBoundarySampler.hpp"

#include "PolygonalBasis2d.hpp"
#include "PolygonalBasis3d.hpp"

#include "Assembler.hpp"
#include "RhsAssembler.hpp"

#include "Laplacian.hpp"
#include "LinearElasticity.hpp"

#include "LinearSolver.hpp"
#include "FEMSolver.hpp"

#include "json.hpp"

#include "CustomSerialization.hpp"
#include "VTUWriter.hpp"

#include <igl/Timer.h>
#include <igl/serialize.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

#include <unsupported/Eigen/SparseExtra>

#include <iostream>
#include <algorithm>
#include <memory>


using namespace Eigen;


namespace poly_fem
{

	namespace
	{
		double compute_mesh_size(const Mesh &mesh, const std::vector< ElementBases > &bases, const int n_samples)
		{
			double mesh_size = 0;
			Eigen::MatrixXd samples, mapped, p0, p1, p;

			if(true || mesh.is_simplicial())
			{
				mesh.get_edges(p0, p1);
				p = p0-p1;
				std::cout << std::endl;
				std::cout << "hmin: " << p.rowwise().norm().minCoeff() << std::endl;
				std::cout << "hmax: " << p.rowwise().norm().maxCoeff() << std::endl;
				std::cout << "havg: " << p.rowwise().norm().mean() << std::endl;
				return p.rowwise().norm().maxCoeff();
			}

			if(mesh.is_volume())
			{
				samples.resize(12*n_samples, 3);
				const Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 1);

				//X
				int ii = 0;
				samples.block(ii*n_samples, 0, n_samples, 1) = t;
				samples.block(ii*n_samples, 1, n_samples, 1).setZero();
				samples.block(ii*n_samples, 2, n_samples, 1).setZero();

				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1) = t;
				samples.block(ii*n_samples, 1, n_samples, 1).setOnes();
				samples.block(ii*n_samples, 2, n_samples, 1).setZero();

				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1) = t;
				samples.block(ii*n_samples, 1, n_samples, 1).setZero();
				samples.block(ii*n_samples, 2, n_samples, 1).setOnes();

				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1) = t;
				samples.block(ii*n_samples, 1, n_samples, 1).setOnes();
				samples.block(ii*n_samples, 2, n_samples, 1).setOnes();

				//Y
				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1).setZero();
				samples.block(ii*n_samples, 1, n_samples, 1) = t;
				samples.block(ii*n_samples, 2, n_samples, 1).setZero();

				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1).setOnes();
				samples.block(ii*n_samples, 1, n_samples, 1) = t;
				samples.block(ii*n_samples, 2, n_samples, 1).setZero();

				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1).setZero();
				samples.block(ii*n_samples, 1, n_samples, 1) = t;
				samples.block(ii*n_samples, 2, n_samples, 1).setOnes();

				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1).setOnes();
				samples.block(ii*n_samples, 1, n_samples, 1) = t;
				samples.block(ii*n_samples, 2, n_samples, 1).setOnes();

				//Z
				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1).setZero();
				samples.block(ii*n_samples, 1, n_samples, 1).setZero();
				samples.block(ii*n_samples, 2, n_samples, 1) = t;

				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1).setOnes();
				samples.block(ii*n_samples, 1, n_samples, 1).setZero();
				samples.block(ii*n_samples, 2, n_samples, 1) = t;

				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1).setZero();
				samples.block(ii*n_samples, 1, n_samples, 1).setOnes();
				samples.block(ii*n_samples, 2, n_samples, 1) = t;

				++ii;
				samples.block(ii*n_samples, 0, n_samples, 1).setOnes();
				samples.block(ii*n_samples, 1, n_samples, 1).setOnes();
				samples.block(ii*n_samples, 2, n_samples, 1) = t;


				for(std::size_t i = 0; i < bases.size(); ++i){
					if(mesh.is_polytope(i)) continue;

					bases[i].eval_geom_mapping(samples, mapped);

					for(int j = 0; j < 12; ++j)
					{
						double current_edge = 0;
						for(int k = 0; k < n_samples-1; ++k){
							p0 = mapped.row(j*n_samples + k);
							p1 = mapped.row(j*n_samples + k+1);
							p = p0-p1;

							current_edge += p.norm();
						}

						mesh_size = std::max(current_edge, mesh_size);
					}
				}
			}
			else
			{
				QuadBoundarySampler::sample(true, true, true, true, n_samples, false, samples);

				for(std::size_t i = 0; i < bases.size(); ++i){
					if(mesh.is_polytope(i)) continue;

					bases[i].eval_geom_mapping(samples, mapped);

					for(int j = 0; j < 4; ++j)
					{
						double current_edge = 0;
						for(int k = 0; k < n_samples-1; ++k){
							p0 = mapped.row(j*n_samples + k);
							p1 = mapped.row(j*n_samples + k+1);
							p = p0-p1;

							current_edge += p.norm();
						}

						mesh_size = std::max(current_edge, mesh_size);
					}
				}
			}

			return mesh_size;
		}
	}

	void State::save_json(std::ostream &out)
	{
		std::cout<<"Saving json..."<<std::flush;
		using json = nlohmann::json;
		json j;

		j["quadrature_order"] = quadrature_order;
		j["mesh_path"] = mesh_path;
		j["discr_order"] = discr_order;
		j["harmonic_samples_res"] = harmonic_samples_res;
		j["use_splines"] = use_splines;
		j["iso_parametric"] = iso_parametric;
		j["problem"] = problem.problem_num();
		j["mat_size"] = mat_size;
		j["solver_type"] = solver_type;
		j["precond_type"] = precond_type;
		j["lambda"] = lambda;
		j["mu"] = mu;
		j["refinenemt_location"] = refinenemt_location;

		j["num_boundary_samples"] = n_boundary_samples;
		j["num_refs"] = n_refs;
		j["num_bases"] = n_bases;
		j["num_non_zero"] = nn_zero;
		j["num_flipped"] = n_flipped;
		j["num_dofs"] = num_dofs;

		j["mesh_size"] = mesh_size;

		j["err_l2"] = l2_err;
		j["err_linf"] = linf_err;
		j["err_lp"] = lp_err;

		// j["errors"] = errors;

		j["time_building_basis"] = building_basis_time;
		j["time_loading_mesh"] = loading_mesh_time;
		j["time_computing_assembly_values"] = computing_assembly_values_time;
		j["time_assembling_stiffness_mat"] = assembling_stiffness_mat_time;
		j["time_assigning_rhs"] = assigning_rhs_time;
		j["time_solving"] = solving_time;
		j["time_computing_errors"] = computing_errors_time;

		j["solver_info"] = solver_info;

		j["count_regular"] = regular_count;
		j["count_regular_boundary"] = regular_boundary_count;
		j["count_simple_singular"] = simple_singular_count;
		j["count_multi_singular"] = multi_singular_count;
		j["count_boundary"] = boundary_count;
		j["count_non_regular_boundary"] = non_regular_boundary_count;
		j["count_non_regular"] = non_regular_count;
		j["count_undefined"] = undefined_count;
		j["count_multi_singular_boundary"] = multi_singular_boundary_count;


		out << j.dump(4) << std::endl;

		std::cout<<"done"<<std::endl;
	}


	void State::interpolate_function(const MatrixXd &fun, const MatrixXd &local_pts, MatrixXd &result)
	{
		MatrixXd tmp;

		int actual_dim = 1;
		if(problem.problem_num() == 3)
			actual_dim = mesh->is_volume() ? 3:2;

		result.resize(local_pts.rows() * mesh->n_elements(), actual_dim);

		for(std::size_t i = 0; i < bases.size(); ++i)
		{
			const ElementBases &bs = bases[i];
			bs.evaluate_bases(local_pts, tmp);

			MatrixXd local_res = MatrixXd::Zero(local_pts.rows(), actual_dim);

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];

				// b.basis(local_pts, tmp);
				for(std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					for(int d = 0; d < actual_dim; ++d)
					{
						local_res.col(d) += b.global()[ii].val * tmp.col(j) * fun(b.global()[ii].index*actual_dim + d);
					}
				}
			}

			result.block(i*local_pts.rows(), 0, local_pts.rows(), actual_dim) = local_res;
		}
	}

	void State::load_mesh()
	{
		bases.clear();
		geom_bases.clear();
		// values.clear();
		// geom_values.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		errors.clear();
		polys.clear();
		poly_edge_to_data.clear();
		parent_elements.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		n_bases = 0;



		igl::Timer timer; timer.start();
		std::cout<<"Loading mesh..."<<std::flush;
		mesh = Mesh::create(mesh_path);
		if (!mesh) {
			return;
		}

		// if(!flipped_elements.empty())
		// {
		// 	mesh->compute_elements_tag();
		// 	for(auto el_id : flipped_elements)
		// 		mesh->set_tag(el_id, ElementType::InteriorPolytope);
		// }

		mesh->refine(n_refs, refinenemt_location, parent_elements);

		mesh->normalize();

		mesh->compute_boundary_ids();


		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;
	}

	void State::compute_mesh_stats()
	{
		bases.clear();
		geom_bases.clear();
		// values.clear();
		// geom_values.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		errors.clear();
		polys.clear();
		poly_edge_to_data.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		n_bases = 0;

		// els_tag[4]=ElementType::MultiSingularInteriorCube;
		// els_tag[24]=ElementType::MultiSingularInteriorCube;

		regular_count = 0;
		regular_boundary_count = 0;
		simple_singular_count = 0;
		multi_singular_count = 0;
		boundary_count = 0;
		non_regular_boundary_count = 0;
		non_regular_count = 0;
		undefined_count = 0;
		multi_singular_boundary_count = 0;

		const auto &els_tag = mesh->elements_tag();

		for(std::size_t i = 0; i < els_tag.size(); ++i)
		{
			const ElementType type = els_tag[i];

			switch(type)
			{
				case ElementType::RegularInteriorCube: regular_count++; break;
				case ElementType::RegularBoundaryCube: regular_boundary_count++; break;
				case ElementType::SimpleSingularInteriorCube: simple_singular_count++; break;
				case ElementType::MultiSingularInteriorCube: multi_singular_count++; break;
				case ElementType::SimpleSingularBoundaryCube: boundary_count++; break;
				case ElementType::InterfaceCube:
				case ElementType::MultiSingularBoundaryCube: multi_singular_boundary_count++; break;
				case ElementType::BoundaryPolytope: non_regular_boundary_count++; break;
				case ElementType::InteriorPolytope: non_regular_count++; break;
				case ElementType::Undefined: undefined_count++; break;
			}
		}

		std::cout <<
		"regular_count:\t " << regular_count <<"\n"<<
		"regular_boundary_count:\t " << regular_boundary_count <<"\n"<<
		"simple_singular_count:\t " << simple_singular_count <<"\n"<<
		"multi_singular_count:\t " << multi_singular_count <<"\n"<<
		"singular_boundary_count:\t " << boundary_count <<"\n"<<
		"multi_singular_boundary_count:\t " << multi_singular_boundary_count <<"\n"<<
		"polytope_count:\t " <<  non_regular_count <<"\n"<<
		"polytope_boundary_count:\t " << non_regular_boundary_count <<"\n"<<
		"undefined_count:\t " << undefined_count <<"\n"<<
		"total count:\t " << mesh->n_elements() <<"\n"<<
		std::endl;
	}


	void compute_integral_constraints(
		const Mesh3D &mesh,
		const int n_bases,
		const std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		Eigen::MatrixXd &basis_integrals)
	{
		assert(mesh.is_volume());

		basis_integrals.resize(n_bases, 9);
		basis_integrals.setZero();
		Eigen::MatrixXd rhs(n_bases, 9);
		rhs.setZero();

		const int n_elements = mesh.n_elements();
		for(int e = 0; e < n_elements; ++e) {
		// if (mesh.is_polytope(e)) {
		// 	continue;
		// }
		// ElementAssemblyValues vals = values[e];
		// const ElementAssemblyValues &gvals = gvalues[e];
			ElementAssemblyValues vals;
			vals.compute(e, mesh.is_volume(), bases[e], gbases[e]);


		// Computes the discretized integral of the PDE over the element
			const int n_local_bases = int(vals.basis_values.size());
			for(int j = 0; j < n_local_bases; ++j) {
				const AssemblyValues &v=vals.basis_values[j];
				const double integral_100 = (v.grad_t_m.col(0).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_010 = (v.grad_t_m.col(1).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_001 = (v.grad_t_m.col(2).array() * vals.det.array() * vals.quadrature.weights.array()).sum();

				const double integral_110 = ((vals.val.col(1).array() * v.grad_t_m.col(0).array() + vals.val.col(0).array() * v.grad_t_m.col(1).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_011 = ((vals.val.col(2).array() * v.grad_t_m.col(1).array() + vals.val.col(1).array() * v.grad_t_m.col(2).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_101 = ((vals.val.col(0).array() * v.grad_t_m.col(2).array() + vals.val.col(2).array() * v.grad_t_m.col(0).array()) * vals.det.array() * vals.quadrature.weights.array()).sum();

				const double integral_200 = 2*(vals.val.col(0).array() * v.grad_t_m.col(0).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_020 = 2*(vals.val.col(1).array() * v.grad_t_m.col(1).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
				const double integral_002 = 2*(vals.val.col(2).array() * v.grad_t_m.col(2).array() * vals.det.array() * vals.quadrature.weights.array()).sum();

				const double area = (v.val.array() * vals.det.array() * vals.quadrature.weights.array()).sum();

				for(size_t ii = 0; ii < v.global.size(); ++ii) {
					basis_integrals(v.global[ii].index, 0) += integral_100 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 1) += integral_010 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 2) += integral_001 * v.global[ii].val;

					basis_integrals(v.global[ii].index, 3) += integral_110 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 4) += integral_011 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 5) += integral_101 * v.global[ii].val;

					basis_integrals(v.global[ii].index, 6) += integral_200 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 7) += integral_020 * v.global[ii].val;
					basis_integrals(v.global[ii].index, 8) += integral_002 * v.global[ii].val;

					rhs(v.global[ii].index, 6) += -2.0 * area * v.global[ii].val;
					rhs(v.global[ii].index, 7) += -2.0 * area * v.global[ii].val;
					rhs(v.global[ii].index, 8) += -2.0 * area * v.global[ii].val;
				}
			}
		}

		basis_integrals -= rhs;
	}

	void State::build_basis()
	{
		bases.clear();
		geom_bases.clear();
		// values.clear();
		// geom_values.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		errors.clear();
		polys.clear();
		poly_edge_to_data.clear();
		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		n_bases = 0;


		igl::Timer timer; timer.start();
		std::cout<<"Building "<< (iso_parametric? "isoparametric":"not isoparametric") <<" basis..."<<std::flush;

		local_boundary.clear();
		boundary_nodes.clear();
		std::map<int, InterfaceData> poly_edge_to_data_geom; //temp dummy variable

		if(mesh->is_volume())
		{
			const Mesh3D &tmp_mesh = *dynamic_cast<Mesh3D *>(mesh.get());
			if(use_splines)
			{
				if(!iso_parametric)
					HexBasis3d::build_bases(tmp_mesh, quadrature_order, discr_order, geom_bases, local_boundary, boundary_nodes, poly_edge_to_data_geom);

				n_bases = SplineBasis3d::build_bases(tmp_mesh, quadrature_order, bases, local_boundary, boundary_nodes, poly_edge_to_data);
			}
			else
			{
				if(mesh->is_simplicial())
				{
					n_bases = TetBasis3d::build_bases(tmp_mesh, quadrature_order, discr_order, bases, local_boundary, boundary_nodes, poly_edge_to_data);
				}
				else
				{
					if (!iso_parametric)
						HexBasis3d::build_bases(tmp_mesh, quadrature_order, 1, geom_bases, local_boundary, boundary_nodes, poly_edge_to_data_geom);

					n_bases = HexBasis3d::build_bases(tmp_mesh, quadrature_order, discr_order, bases, local_boundary, boundary_nodes, poly_edge_to_data);
				}
			}
		}
		else
		{
			const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
			if(use_splines)
			{
				if(!iso_parametric)
					QuadBasis2d::build_bases(tmp_mesh, quadrature_order, discr_order, geom_bases, local_boundary, boundary_nodes, poly_edge_to_data_geom);

				n_bases = SplineBasis2d::build_bases(tmp_mesh, quadrature_order, bases, local_boundary, boundary_nodes, poly_edge_to_data);
			}
			else
			{
				if(mesh->is_simplicial())
				{
					n_bases = TriBasis2d::build_bases(tmp_mesh, quadrature_order, discr_order, bases, local_boundary, boundary_nodes, poly_edge_to_data);
				}
				else
				{
					if(!iso_parametric)
						QuadBasis2d::build_bases(tmp_mesh, quadrature_order, 1, geom_bases, local_boundary, boundary_nodes, poly_edge_to_data_geom);

					n_bases = QuadBasis2d::build_bases(tmp_mesh, quadrature_order, discr_order, bases, local_boundary, boundary_nodes, poly_edge_to_data);
				}
			}
		}

		auto &bs = iso_parametric ? bases : geom_bases;

		// for(int k =0; k < 1; ++k)
		// {
		// 	Eigen::MatrixXd nodes(n_bases, 2);
		// 	nodes.setZero();
		// 	Eigen::Matrix<bool, Eigen::Dynamic, 1> nodes_setted(n_bases);
		// 	nodes_setted.setConstant(false);

		// 	for(size_t i = 0; i < bs.size(); ++i)
		// 	{
		// 		auto &lbs = bs[i].bases;


		// 		if(!mesh->is_spline_compatible(i))
		// 		{
		// 			for(size_t b = 0; b < lbs.size(); ++b)
		// 			{
		// 				if(lbs[b].global().size() > 1) continue;
		// 				if(nodes_setted(lbs[b].global().front().index)) continue;
		// 				auto &c_glob = lbs[b].global().front();


		// 				if(std::find(boundary_nodes.begin(), boundary_nodes.end(), c_glob.index) != boundary_nodes.end())
		// 					continue;

		// 				double count = 10;
		// 				auto node = c_glob.node;
		// 				node *= count;
		// 				// node.setZero();

		// 				for(size_t bi = 0; bi < lbs.size(); ++bi)
		// 				{
		// 					for(size_t ii = 0; ii < lbs[bi].global().size(); ++ii)
		// 					{
		// 						auto &glob = lbs[bi].global()[ii];
		// 						double w = glob.val;
		// 						node += glob.node*w;

		// 						count+=w;
		// 					}
		// 				}

		// 				node /= count;

		// 				nodes.row(c_glob.index) = node;
		// 				nodes_setted(c_glob.index) = true;
		// 			}
		// 		}
		// 		else
		// 		{
		// 			if(lbs.size() != 9)
		// 				continue;

		// 			double count = 1;
		// 			auto &c_glob = lbs[3*1+1].global().front();
		// 			auto node = c_glob.node;
		// 			// node.setZero();
		// 			for(size_t b = 0; b < lbs.size(); ++b)
		// 			{
		// 				if(b == 3*1+1) continue;

		// 				double w = 1;
		// 				if(std::find(boundary_nodes.begin(), boundary_nodes.end(), lbs[b].global().front().index) != boundary_nodes.end())
		// 					w = 2;
		// 				node += lbs[b].global().front().node*w;

		// 				count+=w;
		// 			}

		// 			node /= count;

		// 			nodes.row(c_glob.index) = node;
		// 			nodes_setted(c_glob.index) = true;
		// 		}
		// 	}

		// 	for(size_t i = 0; i < bs.size(); ++i)
		// 	{
		// 		auto &lbs = bs[i].bases;
		// 		for(size_t b = 0; b < lbs.size(); ++b)
		// 		{
		// 			for(size_t ii = 0; ii < lbs[b].global().size(); ++ii)
		// 			{
		// 				auto &glob = lbs[b].global()[ii];

		// 				if(!nodes_setted(glob.index)) continue;

		// 				glob.node = nodes.row(glob.index);
		// 			}
		// 		}
		// 	}
		// }



		n_flipped = 0;
		// flipped_elements.clear();
		for(size_t i = 0; i < bs.size(); ++i)
		{
			if(!mesh->is_simplicial() && mesh->is_polytope(i)) continue;

			ElementAssemblyValues vals;
			if(!vals.is_geom_mapping_positive(mesh->is_volume(), bs[i]))
			{
				// if(!parent_elements.empty())
				// 	flipped_elements.push_back(parent_elements[i]);
				// std::cout<<"Basis "<< i << ( parent_elements.size() > 0 ? (" -> " + std::to_string(parent_elements[i])) : "") << " has negative volume"<<std::endl;
				++n_flipped;
			}
		}

		// std::sort(flipped_elements.begin(), flipped_elements.end());
		// auto it = std::unique(flipped_elements.begin(), flipped_elements.end());
		// flipped_elements.resize(std::distance(flipped_elements.begin(), it));


		problem.remove_neumann_nodes(*mesh, bases, local_boundary, boundary_nodes);

		if(problem.problem_num() == 3)
		{
			const int dim = mesh->is_volume() ? 3:2;
			const std::size_t n_b_nodes = boundary_nodes.size();

			for(std::size_t i = 0; i < n_b_nodes; ++i)
			{
				boundary_nodes[i] *= dim;
				for(int d = 1; d < dim; ++d)
					boundary_nodes.push_back(boundary_nodes[i]+d);
			}
		}


		const auto &curret_bases =  iso_parametric ? bases : geom_bases;
		const int n_samples = 10;
		mesh_size = compute_mesh_size(*mesh, curret_bases, n_samples);

		timer.stop();
		building_basis_time = timer.getElapsedTime();
		std::cout<<" took "<<building_basis_time<<"s"<<std::endl;

		std::cout<<"flipped elements "<<n_flipped<<std::endl;
		std::cout<<"h: "<<mesh_size<<std::endl;
		std::cout<<"n bases: "<<n_bases<<std::endl;
	}


	void State::build_polygonal_basis()
	{
		// values.clear();
		// geom_values.clear();
		errors.clear();
		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		if(mesh->is_simplicial())
			return;

		igl::Timer timer; timer.start();
		std::cout<<"Computing polygonal basis..."<<std::flush;

		std::sort(boundary_nodes.begin(), boundary_nodes.end());

		if(iso_parametric) {
			// ElementAssemblyValues::compute_assembly_values(mesh->is_volume(), bases, values);

			if(mesh->is_volume()) {
				PolygonalBasis3d::build_bases(harmonic_samples_res, *dynamic_cast<Mesh3D *>(mesh.get()), n_bases, quadrature_order, integral_constraints, bases, bases, poly_edge_to_data, polys_3d);
				// Eigen::MatrixXd I;
				// compute_integral_constraints(*dynamic_cast<Mesh3D *>(mesh.get()), n_bases, bases, bases, I);
				// for (int r = 0; r < I.rows(); ++r) {
					// std::cout << r << ": " << I.row(r) << std::endl;
				// }
			} else {
				PolygonalBasis2d::build_bases(harmonic_samples_res, *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, quadrature_order, integral_constraints, bases, bases, poly_edge_to_data, polys);
			}

			// for(std::size_t e = 0; e < bases.size(); ++e) {
			// 	if(mesh->is_polytope(e)){
			// 		values[e].compute(e, mesh->is_volume(), bases[e]);
			// 	}
			// }
		}
		else
		{
			// ElementAssemblyValues::compute_assembly_values(mesh->is_volume(), geom_bases, geom_values);
			// ElementAssemblyValues::compute_assembly_values(mesh->is_volume(), bases, values);

			if(mesh->is_volume())
			{
				PolygonalBasis3d::build_bases(harmonic_samples_res, *dynamic_cast<Mesh3D *>(mesh.get()), n_bases, quadrature_order, integral_constraints, bases, geom_bases, poly_edge_to_data, polys_3d);
			}
			else
				PolygonalBasis2d::build_bases(harmonic_samples_res, *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, quadrature_order, integral_constraints, bases, geom_bases, poly_edge_to_data, polys);

			// for(std::size_t e = 0; e < bases.size(); ++e)
			// {
			// 	if(mesh->is_polytope(e)){
			// 		geom_values[e].compute(e, mesh->is_volume(), geom_bases[e]);
			// 		values[e].compute(e, mesh->is_volume(), bases[e]);
			// 	}
			// }
		}


		// Eigen::MatrixXd c;
		// compute_integral_constraints(*dynamic_cast<Mesh2D *>(mesh), n_bases, bases, bases, c);
		// for (int r = 0; r < c.rows(); ++r) {
		// 	std::cout << r << ": " << c.row(r) << std::endl;
		// }

		timer.stop();
		computing_assembly_values_time = timer.getElapsedTime();
		std::cout<<" took "<<computing_assembly_values_time<<"s"<<std::endl;
	}

	void State::assemble_stiffness_mat()
	{
		errors.clear();
		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		igl::Timer timer; timer.start();
		std::cout<<"Assembling stiffness mat..."<<std::flush;

		if(problem.problem_num() == 3)
		{
			Assembler<LinearElasticity> assembler;
			LinearElasticity &le = assembler.local_assembler();
			le.mu() = mu;
			le.lambda() = lambda;
			le.size() = mesh->is_volume()? 3:2;

			if(iso_parametric)
				assembler.assemble(mesh->is_volume(), n_bases, bases, bases, stiffness);
			else
				assembler.assemble(mesh->is_volume(), n_bases, bases, geom_bases, stiffness);

			// std::cout<<MatrixXd(stiffness)<<std::endl;
			// assembler.set_identity(boundary_nodes, stiffness);
		}
		else
		{
			Assembler<Laplacian> assembler;
			if(iso_parametric)
				assembler.assemble(mesh->is_volume(), n_bases, bases, bases, stiffness);
			else
				assembler.assemble(mesh->is_volume(), n_bases, bases, geom_bases, stiffness);

			// std::cout<<MatrixXd(stiffness)-MatrixXd(stiffness.transpose())<<"\n\n"<<std::endl;
			// std::cout<<MatrixXd(stiffness).rowwise().sum()<<"\n\n"<<std::endl;
			// assembler.set_identity(boundary_nodes, stiffness);
		}

		timer.stop();
		assembling_stiffness_mat_time = timer.getElapsedTime();
		std::cout<<" took "<<assembling_stiffness_mat_time<<"s"<<std::endl;

		nn_zero = stiffness.nonZeros();
		num_dofs = stiffness.rows();
		mat_size = (long long) stiffness.rows() * (long long) stiffness.cols();
		std::cout<<"sparsity: "<<nn_zero<<"/"<<mat_size<<std::endl;


		// {
		// 	std::ofstream of;
		// 	of.open("stiffness.txt");
		// 	of.precision(100);
		// 	for(long i = 0; i < stiffness.rows(); ++i)
		// 	{
		// 		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(stiffness, i); it; ++it)
		// 		{
		// 			of << it.row() << " " << it.col() << " "<<it.valueRef() <<"\n";
		// 		}
		// 	}
		// 	of.close();
		// }
	}

	void State::assemble_rhs()
	{
		errors.clear();
		rhs.resize(0, 0);
		sol.resize(0, 0);

		igl::Timer timer; timer.start();
		std::cout<<"Assigning rhs..."<<std::flush;

		const int size = problem.problem_num() == 3 ? (mesh->is_volume() ? 3:2) : 1;

		if(iso_parametric)
		{
			RhsAssembler rhs_assembler(*mesh, n_bases, size, bases, bases, problem);
			rhs_assembler.assemble(rhs);
			rhs *= -1;
			rhs_assembler.set_bc(local_boundary, boundary_nodes, n_boundary_samples, rhs);
		}
		else
		{
			RhsAssembler rhs_assembler(*mesh, n_bases, size, bases, geom_bases, problem);
			rhs_assembler.assemble(rhs);
			rhs *= -1;
			rhs_assembler.set_bc(local_boundary, boundary_nodes, n_boundary_samples, rhs);
		}

		timer.stop();
		assigning_rhs_time = timer.getElapsedTime();
		std::cout<<" took "<<assigning_rhs_time<<"s"<<std::endl;

		// {
		// 	std::ofstream of;
		// 	of.open("rhs.txt");
		// 	of.precision(100);
		// 	of<<rhs;
		// 	of.close();
		// }
	}

	void State::solve_problem()
	{
		errors.clear();
		sol.resize(0, 0);

		igl::Timer timer; timer.start();
		std::cout<<"Solving... "<<std::flush;

		json params = {
			// {"mtype", 1}, // matrix type for Pardiso (2 = SPD)
			// {"max_iter", 0}, // for iterative solvers
			// {"tolerance", 1e-9}, // for iterative solvers
		};
		auto solver = LinearSolver::create(solver_type, precond_type);
		solver->setParameters(params);

		Eigen::SparseMatrix<double> A = stiffness;
		Eigen::VectorXd x, b = rhs;
		dirichlet_solve(*solver, A, b, boundary_nodes, x);
		sol = x;
		solver->getInfo(solver_info);

		timer.stop();
		solving_time = timer.getElapsedTime();
		std::cout<<" took "<<solving_time<<"s"<<std::endl;
		std::cout<<"Solver error: "<<(A*sol-b).norm()<<std::endl;
	}

	void State::compute_errors()
	{
		errors.clear();

		if(!problem.has_exact_sol()) return;

		igl::Timer timer; timer.start();
		std::cout<<"Computing errors..."<<std::flush;
		using std::max;

		const int n_el=int(bases.size());

		MatrixXd v_exact, v_approx;

		errors.clear();

		l2_err = 0;
		linf_err = 0;
		lp_err = 0;

		for(int e = 0; e < n_el; ++e)
		{
			// const auto &vals    = values[e];
			// const auto &gvalues = iso_parametric ? values[e] : geom_values[e];

			ElementAssemblyValues vals;

			if(iso_parametric)
				vals.compute(e, mesh->is_volume(), bases[e], bases[e]);
			else
				vals.compute(e, mesh->is_volume(), bases[e], geom_bases[e]);

			problem.exact(vals.val, v_exact);

			v_approx = MatrixXd::Zero(v_exact.rows(), v_exact.cols());

			const int n_loc_bases=int(vals.basis_values.size());

			for(int i = 0; i < n_loc_bases; ++i)
			{
				auto val=vals.basis_values[i];

				for(std::size_t ii = 0; ii < val.global.size(); ++ii)
					v_approx += val.global[ii].val * sol(val.global[ii].index) * val.val;
			}

			const auto err = (v_exact-v_approx).cwiseAbs();

			for(long i = 0; i < err.size(); ++i)
				errors.push_back(err(i));

			linf_err = max(linf_err, err.maxCoeff());
			l2_err += (err.array() * err.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			lp_err += (err.array().pow(8.) * vals.det.array() * vals.quadrature.weights.array()).sum();
		}

		l2_err = sqrt(fabs(l2_err));
		lp_err = pow(fabs(lp_err), 1./8.);

		timer.stop();
		computing_errors_time = timer.getElapsedTime();
		std::cout<<" took "<<computing_errors_time<<"s"<<std::endl;

		std::cout << "-- L2 error: " << l2_err << std::endl;
		std::cout << "-- Lp error: " << lp_err << std::endl;
		// std::cout<<l2_err<<" "<<linf_err<<" "<<lp_err<<std::endl;
	}

	State &State::state(){
		static State instance;

		return instance;
	}

	void State::init(const std::string &mesh_path_, const int n_refs_, const int problem_num)
	{
		n_refs = n_refs_;
		mesh_path = mesh_path_;

		problem.set_problem_num(problem_num);
		auto solvers = LinearSolver::availableSolvers();
		if (std::find(solvers.begin(), solvers.end(), solver_type) == solvers.end()) {
			solver_type = LinearSolver::defaultSolver();
		}
		auto precond = LinearSolver::availablePrecond();
		if (std::find(precond.begin(), precond.end(), precond_type) == precond.end()) {
			precond_type = LinearSolver::defaultPrecond();
		}
	}

	void State::sertialize(const std::string &file_name)
	{
		igl::serialize(quadrature_order, "quadrature_order", file_name, true);
		igl::serialize(n_boundary_samples, "n_boundary_samples", file_name);

		igl::serialize(mesh_path, "mesh_path", file_name);
		igl::serialize(n_refs, "n_refs", file_name);

		igl::serialize(use_splines, "use_splines", file_name);

		igl::serialize(problem, "problem", file_name);

		igl::serialize(n_bases, "n_bases", file_name);

		igl::serialize(bases, "bases", file_name);
		igl::serialize(boundary_nodes, "boundary_nodes", file_name);
		igl::serialize(local_boundary, "local_boundary", file_name);


		igl::serialize(*mesh, "mesh", file_name);

		igl::serialize(polys, "polys", file_name);


		igl::serialize(stiffness, "stiffness", file_name);
		igl::serialize(rhs, "rhs", file_name);
		igl::serialize(sol, "sol", file_name);

		igl::serialize(mesh_size, "mesh_size", file_name);
		igl::serialize(l2_err, "l2_err", file_name);
		igl::serialize(linf_err, "linf_err", file_name);
		igl::serialize(nn_zero, "nn_zero", file_name);
		igl::serialize(mat_size, "mat_size", file_name);

		igl::serialize(building_basis_time, "building_basis_time", file_name);
		igl::serialize(loading_mesh_time, "loading_mesh_time", file_name);
		igl::serialize(computing_assembly_values_time, "computing_assembly_values_time", file_name);
		igl::serialize(assembling_stiffness_mat_time, "assembling_stiffness_mat_time", file_name);
		igl::serialize(assigning_rhs_time, "assigning_rhs_time", file_name);
		igl::serialize(solving_time, "solving_time", file_name);
		igl::serialize(computing_errors_time, "computing_errors_time", file_name);
	}


	void State::save_vtu(const std::string &path)
	{
		if(!mesh->is_volume()){
			std::cerr<<"Saving vtu supported only for volume"<<std::endl;
			return;
		}
		if(mesh->is_simplicial()){
			std::cerr<<"Saving vtu supported only for pure hex meshes"<<std::endl;
			return;
		}

		const double area_param = 0.00001*mesh->n_elements();
		// const double area_param = 1;

		std::stringstream buf;
		buf.precision(100);
		buf.setf(std::ios::fixed, std::ios::floatfield);

		Eigen::MatrixXd hex_pts;
		Eigen::MatrixXi hex_tets;
		Eigen::MatrixXi dummy;

		buf<<"Qpq1.414a"<<area_param;
		{
			MatrixXd pts(8,3); pts <<
			0, 0, 0,
			0, 1, 0,
			1, 1, 0,
			1, 0, 0,

			0, 0, 1, //4
			0, 1, 1,
			1, 1, 1,
			1, 0, 1;

			Eigen::MatrixXi faces(12,3); faces <<
			1, 2, 0,
			0, 2, 3,

			5, 4, 6,
			4, 7, 6,

			1, 0, 4,
			1, 4, 5,

			2, 1, 5,
			2, 5, 6,

			3, 2, 6,
			3, 6, 7,

			0, 3, 7,
			0, 7, 4;
			igl::copyleft::tetgen::tetrahedralize(pts, faces, buf.str(), hex_pts, hex_tets, dummy);
		}

		const auto &current_bases = iso_parametric ? bases : geom_bases;
		int tet_total_size = 0;
		int pts_total_size = 0;

		for(size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if(mesh->is_cube(i)){
				pts_total_size += hex_pts.rows();
				tet_total_size += hex_tets.rows();
			}
		}

		Eigen::MatrixXd points(pts_total_size, 3);
		Eigen::MatrixXi tets(tet_total_size, 4);

		MatrixXd mapped, tmp;
		int tet_index = 0, pts_index = 0;
		for(size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];
			if(mesh->is_cube(i))
			{
				bs.eval_geom_mapping(hex_pts, mapped);
				tets.block(tet_index, 0, hex_tets.rows(), 4) = hex_tets.array() + pts_index;
				tet_index += hex_tets.rows();

				points.block(pts_index, 0, mapped.rows(), mapped.cols()) = mapped;
				pts_index += mapped.rows();
			}
		}

		assert(pts_index == points.rows());
		assert(tet_index == tets.rows());

		Eigen::MatrixXd fun;
		interpolate_function(sol, hex_pts, fun);

		VTUWriter writer;
		writer.add_filed("sol", fun);
		writer.write_tet_mesh(path, points, tets);
	}


	void State::compute_poly_basis_error(const std::string &path)
	{
		MatrixXd fun = MatrixXd::Zero(n_bases, 1);
		MatrixXd tmp, mapped;
		MatrixXd v_approx, v_exact;

		int poly_index = -1;

		for(std::size_t i = 0; i < bases.size(); ++i)
		{
			const ElementBases &basis = bases[i];
			if(!basis.has_parameterization){
				poly_index = i;
				continue;
			}

			for(std::size_t j = 0; j < basis.bases.size(); ++j)
			{
				for(std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
				{
					const Local2Global &l2g = basis.bases[j].global()[kk];
					const int g_index = l2g.index;

					const auto &node = l2g.node;
					problem.exact(node, tmp);

					fun(g_index) = tmp(0);
				}
			}
		}

		if(poly_index == -1)
			poly_index = 0;

		auto &poly_basis = bases[poly_index];
		ElementAssemblyValues vals;
		vals.compute(poly_index, true, poly_basis, poly_basis);

		problem.exact(vals.val, v_exact);
		v_approx = MatrixXd::Zero(v_exact.rows(), v_exact.cols());

		const int n_loc_bases=int(vals.basis_values.size());

		for(int i = 0; i < n_loc_bases; ++i)
		{
			auto &val=vals.basis_values[i];

			for(std::size_t ii = 0; ii < val.global.size(); ++ii)
			{
				v_approx += val.global[ii].val * fun(val.global[ii].index) * val.val;
			}
		}

		const auto err = (v_exact-v_approx).cwiseAbs();

		double l2_err_interp = 0;
		double lp_err_interp = 0;
		l2_err_interp += (err.array() * err.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
		lp_err_interp += (err.array().pow(8.) * vals.det.array() * vals.quadrature.weights.array()).sum();

		l2_err_interp = sqrt(fabs(l2_err_interp));
		lp_err_interp = pow(fabs(lp_err_interp), 1./8.);


		using json = nlohmann::json;
		json j;

		j["mesh_path"] = mesh_path;
		j["err_l2"] = l2_err_interp;
		j["err_lp"] = lp_err_interp;

		std::ofstream out(path);
		out << j.dump(4) << std::endl;
		out.close();
	}

}
