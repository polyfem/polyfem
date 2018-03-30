#include "State.hpp"

#include "Mesh2D.hpp"
#include "Mesh3D.hpp"

#include "FEBasis2d.hpp"
#include "FEBasis3d.hpp"

#include "SpectralBasis2d.hpp"

#include "SplineBasis2d.hpp"
#include "SplineBasis3d.hpp"

#include "EdgeSampler.hpp"

#include "PolygonalBasis2d.hpp"
#include "PolygonalBasis3d.hpp"

#include "AssemblerUtils.hpp"
#include "RhsAssembler.hpp"

#include "LinearSolver.hpp"
#include "FEMSolver.hpp"

#include "RefElementSampler.hpp"

#include "Common.hpp"

#include "CustomSerialization.hpp"
#include "VTUWriter.hpp"

#include "NLProblem.hpp"
#include "SparseNewtonDescentSolver.hpp"

#include <igl/Timer.h>
#include <igl/serialize.h>


#include <unsupported/Eigen/SparseExtra>

#include <iostream>
#include <algorithm>
#include <memory>

#include "autodiff.h"
DECLARE_DIFFSCALAR_BASE();


using namespace Eigen;


namespace poly_fem
{
	State::State()
	{
		problem = ProblemFactory::factory().get_problem("Linear");
	}

	void State::compute_mesh_size(const Mesh &mesh_in, const std::vector< ElementBases > &bases_in, const int n_samples)
	{
		Eigen::MatrixXd samples, mapped, p0, p1, p;

		mesh_size = 0;
		average_edge_length = 0;
		min_edge_length = std::numeric_limits<double>::max();

		if(true)
		{
			mesh_in.get_edges(p0, p1);
			p = p0-p1;
			min_edge_length = p.rowwise().norm().minCoeff();
			average_edge_length = p.rowwise().norm().mean();
			mesh_size = p.rowwise().norm().maxCoeff();

			std::cout << std::endl;
			std::cout << "hmin: " << min_edge_length << std::endl;
			std::cout << "hmax: " << mesh_size << std::endl;
			std::cout << "havg: " << average_edge_length << std::endl;

			return;
		}

		const int n_edges = mesh_in.is_volume()?12:4;
		if(mesh_in.is_volume())
			EdgeSampler::sample_3d(n_samples, samples);
		else
			EdgeSampler::sample_2d(n_samples, samples);


		int n = 0;
		for(std::size_t i = 0; i < bases_in.size(); ++i){
			if(mesh_in.is_polytope(i)) continue;

			bases_in[i].eval_geom_mapping(samples, mapped);

			for(int j = 0; j < n_edges; ++j)
			{
				double current_edge = 0;
				for(int k = 0; k < n_samples-1; ++k){
					p0 = mapped.row(j*n_samples + k);
					p1 = mapped.row(j*n_samples + k+1);
					p = p0-p1;

					current_edge += p.norm();
				}

				mesh_size = std::max(current_edge, mesh_size);
				min_edge_length = std::min(current_edge, min_edge_length);
				average_edge_length += current_edge;
				++n;
			}
		}

		average_edge_length /= n;
	}

	void State::save_json(std::ostream &out)
	{
		std::cout<<"Saving json..."<<std::flush;
		using json = nlohmann::json;
		json j;

		j["args"] = args;
		j["quadrature_order"] = args["quadrature_order"];
		j["mesh_path"] = mesh_path();
		j["discr_order"] = args["discr_order"];
		j["harmonic_samples_res"] = args["n_harmonic_samples"];
		j["use_splines"] = args["use_spline"];
		j["iso_parametric"] = iso_parametric();
		j["problem"] = problem->name();
		j["mat_size"] = mat_size;
		j["solver_type"] = args["solver_type"];
		j["precond_type"] = args["precond_type"];
		j["params"] = args["params"];

		j["refinenemt_location"] = args["refinenemt_location"];

		j["num_boundary_samples"] = args["n_boundary_samples"];
		j["num_refs"] = args["n_refs"];
		j["num_bases"] = n_bases;
		j["num_non_zero"] = nn_zero;
		j["num_flipped"] = n_flipped;
		j["num_dofs"] = num_dofs;
		j["num_vertices"] = mesh->n_vertices();
		j["num_elements"] = mesh->n_elements();

		j["mesh_size"] = mesh_size;

		j["min_edge_length"] = min_edge_length;
		j["average_edge_length"] = average_edge_length;

		j["err_l2"] = l2_err;
		j["err_h1"] = h1_err;
		j["err_h1_semi"] = h1_semi_err;
		j["err_linf"] = linf_err;
		j["err_linf_grad"] = grad_max_err;
		j["err_lp"] = lp_err;

		j["spectrum"] = {spectrum(0), spectrum(1), spectrum(2), spectrum(3)};

		// j["errors"] = errors;

		j["time_building_basis"] = building_basis_time;
		j["time_loading_mesh"] = loading_mesh_time;
		j["time_computing_assembly_values"] = computing_assembly_values_time;
		j["time_assembling_stiffness_mat"] = assembling_stiffness_mat_time;
		j["time_assigning_rhs"] = assigning_rhs_time;
		j["time_solving"] = solving_time;
		j["time_computing_errors"] = computing_errors_time;

		j["solver_info"] = solver_info;

		j["count_simplex"] = simplex_count;
		j["count_regular"] = regular_count;
		j["count_regular_boundary"] = regular_boundary_count;
		j["count_simple_singular"] = simple_singular_count;
		j["count_multi_singular"] = multi_singular_count;
		j["count_boundary"] = boundary_count;
		j["count_non_regular_boundary"] = non_regular_boundary_count;
		j["count_non_regular"] = non_regular_count;
		j["count_undefined"] = undefined_count;
		j["count_multi_singular_boundary"] = multi_singular_boundary_count;

		j["is_simplicial"] = mesh->n_elements() == simplex_count;

		j["formulation"] = formulation();


		out << j.dump(4) << std::endl;

		std::cout<<"done"<<std::endl;
	}


	void State::interpolate_function(const int n_points, const MatrixXd &fun, MatrixXd &result)
	{
		MatrixXd tmp;

		int actual_dim = 1;
		if(!problem->is_scalar())
			actual_dim = mesh->dimension();

		result.resize(n_points, actual_dim);

		int index = 0;
		const auto &sampler = RefElementSampler::sampler();


		for(int i = 0; i < int(bases.size()); ++i)
		{
			const ElementBases &bs = bases[i];
			MatrixXd local_pts;

			if(mesh->is_simplex(i))
				local_pts = sampler.simplex_points();
			else if(mesh->is_cube(i))
				local_pts = sampler.cube_points();
			// else
				// local_pts = vis_pts_poly[i];

			MatrixXd local_res = MatrixXd::Zero(local_pts.rows(), actual_dim);
			bs.evaluate_bases(local_pts, tmp);
			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];

				for(int d = 0; d < actual_dim; ++d)
				{
					for(std::size_t ii = 0; ii < b.global().size(); ++ii)
						local_res.col(d) += b.global()[ii].val * tmp.col(j) * fun(b.global()[ii].index*actual_dim + d);
				}
			}

			result.block(index, 0, local_res.rows(), actual_dim) = local_res;
			index += local_res.rows();
		}
	}

	void State::compute_scalar_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result)
	{
		result.resize(n_points, 1);

		int index = 0;
		const auto &sampler = RefElementSampler::sampler();
		const auto &assembler = AssemblerUtils::instance();

		Eigen::MatrixXd local_val;

		for(int i = 0; i < int(bases.size()); ++i)
		{
			const ElementBases &bs = bases[i];
			Eigen::MatrixXd local_pts;

			if(mesh->is_simplex(i))
				local_pts = sampler.simplex_points();
			else if(mesh->is_cube(i))
				local_pts = sampler.cube_points();
			// else
				// local_pts = vis_pts_poly[i];

			assembler.compute_scalar_value(tensor_formulation(), bs, local_pts, sol, local_val);

			result.block(index, 0, local_val.rows(), 1) = local_val;
			index += local_val.rows();
		}
	}

	void State::load_mesh()
	{
		bases.clear();
		geom_bases.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();
		parent_elements.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		n_bases = 0;



		igl::Timer timer; timer.start();
		std::cout<<"Loading mesh..."<<std::flush;
		mesh = Mesh::create(mesh_path());
		if (!mesh) {
			return;
		}

		// if(!flipped_elements.empty())
		// {
		// 	mesh->compute_elements_tag();
		// 	for(auto el_id : flipped_elements)
		// 		mesh->set_tag(el_id, ElementType::InteriorPolytope);
		// }

		if(args["normalize_mesh"])
			mesh->normalize();

		mesh->refine(args["n_refs"], args["refinenemt_location"], parent_elements);

		mesh->compute_boundary_ids();


		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

		RefElementSampler::sampler().init(mesh->is_volume(), mesh->n_elements());
	}

	void State::compute_mesh_stats()
	{
		bases.clear();
		geom_bases.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		n_bases = 0;

		simplex_count = 0;
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
				case ElementType::Simplex: simplex_count++; break;
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
		"simplex_count:\t " << simplex_count <<"\n"<<
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
		boundary_nodes.clear();
		local_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();
		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		n_bases = 0;


		igl::Timer timer; timer.start();
		std::cout<<"Building "<< (iso_parametric()? "isoparametric":"not isoparametric") <<" basis..."<<std::flush;

		local_boundary.clear();
		std::map<int, InterfaceData> poly_edge_to_data_geom; //temp dummy variable

		if(mesh->is_volume())
		{
			const Mesh3D &tmp_mesh = *dynamic_cast<Mesh3D *>(mesh.get());
			if(args["use_spline"])
			{
				if(!iso_parametric())
					FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], args["discr_order"], geom_bases, local_boundary, poly_edge_to_data_geom);

				n_bases = SplineBasis3d::build_bases(tmp_mesh, args["quadrature_order"], bases, local_boundary, poly_edge_to_data);

				if(iso_parametric() &&  args["fit_nodes"])
					SplineBasis3d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if (!iso_parametric())
					FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], 1, geom_bases, local_boundary, poly_edge_to_data_geom);

				n_bases = FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], args["discr_order"], bases, local_boundary, poly_edge_to_data);
			}
		}
		else
		{
			const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
			if(args["use_spline"])
			{

				if(!iso_parametric())
					FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], args["discr_order"], geom_bases, local_boundary, poly_edge_to_data_geom);

				n_bases = SplineBasis2d::build_bases(tmp_mesh, args["quadrature_order"], bases, local_boundary, poly_edge_to_data);

				if(iso_parametric() && args["fit_nodes"])
					SplineBasis2d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if(!iso_parametric())
					FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], 1, geom_bases, local_boundary, poly_edge_to_data_geom);

				n_bases = FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], args["discr_order"], bases, local_boundary, poly_edge_to_data);
				// n_bases = SpectralBasis2d::build_bases(tmp_mesh, args["quadrature_order"], args["discr_order"], bases, geom_bases, local_boundary);
			}
		}

		auto &gbases = iso_parametric() ? bases : geom_bases;


		n_flipped = 0;
		// flipped_elements.clear();
		for(size_t i = 0; i < gbases.size(); ++i)
		{
			if(mesh->is_polytope(i)) continue;

			ElementAssemblyValues vals;
			if(!vals.is_geom_mapping_positive(mesh->is_volume(), gbases[i]))
			{
				// if(!parent_elements.empty())
				// 	flipped_elements.push_back(parent_elements[i]);
				std::cout<<"Basis "<< i << ( parent_elements.size() > 0 ? (" -> " + std::to_string(parent_elements[i])) : "") << " has negative volume"<<std::endl;
				++n_flipped;
			}
		}

		// std::sort(flipped_elements.begin(), flipped_elements.end());
		// auto it = std::unique(flipped_elements.begin(), flipped_elements.end());
		// flipped_elements.resize(std::distance(flipped_elements.begin(), it));


		problem->remove_neumann_nodes(*mesh, bases, local_boundary, boundary_nodes);

		if(!problem->is_scalar())
		{
			const int dim = mesh->dimension();
			const std::size_t n_b_nodes = boundary_nodes.size();

			for(std::size_t i = 0; i < n_b_nodes; ++i)
			{
				boundary_nodes[i] *= dim;
				for(int d = 1; d < dim; ++d)
					boundary_nodes.push_back(boundary_nodes[i]+d);
			}
		}

		std::sort(boundary_nodes.begin(), boundary_nodes.end());


		const auto &curret_bases =  iso_parametric() ? bases : geom_bases;
		const int n_samples = 10;
		compute_mesh_size(*mesh, curret_bases, n_samples);

		timer.stop();
		building_basis_time = timer.getElapsedTime();
		std::cout<<" took "<<building_basis_time<<"s"<<std::endl;

		std::cout<<"flipped elements "<<n_flipped<<std::endl;
		std::cout<<"h: "<<mesh_size<<std::endl;
		std::cout<<"n bases: "<<n_bases<<std::endl;
	}


	void State::build_polygonal_basis()
	{
		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);


		// TODO, implement the poly bases for simplices
		// if(mesh->is_simplicial())
		// 	return;

		igl::Timer timer; timer.start();
		std::cout<<"Computing polygonal basis..."<<std::flush;

		// std::sort(boundary_nodes.begin(), boundary_nodes.end());

		if(iso_parametric())
		{
			if(mesh->is_volume())
				PolygonalBasis3d::build_bases(args["n_harmonic_samples"], *dynamic_cast<Mesh3D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, bases, poly_edge_to_data, polys_3d);
			else
				PolygonalBasis2d::build_bases(args["n_harmonic_samples"], *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, bases, poly_edge_to_data, polys);
		}
		else
		{
			if(mesh->is_volume())
				PolygonalBasis3d::build_bases(args["n_harmonic_samples"], *dynamic_cast<Mesh3D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, geom_bases, poly_edge_to_data, polys_3d);
			else
				PolygonalBasis2d::build_bases(args["n_harmonic_samples"], *dynamic_cast<Mesh2D *>(mesh.get()), n_bases, args["quadrature_order"], args["integral_constraints"], bases, geom_bases, poly_edge_to_data, polys);
		}

		timer.stop();
		computing_assembly_values_time = timer.getElapsedTime();
		std::cout<<" took "<<computing_assembly_values_time<<"s"<<std::endl;
	}

	json State::build_json_params()
	{
		json params = args["params"];
		params["size"] = mesh->dimension();

		return params;
	}

	void State::assemble_stiffness_mat()
	{
		stiffness.resize(0, 0);
		sol.resize(0, 0);

		igl::Timer timer; timer.start();
		std::cout<<"Assembling stiffness mat..."<<std::flush;

		auto &assembler = AssemblerUtils::instance();

		if(problem->is_scalar())
		{
			assembler.assemble_scalar_problem(scalar_formulation(), mesh->is_volume(), n_bases, bases, iso_parametric() ? bases : geom_bases, stiffness);
		}
		else
		{
			assembler.assemble_tensor_problem(tensor_formulation(), mesh->is_volume(), n_bases, bases, iso_parametric() ? bases : geom_bases, stiffness);
		}

		timer.stop();
		assembling_stiffness_mat_time = timer.getElapsedTime();
		std::cout<<" took "<<assembling_stiffness_mat_time<<"s"<<std::endl;

		nn_zero = stiffness.nonZeros();
		num_dofs = stiffness.rows();
		mat_size = (long long) stiffness.rows() * (long long) stiffness.cols();
		std::cout<<"sparsity: "<<nn_zero<<"/"<<mat_size<<std::endl;
	}

	void State::assemble_rhs()
	{
		auto p_params = args["problem_params"];
		p_params["formulation"] = formulation();
		problem->set_parameters(p_params);

		const auto params = build_json_params();
		auto &assembler = AssemblerUtils::instance();
		assembler.set_parameters(params);

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		igl::Timer timer; timer.start();
		std::cout<<"Assigning rhs..."<<std::flush;

		const int size = problem->is_scalar() ? 1 : mesh->dimension();

		RhsAssembler rhs_assembler(*mesh, n_bases, size, bases, iso_parametric() ? bases : geom_bases, formulation(), *problem);
		rhs_assembler.assemble(rhs);
		rhs *= -1;
		rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], rhs);

		timer.stop();
		assigning_rhs_time = timer.getElapsedTime();
		std::cout<<" took "<<assigning_rhs_time<<"s"<<std::endl;
	}

	void State::solve_problem()
	{
		sol.resize(0, 0);

		igl::Timer timer; timer.start();
		std::cout<<"Solving... "<<std::flush;


		const json &params = solver_params();

		const auto &assembler = AssemblerUtils::instance();

		if(assembler.is_linear(formulation()))
		{
			auto solver = LinearSolver::create(args["solver_type"], args["precond_type"]);
			solver->setParameters(params);
			Eigen::SparseMatrix<double> A;
			Eigen::VectorXd b;

			// std::cout<<Eigen::MatrixXd(stiffness)<<std::endl;

			A = stiffness;
			Eigen::VectorXd x;
			b = rhs;
			spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, true, true);
			sol = x;
			solver->getInfo(solver_info);
		}
		else
		{
			RhsAssembler rhs_assembler(*mesh, n_bases, mesh->dimension(), bases, iso_parametric() ? bases : geom_bases, formulation(), *problem);

			NLProblem nl_problem(rhs_assembler);
			VectorXd tmp_sol = nl_problem.initial_guess();

			// {
			// 	tmp_sol.setRandom();
			// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad, expected_grad;
			// 	nl_problem.gradient(tmp_sol, actual_grad);

			// 	Eigen::SparseMatrix<double> hessian;
			// 	nl_problem.hessian(tmp_sol, hessian);
			// 	nl_problem.finiteGradient(tmp_sol, expected_grad, 0);
			// 	std::cout<<"difff\n"<<actual_grad - expected_grad<<std::endl;

			// 	tmp_sol.setRandom();
			// 	if(!nl_problem.checkGradient(tmp_sol, 0))
			// 		std::cerr<<"baaaaad grad"<<std::endl;

			// 	// if(!nl_problem.checkHessian(tmp_sol, 0))
			// 		// std::cerr<<"baaaaad hessian"<<std::endl;

			// 	assert(nl_problem.checkGradient(tmp_sol, 0));
			// 	// assert(nl_problem.checkHessian(tmp_sol, 0));
			// 	tmp_sol.setZero();

			// 	exit(0);
			// }

			cppoptlib::SparseNewtonDescentSolver<NLProblem> solver(true);
			solver.minimize(nl_problem, tmp_sol);

			const int full_size 	= n_bases*mesh->dimension();
			const int reduced_size 	= n_bases*mesh->dimension() - boundary_nodes.size();

			NLProblem::reduced_to_full_aux(full_size, reduced_size, tmp_sol, false, sol);
			solver.getInfo(solver_info);
		}

		timer.stop();
		solving_time = timer.getElapsedTime();
		std::cout<<" took "<<solving_time<<"s"<<std::endl;
		// std::cout<<"Solver error: "<<(A*sol-b).norm()<<std::endl;
	}

	void State::compute_errors()
	{

		if(!problem->has_exact_sol()) return;

		int actual_dim = 1;
		if(!problem->is_scalar())
			actual_dim = mesh->dimension();

		igl::Timer timer; timer.start();
		std::cout<<"Computing errors..."<<std::flush;
		using std::max;

		const int n_el=int(bases.size());

		MatrixXd v_exact, v_approx;
		MatrixXd v_exact_grad(0,0), v_approx_grad;


		l2_err = 0;
		h1_err = 0;
		grad_max_err = 0;
		h1_semi_err = 0;
		linf_err = 0;
		lp_err = 0;

		for(int e = 0; e < n_el; ++e)
		{
			// const auto &vals    = values[e];
			// const auto &gvalues = iso_parametric() ? values[e] : geom_values[e];

			ElementAssemblyValues vals;

			if(iso_parametric())
				vals.compute(e, mesh->is_volume(), bases[e], bases[e]);
			else
				vals.compute(e, mesh->is_volume(), bases[e], geom_bases[e]);

			problem->exact(vals.val, v_exact);
			problem->exact_grad(vals.val, v_exact_grad);

			v_approx 	  = MatrixXd::Zero(v_exact.rows(), v_exact.cols());
			v_approx_grad = MatrixXd::Zero(v_exact_grad.rows(), v_exact_grad.cols());

			const int n_loc_bases=int(vals.basis_values.size());

			for(int i = 0; i < n_loc_bases; ++i)
			{
				auto val=vals.basis_values[i];

				for(std::size_t ii = 0; ii < val.global.size(); ++ii){
					for(int d = 0; d < actual_dim; ++d)
					{
						v_approx.col(d) += val.global[ii].val * sol(val.global[ii].index*actual_dim + d) * val.val;
						v_approx_grad.block(0, d*val.grad_t_m.cols(), v_approx_grad.rows(), val.grad_t_m.cols()) += val.global[ii].val * sol(val.global[ii].index*actual_dim + d) * val.grad_t_m;
					}
				}
			}

			const auto err = (v_exact-v_approx).eval().rowwise().norm().eval();
			const auto err_grad = (v_exact_grad - v_approx_grad).eval().rowwise().norm().eval();

			// for(long i = 0; i < err.size(); ++i)
				// errors.push_back(err(i));

			linf_err = max(linf_err, err.maxCoeff());
			grad_max_err = max(linf_err, err_grad.maxCoeff());

			l2_err += (err.array() * err.array() 			* vals.det.array() * vals.quadrature.weights.array()).sum();
			h1_err += (err_grad.array() * err_grad.array() 	* vals.det.array() * vals.quadrature.weights.array()).sum();
			lp_err += (err.array().pow(8.) 					* vals.det.array() * vals.quadrature.weights.array()).sum();
		}

		h1_semi_err = sqrt(fabs(h1_err));
		h1_err = sqrt(fabs(l2_err) + fabs(h1_err));
		l2_err = sqrt(fabs(l2_err));

		lp_err = pow(fabs(lp_err), 1./8.);

		timer.stop();
		computing_errors_time = timer.getElapsedTime();
		std::cout<<" took "<<computing_errors_time<<"s"<<std::endl;

		std::cout << "-- L2 error: " << l2_err << std::endl;
		std::cout << "-- Lp error: " << lp_err << std::endl;
		std::cout << "-- H1 error: " << h1_err << std::endl;
		std::cout << "-- H1 semi error: " << h1_semi_err << std::endl;

		std::cout << "\n --Linf error: " << linf_err << std::endl;
		std::cout << "-- grad max error: " << grad_max_err << std::endl;
		// std::cout<<l2_err<<" "<<linf_err<<" "<<lp_err<<std::endl;
	}

	State &State::state(){
		static State instance;

		return instance;
	}

	void State::init(const json &args_in)
	{
		this->args = args_in;
		problem = ProblemFactory::factory().get_problem(args["problem"]);
		problem->set_parameters(args["problem_params"]);
	}


	void State::save_vtu(const std::string &path)
	{
		// if(!mesh->is_volume()){
		// 	std::cerr<<"Saving vtu supported only for volume"<<std::endl;
		// 	return;
		// }

		const auto &sampler = RefElementSampler::sampler();


		const auto &current_bases = iso_parametric() ? bases : geom_bases;
		int tet_total_size = 0;
		int pts_total_size = 0;

		for(size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if(mesh->is_simplex(i))
			{
				tet_total_size += sampler.simplex_volume().rows();
				pts_total_size += sampler.simplex_points().rows();
			}
			else if(mesh->is_cube(i)){
				tet_total_size += sampler.cube_volume().rows();
				pts_total_size += sampler.cube_points().rows();
			}
		}

		Eigen::MatrixXd points(pts_total_size, mesh->dimension());
		Eigen::MatrixXi tets(tet_total_size, mesh->is_volume()?4:3);

		MatrixXd mapped, tmp;
		int tet_index = 0, pts_index = 0;
		for(size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if(mesh->is_simplex(i))
			{
				bs.eval_geom_mapping(sampler.simplex_points(), mapped);
				tets.block(tet_index, 0, sampler.simplex_volume().rows(), tets.cols()) = sampler.simplex_volume().array() + pts_index;

				tet_index += sampler.simplex_volume().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				pts_index += mapped.rows();
			}
			else if(mesh->is_cube(i))
			{
				bs.eval_geom_mapping(sampler.cube_points(), mapped);
				tets.block(tet_index, 0, sampler.cube_volume().rows(), tets.cols()) = sampler.cube_volume().array() + pts_index;
				tet_index += sampler.cube_volume().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				pts_index += mapped.rows();
			}
		}

		assert(pts_index == points.rows());
		assert(tet_index == tets.rows());

		Eigen::MatrixXd fun, exact_fun;

		problem->exact(points, exact_fun);
		interpolate_function(pts_index, sol, fun);

		MatrixXd err = (fun - exact_fun).eval().rowwise().norm();

		VTUWriter writer;

		if(fun.cols() != 1 && !mesh->is_volume())
		{
			fun.conservativeResize(fun.rows(), 3);
			fun.col(2).setZero();

			exact_fun.conservativeResize(exact_fun.rows(), 3);
			exact_fun.col(2).setZero();
		}

		writer.add_filed("solution", fun);
		writer.add_filed("exact", exact_fun);
		writer.add_filed("error", err);




		if(fun.cols() != 1)
		{
			Eigen::MatrixXd scalar_val;
			compute_scalar_value(pts_index, sol, scalar_val);
			writer.add_filed("scalar_value", scalar_val);
		}

		writer.write_tet_mesh(path, points, tets);
	}


	// void State::compute_poly_basis_error(const std::string &path)
	// {
	// 	auto dx = [](const Eigen::MatrixXd &pts, Eigen::MatrixXd &val){ auto x = pts.col(0).array(); auto y = pts.col(1).array();  auto z = pts.col(2).array(); val =  (-59535 * x + 13230) * exp(-0.81e2 / 0.4e1 *  x *  x +  (9 * x) - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) / 0.1960e4 +  (-39690 * x + 30870) * exp(-0.81e2 / 0.4e1 *  x *  x + 0.63e2 / 0.2e1 *  x - 0.83e2 / 0.4e1 - 0.81e2 / 0.2e1 * y * y + 0.36e2 * y) / 0.1960e4 +  (-4860 * x - 540) * exp(-0.81e2 / 0.49e2 *  x *  x - 0.18e2 / 0.49e2 *  x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) / 0.1960e4 + 0.162e3 / 0.5e1 * ( x - 0.4e1 / 0.9e1) * exp(- (81 * x * x) - 0.162e3 * y * y +  (72 * x) + 0.216e3 * y - 0.90e2);};
	// 	auto dy = [](const Eigen::MatrixXd &pts, Eigen::MatrixXd &val){ auto x = pts.col(0).array(); auto y = pts.col(1).array();  auto z = pts.col(2).array(); val =  -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) * y + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) - 0.27e2 / 0.40e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) - 0.81e2 / 0.2e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.2e1 * y * y + 0.36e2 * y) * y + 0.18e2 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.2e1 * y * y + 0.36e2 * y) + 0.324e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.162e3 * y * y + 0.72e2 * x + 0.216e3 * y - 0.90e2) * y - 0.216e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.162e3 * y * y + 0.72e2 * x + 0.216e3 * y - 0.90e2);};
	// 	auto dz = [](const Eigen::MatrixXd &pts, Eigen::MatrixXd &val){ auto x = pts.col(0).array(); auto y = pts.col(1).array();  auto z = pts.col(2).array(); val = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) * z + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) - 0.27e2 / 0.40e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z); };



	// 	MatrixXd fun = MatrixXd::Zero(n_bases, 1);
	// 	MatrixXd tmp, mapped;
	// 	MatrixXd v_approx, v_exact;

	// 	int poly_index = -1;

	// 	for(std::size_t i = 0; i < bases.size(); ++i)
	// 	{
	// 		const ElementBases &basis = bases[i];
	// 		if(!basis.has_parameterization){
	// 			poly_index = i;
	// 			continue;
	// 		}

	// 		for(std::size_t j = 0; j < basis.bases.size(); ++j)
	// 		{
	// 			for(std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
	// 			{
	// 				const Local2Global &l2g = basis.bases[j].global()[kk];
	// 				const int g_index = l2g.index;

	// 				const auto &node = l2g.node;
	// 				problem->exact(node, tmp);

	// 				fun(g_index) = tmp(0);
	// 			}
	// 		}
	// 	}

	// 	if(poly_index == -1)
	// 		poly_index = 0;

	// 	auto &poly_basis = bases[poly_index];
	// 	ElementAssemblyValues vals;
	// 	vals.compute(poly_index, true, poly_basis, poly_basis);

	// 	// problem.exact(vals.val, v_exact);
	// 	v_exact.resize(vals.val.rows(), vals.val.cols());
	// 	dx(vals.val, tmp); v_exact.col(0) = tmp;
	// 	dy(vals.val, tmp); v_exact.col(1) = tmp;
	// 	dz(vals.val, tmp); v_exact.col(2) = tmp;

	// 	v_approx = MatrixXd::Zero(v_exact.rows(), v_exact.cols());

	// 	const int n_loc_bases=int(vals.basis_values.size());

	// 	for(int i = 0; i < n_loc_bases; ++i)
	// 	{
	// 		auto &val=vals.basis_values[i];

	// 		for(std::size_t ii = 0; ii < val.global.size(); ++ii)
	// 		{
	// 			// v_approx += val.global[ii].val * fun(val.global[ii].index) * val.val;
	// 			v_approx += val.global[ii].val * fun(val.global[ii].index) * val.grad;
	// 		}
	// 	}

	// 	const Eigen::MatrixXd err = (v_exact-v_approx).cwiseAbs();


	// 	using json = nlohmann::json;
	// 	json j;
	// 	j["mesh_path"] = mesh_path;

	// 	for(long c = 0; c < v_approx.cols();++c){
	// 		double l2_err_interp = 0;
	// 		double lp_err_interp = 0;

	// 		l2_err_interp += (err.col(c).array() * err.col(c).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
	// 		lp_err_interp += (err.col(c).array().pow(8.) * vals.det.array() * vals.quadrature.weights.array()).sum();

	// 		l2_err_interp = sqrt(fabs(l2_err_interp));
	// 		lp_err_interp = pow(fabs(lp_err_interp), 1./8.);


	// 		j["err_l2_"+std::to_string(c)] = l2_err_interp;
	// 		j["err_lp_"+std::to_string(c)] = lp_err_interp;
	// 	}

	// 	std::ofstream out(path);
	// 	out << j.dump(4) << std::endl;
	// 	out.close();
	// }

}
