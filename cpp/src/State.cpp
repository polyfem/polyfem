#include "State.hpp"

#include "Mesh2D.hpp"
#include "Mesh3D.hpp"

#include "FEBasis2d.hpp"
#include "FEBasis3d.hpp"

#include "SplineBasis2d.hpp"
#include "SplineBasis3d.hpp"

#include "QuadBoundarySampler.hpp"
#include "HexBoundarySampler.hpp"

#include "PolygonalBasis2d.hpp"

#include "Assembler.hpp"
#include "RhsAssembler.hpp"

#include "Laplacian.hpp"
#include "LinearElasticity.hpp"

#include "json.hpp"

#include "CustomSerialization.hpp"

#include <igl/Timer.h>
#include <igl/serialize.h>

#include <unsupported/Eigen/SparseExtra>

#include <iostream>
#include <algorithm>


// #ifdef POLY_FEM_WITH_SUPERLU
// #include <Eigen/SuperLUSupport>
// #endif

#ifdef POLY_FEM_WITH_UMFPACK
#include <Eigen/UmfPackSupport>
#endif
#include<Eigen/SparseLU>

using namespace Eigen;


namespace poly_fem
{

	void State::save_json(const std::string &name)
	{
		using json = nlohmann::json;
		json j;

		j["quadrature_order"] = quadrature_order;
		j["n_boundary_samples"] = n_boundary_samples;

		j["mesh_path"] = mesh_path;
		j["n_refs"] = n_refs;

		j["use_splines"] = use_splines;
		j["problem"] = problem.problem_num();

		j["n_bases"] = n_bases;

		j["mesh_size"] = mesh_size;

		j["l2_err"] = l2_err;
		j["linf_err"] = linf_err;
		j["lp_err"] = lp_err;

		// j["errors"] = errors;

		j["nn_zero"] = nn_zero;
		j["mat_size"] = mat_size;


		j["building_basis_time"] = building_basis_time;
		j["loading_mesh_time"] = loading_mesh_time;
		j["computing_assembly_values_time"] = computing_assembly_values_time;
		j["assembling_stiffness_mat_time"] = assembling_stiffness_mat_time;
		j["assigning_rhs_time"] = assigning_rhs_time;
		j["solving_time"] = solving_time;
		j["computing_errors_time"] = computing_errors_time;



		std::ofstream o(name);
		o << std::setw(4) << j << std::endl;
		o.close();

	}


	void State::compute_mesh_stats()
	{
		mesh->compute_element_tag(els_tag);

		int regular_count = 0;
		int regular_boundary_count = 0;
		int simple_singular_count = 0;
		int multi_singular_count = 0;
		int boundary_count = 0;
		int non_regular_boundary_count = 0;
		int non_regular_count = 0;
		int undefined_count = 0;
		int multi_singular_boundary_count = 0;

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

				case ElementType::MultiSingularBoundaryCube: multi_singular_boundary_count++; break;

				case ElementType::BoundaryPolytope: non_regular_boundary_count++; break;

				case ElementType::InteriorPolytope: non_regular_count++; break;

				case ElementType::Undefined: undefined_count++; break;
			}
		}

		std::cout <<"regular_count: " << regular_count <<" regular_boundary_count: " << regular_boundary_count << " simple_singular_count: " << simple_singular_count << " multi_singular_count: " << multi_singular_count << " singular_boundary_count: " << boundary_count <<" multi_singular_boundary_count: " << multi_singular_boundary_count << " non_regular_count: " <<  non_regular_count << " non_regular_boundary_count: " << non_regular_boundary_count << " undefined_count: " << undefined_count <<std::endl;
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

			MatrixXd local_res = MatrixXd::Zero(local_pts.rows(), actual_dim);

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];

				b.basis(local_pts, tmp);
				for(std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					for(int d = 0; d < actual_dim; ++d)
					{
						local_res.col(d) += b.global()[ii].val * tmp * fun(b.global()[ii].index*actual_dim + d);
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
		values.clear();
		geom_values.clear();
		bounday_nodes.clear();
		local_boundary.clear();
		boundary_tag.clear();
		errors.clear();
		polys.clear();
		els_tag.clear();
		poly_edge_to_data.clear();
		delete mesh;



		igl::Timer timer; timer.start();
		std::cout<<"Loading mesh..."<<std::flush;
		std::string extension = mesh_path.substr(mesh_path.find_last_of(".") + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
		const bool is_volume = extension == "hybrid";

		if(is_volume)
			mesh = new Mesh3D();
		else
			mesh = new Mesh2D();

		mesh->load(mesh_path);
		mesh->refine(n_refs, refinenemt_location);

		mesh->set_boundary_tags(boundary_tag);

		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;
	}

	void State::build_basis()
	{
		igl::Timer timer; timer.start();
		std::cout<<"Building basis..."<<std::flush;

		local_boundary.clear();
		bounday_nodes.clear();
		std::map<int, InterfaceData> poly_edge_to_data_geom; //temp dummy variable

		if(mesh->is_volume())
		{
			const Mesh3D &tmp_mesh = *static_cast<Mesh3D *>(mesh);
			if(use_splines)
				n_bases = SplineBasis3d::build_bases(tmp_mesh, els_tag, quadrature_order, bases, local_boundary, bounday_nodes, polys);
			else
				n_bases = FEBasis3d::build_bases(tmp_mesh, quadrature_order, discr_order, bases, local_boundary, bounday_nodes);
		}
		else
		{
			const Mesh2D &tmp_mesh = *static_cast<Mesh2D *>(mesh);
			if(use_splines){
				if(iso_parametric)
					n_bases = SplineBasis2d::build_bases(tmp_mesh, els_tag, quadrature_order, bases, local_boundary, bounday_nodes, poly_edge_to_data);
				else
				{
					n_geom_bases = FEBasis2d::build_bases(tmp_mesh, quadrature_order, discr_order, geom_bases, local_boundary, bounday_nodes, poly_edge_to_data_geom);
					n_bases = SplineBasis2d::build_bases(tmp_mesh, els_tag, quadrature_order, bases, local_boundary, bounday_nodes, poly_edge_to_data);
				}
			}
			else
			{
				if(iso_parametric)
					n_bases = FEBasis2d::build_bases(tmp_mesh, quadrature_order, discr_order, bases, local_boundary, bounday_nodes, poly_edge_to_data);
				else
				{
					n_geom_bases = FEBasis2d::build_bases(tmp_mesh, quadrature_order, discr_order, geom_bases, local_boundary, bounday_nodes, poly_edge_to_data_geom);
					n_bases = FEBasis2d::build_bases(tmp_mesh, quadrature_order, discr_order, bases, local_boundary, bounday_nodes, poly_edge_to_data_geom);
				}
			}
		}

		problem.remove_neumann_nodes(bases, boundary_tag, local_boundary, bounday_nodes);

		if(problem.problem_num() == 3)
		{
			const int dim = mesh->is_volume() ? 3:2;
			const std::size_t n_b_nodes = bounday_nodes.size();

			for(std::size_t i = 0; i < n_b_nodes; ++i)
			{
				bounday_nodes[i] *= dim;
				for(int d = 1; d < dim; ++d)
					bounday_nodes.push_back(bounday_nodes[i]+d);
			}
		}


		const int n_samples = 10;
		mesh_size = 0;
		Eigen::MatrixXd samples, mapped, p0, p1, p;
		auto &curret_bases =  iso_parametric ? bases : geom_bases;

		if(mesh->is_volume())
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


			for(std::size_t i = 0; i < curret_bases.size(); ++i){
				if(mesh->n_element_vertices(int(i)) != 8) continue;

				curret_bases[i].eval_geom_mapping(samples, mapped);

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

			for(std::size_t i = 0; i < curret_bases.size(); ++i){
				if(mesh->n_element_vertices(int(i)) != 4) continue;

				curret_bases[i].eval_geom_mapping(samples, mapped);

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

		timer.stop();
		building_basis_time = timer.getElapsedTime();
		std::cout<<" took "<<building_basis_time<<"s"<<std::endl;

		std::cout<<" h: "<<mesh_size<<std::endl;
		std::cout<<"n bases: "<<n_bases<<std::endl;
	}

	void State::compute_assembly_vals()
	{
		igl::Timer timer; timer.start();
		std::cout<<"Computing assembly values..."<<std::flush;

		std::sort(bounday_nodes.begin(), bounday_nodes.end());

		if(iso_parametric){
			ElementAssemblyValues::compute_assembly_values(mesh->is_volume(), bases, values);

			if(mesh->is_volume())
			{
				if(!poly_edge_to_data.empty())
					assert(false);
			}
			else
				PolygonalBasis2d::build_bases(harminic_samples_res, *static_cast<Mesh2D *>(mesh), n_bases, els_tag, quadrature_order, values, values, bases, bases, poly_edge_to_data, polys);

			for(std::size_t e = 0; e < bases.size(); ++e)
			{
				if(els_tag[e] != ElementType::InteriorPolytope && els_tag[e] != ElementType::BoundaryPolytope)
					continue;

				values[e].compute(mesh->is_volume(), bases[e]);
			}
		}
		else
		{
			ElementAssemblyValues::compute_assembly_values(mesh->is_volume(), geom_bases, geom_values);
			ElementAssemblyValues::compute_assembly_values(mesh->is_volume(), bases, values);

			if(mesh->is_volume())
			{
				if(!poly_edge_to_data.empty())
					assert(false);
			}
			else
				PolygonalBasis2d::build_bases(harminic_samples_res, *static_cast<Mesh2D *>(mesh), n_bases, els_tag, quadrature_order, values, geom_values, bases, geom_bases, poly_edge_to_data, polys);

			for(std::size_t e = 0; e < bases.size(); ++e)
			{
				if(els_tag[e] != ElementType::InteriorPolytope && els_tag[e] != ElementType::BoundaryPolytope)
					continue;

				geom_values[e].compute(mesh->is_volume(), geom_bases[e]);
				values[e].compute(mesh->is_volume(), bases[e]);
			}
		}



		timer.stop();
		computing_assembly_values_time = timer.getElapsedTime();
		std::cout<<" took "<<computing_assembly_values_time<<"s"<<std::endl;
	}

	void State::assemble_stiffness_mat()
	{
		igl::Timer timer; timer.start();
		std::cout<<"Assembling stiffness mat..."<<std::flush;

		if(problem.problem_num() == 3)
		{
			Assembler<LinearElasticity> assembler;
			LinearElasticity &le = static_cast<LinearElasticity &>(assembler.local_assembler());
			le.mu() = mu;
			le.lambda() = lambda;
			le.size() = mesh->is_volume()? 3:2;

			if(iso_parametric)
				assembler.assemble(n_bases, values, values, stiffness);
			else
				assembler.assemble(n_bases, values, geom_values, stiffness);

			// std::cout<<MatrixXd(stiffness)<<std::endl;
			assembler.set_identity(bounday_nodes, stiffness);
		}
		else
		{
			Assembler<Laplacian> assembler;
			if(iso_parametric)
				assembler.assemble(n_bases, values, values, stiffness);
			else
				assembler.assemble(n_bases, values, geom_values, stiffness);

			// std::cout<<MatrixXd(stiffness)-MatrixXd(stiffness.transpose())<<"\n\n"<<std::endl;
			// std::cout<<MatrixXd(stiffness).rowwise().sum()<<"\n\n"<<std::endl;

			assembler.set_identity(bounday_nodes, stiffness);
		}

		timer.stop();
		assembling_stiffness_mat_time = timer.getElapsedTime();
		std::cout<<" took "<<assembling_stiffness_mat_time<<"s"<<std::endl;

		nn_zero = stiffness.nonZeros();
		mat_size = stiffness.size();
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
		igl::Timer timer; timer.start();
		std::cout<<"Assigning rhs..."<<std::flush;

		const int size = problem.problem_num() == 3 ? (mesh->is_volume() ? 3:2) : 1;
		RhsAssembler rhs_assembler;
		if(iso_parametric)
		{
			rhs_assembler.assemble(n_bases, size, values, values, problem, rhs);
			rhs *= -1;
			rhs_assembler.set_bc(size, bases, bases, mesh->is_volume(), local_boundary, bounday_nodes, n_boundary_samples, problem, rhs);
		}
		else
		{
			rhs_assembler.assemble(n_bases, size, values, geom_values, problem, rhs);
			rhs *= -1;
			rhs_assembler.set_bc(size, bases, geom_bases, mesh->is_volume(), local_boundary, bounday_nodes, n_boundary_samples, problem, rhs);
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
		igl::Timer timer; timer.start();
		std::cout<<"Solving ";

// #ifndef POLY_FEM_WITH_SUPERLU
// 		typedef SparseMatrix<double> SolverMat;
// 		SuperLU<SolverMat> solver;
// 		std::cout<<"with SuperLU direct solver..."<<std::flush;

// 		solver.compute(SolverMat(stiffness));
// 		sol = solver.solve(rhs);
// #else // POLY_FEM_WITH_SUPERLU
// #ifdef POLY_FEM_WITH_UMFPACK
// 		UmfPackLU<SparseMatrix<double, Eigen::RowMajor> > solver;
// 		std::cout<<"with UmfPack direct solver..."<<std::flush;

// 		solver.compute(stiffness);
// 		sol = solver.solve(rhs);
// #else //POLY_FEM_WITH_UMFPACK
		BiCGSTAB<SparseMatrix<double, Eigen::RowMajor> > solver;
		std::cout<<"with BiCGSTAB iterative solver..."<<std::flush;

		Eigen::saveMarket(stiffness, "A.mat");
		Eigen::saveMarketVector(rhs, "b.mat");
		sol = solver.compute(stiffness).solve(rhs);
// #endif //POLY_FEM_WITH_UMFPACK
// #endif  //POLY_FEM_WITH_SUPERLU

		timer.stop();
		solving_time = timer.getElapsedTime();
		std::cout<<" took "<<solving_time<<"s"<<std::endl;
		std::cout<<"Solver error: "<<(stiffness*sol-rhs).norm()<<std::endl;

		// {
		// 	std::ofstream of;
		// 	of.open("sol.txt");
		// 	of.precision(100);
		// 	of<<sol;
		// 	of.close();
		// }
	}

	void State::compute_errors()
	{
		if(!problem.has_exact_sol()) return;

		igl::Timer timer; timer.start();
		std::cout<<"Computing errors..."<<std::flush;
		using std::max;

		const int n_el=int(values.size());

		MatrixXd v_exact, v_approx;

		errors.clear();

		l2_err = 0;
		linf_err = 0;
		lp_err = 0;

		for(int e = 0; e < n_el; ++e)
		{
			const auto &vals    = values[e];
			const auto &gvalues = iso_parametric ? values[e] : geom_values[e];

			problem.exact(gvalues.val, v_exact);

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
			l2_err += (err.array() * err.array() * gvalues.det.array() * vals.quadrature.weights.array()).sum();
			lp_err += (err.array().pow(8.) * gvalues.det.array() * vals.quadrature.weights.array()).sum();
		}

		l2_err = sqrt(fabs(l2_err));
		lp_err = pow(fabs(lp_err), 1./8.);

		timer.stop();
		computing_errors_time = timer.getElapsedTime();
		std::cout<<" took "<<computing_errors_time<<"s"<<std::endl;

		std::cout<<l2_err<<" "<<linf_err<<" "<<lp_err<<std::endl;
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
		igl::serialize(values, "values", file_name);
		igl::serialize(bounday_nodes, "bounday_nodes", file_name);
		igl::serialize(local_boundary, "local_boundary", file_name);

		igl::serialize(boundary_tag, "boundary_tag", file_name);


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

}
