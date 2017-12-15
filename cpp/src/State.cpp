#include "State.hpp"

#include "HexBasis.hpp"
#include "QuadBasis.hpp"
#include "Spline2dBasis.hpp"
#include "HexQuadrature.hpp"
#include "QuadQuadrature.hpp"


#include "Assembler.hpp"
#include "Laplacian.hpp"
#include "LinearElasticity.hpp"

#include "json.hpp"

#include <igl/Timer.h>

#include <iostream>

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

		j["nn_zero"] = nn_zero;
		j["mat_size"] = mat_size;


		std::ofstream o(name);
		o << std::setw(4) << j << std::endl;
		o.close();

	}

	void State::interpolate_function(const MatrixXd &fun, const MatrixXd &local_pts, MatrixXd &result)
	{
		MatrixXd tmp;

		int actual_dim = 1;
		if(linear_elasticity)
			actual_dim = mesh.is_volume() ? 3:2;

		result.resize(local_pts.rows() * mesh.n_elements(), actual_dim);

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
		igl::Timer timer; timer.start();
		std::cout<<"Loading mesh..."<<std::flush;

		mesh.load(mesh_path);
		//TODO refine

		mesh.set_boundary_tags(boundary_tag);
		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;
		mesh_size = mesh.compute_mesh_size();
		std::cout<<" h: "<<mesh_size<<std::endl;
	}

	void State::build_basis()
	{
		igl::Timer timer; timer.start();
		std::cout<<"Building basis..."<<std::flush;

		local_boundary.clear();
		bounday_nodes.clear();

		if(mesh.is_volume())
		{
			if(use_splines)
				assert(false);
			else
				n_bases = HexBasis::build_bases(mesh, quadrature_order, bases, local_boundary, bounday_nodes);
		}
		else
		{
			if(use_splines)
				n_bases = Spline2dBasis::build_bases(mesh, quadrature_order, bases, local_boundary, bounday_nodes, polys);
			else
				n_bases = QuadBasis::build_bases(mesh, quadrature_order, bases, local_boundary, bounday_nodes);
		}

		problem.remove_neumann_nodes(bases, boundary_tag, local_boundary, bounday_nodes);

		if(linear_elasticity)
		{
			const int dim = mesh.is_volume() ? 3:2;
			const std::size_t n_b_nodes = bounday_nodes.size();

			for(std::size_t i = 0; i < n_b_nodes; ++i)
			{
				bounday_nodes[i] *= dim;
				for(int d = 1; d < dim; ++d)
					bounday_nodes.push_back(bounday_nodes[i]+d);
			}
		}

		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

		std::cout<<"n bases: "<<n_bases<<std::endl;
	}


	template<class Assembler>
	void compute_assembly_values(const Mesh &mesh, const std::vector< ElementBases > &bases, std::vector< ElementAssemblyValues > &values)
	{
		Assembler assmbler;
		assmbler.compute_assembly_values(mesh.is_volume(), bases, values);
	}

	void State::compute_assembly_vals()
	{
		igl::Timer timer; timer.start();
		std::cout<<"Computing assembly values..."<<std::flush;

		std::sort(bounday_nodes.begin(), bounday_nodes.end());

		if(linear_elasticity)
			compute_assembly_values<Assembler<LinearElasticity> >(mesh, bases, values);
		else
			compute_assembly_values<Assembler<Laplacian> >(mesh, bases, values);

		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;
	}

	void State::assemble_stiffness_mat()
	{
		igl::Timer timer; timer.start();
		std::cout<<"Assembling stiffness mat..."<<std::flush;

    	// std::cout<<MatrixXd(stiffness)-MatrixXd(stiffness.transpose())<<"\n\n"<<std::endl;
    	// std::cout<<MatrixXd(stiffness).rowwise().sum()<<"\n\n"<<std::endl;

		if(linear_elasticity)
		{
			Assembler<LinearElasticity> assembler;
			assembler.local_assembler().size() = mesh.is_volume() ? 3:2;
			//todo set lame parameters

			assembler.assemble(n_bases, values, values, stiffness);
			assembler.set_identity(bounday_nodes, stiffness);
		}
		else
		{
			Assembler<Laplacian> assembler;
			assembler.assemble(n_bases, values, values, stiffness);
			// std::cout<<MatrixXd(stiffness)<<std::endl;
			assembler.set_identity(bounday_nodes, stiffness);
		}

		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

		nn_zero = stiffness.nonZeros();
		mat_size = stiffness.size();
		std::cout<<"sparsity: "<<nn_zero<<"/"<<mat_size<<std::endl;
	}

	void State::assemble_rhs()
	{
		igl::Timer timer; timer.start();
		std::cout<<"Assigning rhs..."<<std::flush;

		if(linear_elasticity)
		{
			Assembler<LinearElasticity> assembler;
			assembler.local_assembler().size() = mesh.is_volume() ? 3:2;
			assembler.rhs(n_bases, values, values, problem, rhs);
			rhs *= -1;
			assembler.bc(bases, mesh, local_boundary, bounday_nodes, n_boundary_samples, problem, rhs);
		}
		else
		{
			Assembler<Laplacian> assembler;
			assembler.rhs(n_bases, values, values, problem, rhs);
			rhs *= -1;
			assembler.bc(bases, mesh, local_boundary, bounday_nodes, n_boundary_samples, problem, rhs);
		}

		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;
	}

	void State::solve_problem()
	{
		igl::Timer timer; timer.start();
		std::cout<<"Solving..."<<std::flush;

		BiCGSTAB<SparseMatrix<double, Eigen::RowMajor> > solver;
    		// SparseLU<SparseMatrix<double, Eigen::RowMajor> > solver;
		sol = solver.compute(stiffness).solve(rhs);

		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;
	}

	void State::compute_errors()
	{
		if(!problem.has_exact_sol()) return;

		igl::Timer timer; timer.start();
		std::cout<<"Computing errors..."<<std::flush;
		using std::max;

		const int n_el=int(values.size());

		MatrixXd v_exact, v_approx;

		l2_err = 0;
		linf_err = 0;

		for(int e = 0; e < n_el; ++e)
		{
			auto vals    = values[e];
			auto gvalues = values[e];

			problem.exact(gvalues.val, v_exact);

			v_approx = MatrixXd::Zero(v_exact.rows(), v_exact.cols());

			const int n_loc_bases=int(vals.basis_values.size());

			for(int i = 0; i < n_loc_bases; ++i)
			{
				auto val=vals.basis_values[i];

				for(std::size_t ii = 0; ii < val.global.size(); ++ii)
					v_approx += val.global[ii].val * sol(val.global[ii].index) * val.val;
			}

			auto err = (v_exact-v_approx).cwiseAbs();

			linf_err = max(linf_err, err.maxCoeff());
			l2_err += (err.array() * err.array() * gvalues.det.array() * vals.quadrature.weights.array()).sum();
		}

		l2_err = sqrt(fabs(l2_err));

		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

		std::cout<<l2_err<<" "<<linf_err<<std::endl;
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

	void State::sertialize(const std::string &name)
	{

	}

}
