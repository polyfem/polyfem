#include "State.hpp"

#include "HexBasis.hpp"
#include "QuadBasis.hpp"
#include "Spline2dBasis.hpp"
#include "HexQuadrature.hpp"
#include "QuadQuadrature.hpp"


#include "Assembler.hpp"
#include "Laplacian.hpp"
#include "LinearElasticity.hpp"


#include <iostream>

#include <igl/colormap.h>
#include <igl/triangle/triangulate.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/Timer.h>

#include <nanogui/formhelper.h>
#include <nanogui/screen.h>


// ... or using a custom callback
  //       viewer_.ngui->addVariable<bool>("bool",[&](bool val) {
  //     boolVariable = val; // set
  // },[&]() {
  //     return boolVariable; // get
  // });


using namespace Eigen;


namespace poly_fem
{
	void compute_errors(const std::vector< ElementAssemblyValues > &values, const std::vector< ElementAssemblyValues > &geom_values, const Problem &problem, const Eigen::MatrixXd &sol, double &l2_err, double &linf_err)
	{
		using std::max;

		const int n_el=int(values.size());

		MatrixXd v_exact, v_approx;

		l2_err = 0;
		linf_err = 0;

		for(int e = 0; e < n_el; ++e)
		{
			auto vals    = values[e];
			auto gvalues = geom_values[e];

			problem.exact(gvalues.val, v_exact);

			v_approx = MatrixXd::Zero(v_exact.rows(), v_exact.cols());

			const int n_loc_bases=int(vals.basis_values.size());

			for(int i = 0; i < n_loc_bases; ++i)
			{
				auto val=vals.basis_values[i];

				v_approx = v_approx + sol(val.global_index) * val.val;
			}

			auto err = (v_exact-v_approx).cwiseAbs();

			linf_err = max(linf_err, err.maxCoeff());
			l2_err += (err.array() * err.array() * gvalues.det.array() * vals.quadrature.weights.array()).sum();
		}

		l2_err = sqrt(fabs(l2_err));
	}



	void compute_assembly_values(const bool use_hex, const int quadrature_order, const std::vector< std::vector<Basis> > &bases, std::vector< ElementAssemblyValues > &values)
	{
		values.resize(bases.size());

		Quadrature quadrature;
		if(use_hex)
		{
			HexQuadrature quad_quadrature;
			quad_quadrature.get_quadrature(quadrature_order, quadrature);
		}
		else
		{
			QuadQuadrature quad_quadrature;
			quad_quadrature.get_quadrature(quadrature_order, quadrature);
		}

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
					if(use_hex)
						dzmv.row(k) += val.grad(k,2) * b.node();
				}
			}

			if(use_hex)
				vals.finalize(mval, dxmv, dymv, dzmv);
			else
				vals.finalize(mval, dxmv, dymv);
		}
	}


	void State::interpolate_function(const MatrixXd &fun, MatrixXd &result)
	{
		MatrixXd tmp;

		int actual_dim = 1;
		if(linear_elasticity)
			actual_dim = mesh.is_volume() ? 3:2;

		result.resize(vis_pts.rows(), actual_dim);

		for(std::size_t i = 0; i < bases.size(); ++i)
		{
			const std::vector<Basis> &bs = bases[i];

			MatrixXd local_res = MatrixXd::Zero(local_vis_pts.rows(), actual_dim);

			for(std::size_t j = 0; j < bs.size(); ++j)
			{
				const Basis &b = bs[j];

				b.basis(local_vis_pts, tmp);
				for(int d = 0; d < actual_dim; ++d)
					local_res.col(d) += tmp * fun(b.global_index()*actual_dim + d);
			}

			result.block(i*local_vis_pts.rows(), 0, local_vis_pts.rows(), actual_dim) = local_res;
		}
	}


	void State::plot_function(const MatrixXd &fun, double min, double max)
	{
		MatrixXd col;

		if(linear_elasticity)
		{
			const MatrixXd ffun = (fun.array()*fun.array()).colwise().sum().sqrt(); //norm of displacement, maybe replace with stress

			if(min < max)
				igl::colormap(igl::COLOR_MAP_TYPE_INFERNO, fun, min, max, col);
			else
				igl::colormap(igl::COLOR_MAP_TYPE_INFERNO, fun, true, col);

			MatrixXd tmp = vis_pts;

			for(long i = 0; i < fun.cols(); ++i) //apply displacement
				tmp.col(i) += fun.col(i);

			viewer.data.set_mesh(tmp, vis_faces);
		}
		else
		{

			if(min < max)
				igl::colormap(igl::COLOR_MAP_TYPE_INFERNO, fun, min, max, col);
			else
				igl::colormap(igl::COLOR_MAP_TYPE_INFERNO, fun, true, col);

			if(mesh.is_volume())
				viewer.data.set_mesh(vis_pts, vis_faces);
			else
			{
				MatrixXd tmp;
				tmp.resize(fun.rows(),3);
				tmp.col(0)=vis_pts.col(0);
				tmp.col(1)=vis_pts.col(1);
				tmp.col(2)=fun;
				viewer.data.set_mesh(tmp, vis_faces);
			}
		}

		viewer.data.set_colors(col);
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

		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		//UI
		///////////////////////////////////////////////////////
		auto clear_func = [&](){viewer.data.clear(); };

		auto show_mesh_func = [&](){
			clear_func();
			viewer.data.set_mesh(tri_pts, tri_faces);
		};

		auto show_vis_mesh_func = [&](){
			clear_func();
			viewer.data.set_mesh(vis_pts, vis_faces);
		};

		auto show_nodes_func = [&](){
			// for(std::size_t i = 0; i < bounday_nodes.size(); ++i)
			// 	std::cout<<bounday_nodes[i]<<std::endl;

			for(std::size_t i = 0; i < bases.size(); ++i)
			{
				const std::vector<Basis> &basis = bases[i];
				for(std::size_t j = 0; j < basis.size(); ++j)
				{
					MatrixXd nn = MatrixXd::Zero(basis[j].node().rows(), 3);
					nn.block(0, 0, nn.rows(), basis[j].node().cols()) = basis[j].node();

					VectorXd txt_p = nn.row(0);
					for(long k = 0; k < txt_p.size(); ++k)
						txt_p(k) += 0.02;

					MatrixXd col = MatrixXd::Zero(basis[j].node().rows(), 3);
					if(std::find(bounday_nodes.begin(), bounday_nodes.end(), basis[j].global_index()) != bounday_nodes.end())
						col.col(0).setOnes();


					viewer.data.add_points(nn, col);
					viewer.data.add_label(txt_p, std::to_string(basis[j].global_index()));
				}
			}
		};

		auto show_quadrature_func = [&](){
			for(std::size_t i = 0; i < values.size(); ++i)
			{
				const ElementAssemblyValues &vals = values[i];
				viewer.data.add_points(vals.val, MatrixXd::Zero(vals.val.rows(), 3));

				for(long j = 0; j < vals.val.rows(); ++j)
					viewer.data.add_label(vals.val.row(j), std::to_string(j));
			}
		};

		auto show_rhs_func = [&](){
			MatrixXd global_rhs;
			interpolate_function(rhs, global_rhs);

			plot_function(global_rhs, 0, 1);
		};


		auto show_sol_func = [&](){
			MatrixXd global_sol;
			interpolate_function(sol, global_sol);
			plot_function(global_sol, 0, 1);
		};


		auto show_error_func = [&](){

			MatrixXd global_sol;
			interpolate_function(sol, global_sol);

			MatrixXd exact_sol;
			problem.exact(vis_pts, exact_sol);

			const MatrixXd err = (global_sol - exact_sol).array().abs();
			plot_function(err);
		};


		auto show_basis_func = [&](){
			if(vis_basis < 0 || vis_basis >= n_bases) return;

			MatrixXd fun = MatrixXd::Zero(n_bases, 1);
			fun(vis_basis) = 1;

			MatrixXd global_fun;
			interpolate_function(fun, global_fun);
			// global_fun /= 100;
			plot_function(global_fun, 0, 1.);
		};


		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		//Algo phases
		///////////////////////////////////////////////////////
		auto build_vis_mesh_func = [&](){
			igl::Timer timer; timer.start();
			std::cout<<"Building vis mesh..."<<std::flush;

			if(mesh.is_volume())
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

				clear_func();

				MatrixXi tets;
				igl::copyleft::tetgen::tetrahedralize(pts, faces, "Qpq1.414a0.001", local_vis_pts, tets, local_vis_faces);
			}
			else
			{
				MatrixXd pts(4,2); pts <<
				0,0,
				0,1,
				1,1,
				1,0;

				MatrixXi E(4,2); E <<
				0,1,
				1,2,
				2,3,
				3,0;

				MatrixXd H(0,2);
				igl::triangle::triangulate(pts, E, H, "Qqa0.001", local_vis_pts, local_vis_faces);
			}

			vis_pts.resize(local_vis_pts.rows()*mesh.n_elements(), local_vis_pts.cols());
			vis_faces.resize(local_vis_faces.rows()*mesh.n_elements(), 3);

			MatrixXd mapped, tmp;
			for(std::size_t i = 0; i < bases.size(); ++i)
			{
				const std::vector<Basis> &bs = bases[i];
				Basis::eval_geom_mapping(local_vis_pts, bs, mapped);
				vis_pts.block(i*local_vis_pts.rows(), 0, local_vis_pts.rows(), mapped.cols()) = mapped;
				vis_faces.block(i*local_vis_faces.rows(), 0, local_vis_faces.rows(), 3) = local_vis_faces.array() + int(i)*int(local_vis_pts.rows());
			}

			timer.stop();
			std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

			if(skip_visualization) return;

			clear_func();
			show_vis_mesh_func();
		};


		auto load_mesh_func = [&](){
			igl::Timer timer; timer.start();
			std::cout<<"Loading mesh..."<<std::flush;

			mesh.load(mesh_path);
			//TODO refine
			mesh.triangulate_faces(tri_faces, tri_pts);

			timer.stop();
			std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

			if(skip_visualization) return;

			clear_func();
			show_mesh_func();
		};

		auto build_basis_func = [&](){
			igl::Timer timer; timer.start();
			std::cout<<"Building basis..."<<std::flush;

			if(mesh.is_volume())
			{
				if(use_splines)
					assert(false);
				else
					n_bases = HexBasis::build_bases(mesh, bases, bounday_nodes);
			}
			else
			{
				if(use_splines)
					n_bases = Spline2dBasis::build_bases(mesh, bases, bounday_nodes);
				else
					n_bases = QuadBasis::build_bases(mesh, bases, bounday_nodes);
			}
			timer.stop();
			std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

			std::cout<<"n bases: "<<n_bases<<std::endl;

			if(skip_visualization) return;
			clear_func();
			show_mesh_func();
			show_nodes_func();
		};


		auto compute_assembly_vals_func = [&]() {
			igl::Timer timer; timer.start();
			std::cout<<"Computing assembly values..."<<std::flush;

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

			std::sort(bounday_nodes.begin(), bounday_nodes.end());



			compute_assembly_values(mesh.is_volume(), quadrature_order, bases, values);

			timer.stop();
			std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

			if(skip_visualization) return;
			clear_func();
			show_mesh_func();
			show_quadrature_func();
		};

		auto assemble_stiffness_mat_func = [&]() {
			igl::Timer timer; timer.start();
			std::cout<<"Assembling stiffness mat..."<<std::flush;

    		// std::cout<<MatrixXd(stiffness)-MatrixXd(stiffness.transpose())<<"\n\n"<<std::endl;
    		// std::cout<<MatrixXd(stiffness).rowwise().sum()<<"\n\n"<<std::endl;

			if(linear_elasticity)
			{
				Assembler<LinearElasticity> assembler;
				assembler.local_assembler().size() = mesh.is_volume() ? 3:2;

				// std::cout<<stiffness.rows()<<std::endl;
				// for(std::size_t i = 0; i < bounday_nodes.size(); ++i)
				// 	std::cout<<bounday_nodes[i]<<std::endl;

				assembler.assemble(n_bases, values, values, stiffness);
				assembler.set_identity(bounday_nodes, stiffness);
			}
			else
			{
				Assembler<Laplacian> assembler;
				assembler.assemble(n_bases, values, values, stiffness);
				assembler.set_identity(bounday_nodes, stiffness);
			}

			timer.stop();
			std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

			nn_zero = stiffness.nonZeros();
			mat_size = stiffness.size();
			std::cout<<"sparsity: "<<nn_zero<<"/"<<mat_size<<std::endl;
		};


		auto assemble_rhs_func = [&]() {
			igl::Timer timer; timer.start();
			std::cout<<"Assigning rhs..."<<std::flush;

			if(linear_elasticity)
			{
				Assembler<LinearElasticity> assembler;
				assembler.local_assembler().size() = mesh.is_volume() ? 3:2;
				assembler.rhs(n_bases, values, values, problem, rhs);
				rhs *= -1;
				assembler.bc(bases, mesh, bounday_nodes, n_boundary_samples, problem, rhs);
			}
			else
			{
				Assembler<Laplacian> assembler;
				assembler.rhs(n_bases, values, values, problem, rhs);
				rhs *= -1;
				assembler.bc(bases, mesh, bounday_nodes, n_boundary_samples, problem, rhs);
			}

			timer.stop();
			std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

			if(skip_visualization) return;
			clear_func();
			show_rhs_func();
		};

		auto solve_problem_func = [&]() {
			igl::Timer timer; timer.start();
			std::cout<<"Solving..."<<std::flush;

			BiCGSTAB<SparseMatrix<double, Eigen::RowMajor> > solver;
    		// SparseLU<SparseMatrix<double, Eigen::RowMajor> > solver;
			sol = solver.compute(stiffness).solve(rhs);

			timer.stop();
			std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

			if(skip_visualization) return;
			clear_func();
			show_sol_func();
		};

		auto compute_errors_func = [&]() {
			if(linear_elasticity) return;

			igl::Timer timer; timer.start();
			std::cout<<"Computing errors..."<<std::flush;

			compute_errors(values, values, problem, sol, l2_err, linf_err);

			timer.stop();
			std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

			std::cout<<l2_err<<" "<<linf_err<<std::endl;

			if(skip_visualization) return;
			clear_func();
			show_error_func();
		};




		viewer.callback_init = [&](igl::viewer::Viewer& viewer_)
		{
			viewer_.ngui->addWindow(Eigen::Vector2i(220,10),"PolyFEM");

			viewer_.ngui->addGroup("Settings");

			viewer_.ngui->addVariable("quad order", quadrature_order);
			viewer_.ngui->addVariable("b samples", n_boundary_samples);

			viewer_.ngui->addVariable("mesh path", mesh_path);
			viewer_.ngui->addVariable("n refs", n_refs);

			viewer_.ngui->addVariable("spline basis", use_splines);

			viewer_.ngui->addVariable("elasticity", linear_elasticity);

			viewer_.ngui->addVariable<ProblemType>("Problem",
				[&](ProblemType val) { problem.set_problem_num(val); },
				[&]() { return ProblemType(problem.problem_num()); }
				)->setItems({"Linear","Quadratic","Franke", "Elastic"});

			viewer_.ngui->addVariable("skip visualization", skip_visualization);

			viewer_.ngui->addGroup("Runners");
			viewer_.ngui->addButton("Load mesh", load_mesh_func);
			viewer_.ngui->addButton("Build  basis", build_basis_func);
			viewer_.ngui->addButton("Compute vals", compute_assembly_vals_func);
			viewer_.ngui->addButton("Build vis mesh", build_vis_mesh_func);

			viewer_.ngui->addButton("Assemble stiffness", assemble_stiffness_mat_func);
			viewer_.ngui->addButton("Assemble rhs", assemble_rhs_func);
			viewer_.ngui->addButton("Solve", solve_problem_func);
			viewer_.ngui->addButton("Compute errors", compute_errors_func);

			viewer_.ngui->addButton("Run all", [&](){
				load_mesh_func();
				build_basis_func();

				if(!skip_visualization)
					build_vis_mesh_func();

				compute_assembly_vals_func();
				assemble_stiffness_mat_func();
				assemble_rhs_func();
				solve_problem_func();
				compute_errors_func();
			});

			viewer_.ngui->addWindow(Eigen::Vector2i(400,10),"Debug");
			viewer_.ngui->addButton("Clear", clear_func);
			viewer_.ngui->addButton("Show mesh", show_mesh_func);
			viewer_.ngui->addButton("Show vis mesh", show_vis_mesh_func);
			viewer_.ngui->addButton("Show nodes", show_nodes_func);
			viewer_.ngui->addButton("Show quadrature", show_quadrature_func);
			viewer_.ngui->addButton("Show rhs", show_rhs_func);
			viewer_.ngui->addButton("Show sol", show_sol_func);
			viewer_.ngui->addButton("Show error", show_error_func);

			viewer_.ngui->addVariable("basis num",vis_basis);
			viewer_.ngui->addButton("Show basis", show_basis_func);

			// viewer_.ngui->addGroup("Stats");
			// viewer_.ngui->addVariable("NNZ", Type &value)


			viewer_.screen->performLayout();

			return false;
		};

		viewer.launch();
	}

	void State::sertialize(const std::string &name)
	{

	}

}