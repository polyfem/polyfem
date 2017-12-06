#include "State.hpp"

#include "HexBasis.hpp"
#include "QuadBasis.hpp"
#include "Spline2dBasis.hpp"
#include "HexQuadrature.hpp"
#include "QuadQuadrature.hpp"

#include "Assembler.hpp"
#include "Laplacian.hpp"


#include <iostream>

#include <igl/colormap.h>
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

	void build_hex_mesh(const int n_x_el, const int n_y_el, const int n_z_el, Mesh &mesh, std::vector< int > &bounday_nodes)
	{
		const int n_pts = (n_x_el+1)*(n_y_el+1)*(n_z_el+1);
		const int n_els = n_x_el*n_y_el*n_z_el;

		mesh.pts.resize(n_pts, 3);
		mesh.els.resize(n_els, 8);

		for(int k=0; k<=n_z_el;++k)
		{
			for(int j=0; j<=n_y_el;++j)
			{
				for(int i=0; i<=n_x_el;++i)
				{
					const int index = k*(n_x_el+1)*(n_y_el+1)+j*(n_x_el+1)+i;

					if( j == 0 || j == n_y_el || i == 0 || i == n_x_el || k == 0 || k == n_z_el)
						bounday_nodes.push_back(index);

					mesh.pts.row(index)=Vector3d(i,j,k);
				}
			}
		}

		mesh.pts.col(0)/=n_x_el;
		mesh.pts.col(1)/=n_y_el;
		mesh.pts.col(2)/=n_z_el;

		Matrix<int, 1, 8> el;
		for(int k=0; k<n_z_el;++k)
		{
			for(int j=0; j<n_y_el;++j)
			{
				for(int i=0; i<n_x_el;++i)
				{
					const int i1 = k*(n_x_el+1)*(n_y_el+1)+j*(n_x_el+1)+i;
					const int i2 = k*(n_x_el+1)*(n_y_el+1)+j*(n_x_el+1)+i+1;
					const int i3 = k*(n_x_el+1)*(n_y_el+1)+(j+1)*(n_x_el+1)+i+1;
					const int i4 = k*(n_x_el+1)*(n_y_el+1)+(j+1)*(n_x_el+1)+i;

					const int i5 = (k+1)*(n_x_el+1)*(n_y_el+1)+j*(n_x_el+1)+i;
					const int i6 = (k+1)*(n_x_el+1)*(n_y_el+1)+j*(n_x_el+1)+i+1;
					const int i7 = (k+1)*(n_x_el+1)*(n_y_el+1)+(j+1)*(n_x_el+1)+i+1;
					const int i8 = (k+1)*(n_x_el+1)*(n_y_el+1)+(j+1)*(n_x_el+1)+i;

					el << i1, i2, i3, i4, i5, i6, i7, i8;
					mesh.els.row(k*n_x_el*n_y_el+j*n_x_el+i)=el;
				}
			}
		}

		mesh.n_x = n_x_el;
		mesh.n_y = n_y_el;
		mesh.n_z = n_z_el;

		mesh.is_volume = true;
	}

	void triangulate_hex_mesh(const Mesh &mesh, Matrix<int, Dynamic, 3> &vis_faces)
	{
		assert(mesh.els.cols()==8);

		const long n_els = mesh.els.rows();

		const long n_vis_faces = n_els*6*2;
		vis_faces.resize(n_vis_faces, 3);

		long index = 0;
		for (long i = 0; i < n_els; ++i)
		{
			const auto el = mesh.els.row(i);

			vis_faces.row(index++)=Vector3i(el(0),el(1),el(2));
			vis_faces.row(index++)=Vector3i(el(0),el(2),el(3));

			vis_faces.row(index++)=Vector3i(el(4),el(5),el(6));
			vis_faces.row(index++)=Vector3i(el(4),el(6),el(7));

			vis_faces.row(index++)=Vector3i(el(0),el(1),el(5));
			vis_faces.row(index++)=Vector3i(el(0),el(5),el(4));

			vis_faces.row(index++)=Vector3i(el(1),el(2),el(5));
			vis_faces.row(index++)=Vector3i(el(5),el(2),el(6));

			vis_faces.row(index++)=Vector3i(el(3),el(2),el(7));
			vis_faces.row(index++)=Vector3i(el(7),el(2),el(6));

			vis_faces.row(index++)=Vector3i(el(0),el(3),el(4));
			vis_faces.row(index++)=Vector3i(el(4),el(3),el(7));
		}
	}


	void build_quad_mesh(const int n_x_el, const int n_y_el, Mesh &mesh, std::vector< int > &bounday_nodes)
	{
		const int n_pts = (n_x_el+1)*(n_y_el+1);
		const int n_els = n_x_el*n_y_el;

		mesh.pts.resize(n_pts, 2);
		mesh.els.resize(n_els, 4);

		for(int j=0; j<=n_y_el;++j)
		{
			for(int i=0; i<=n_x_el;++i)
			{
				const int index = j*(n_x_el+1)+i;

				if( j == 0 || j == n_y_el || i == 0 || i == n_x_el)
					bounday_nodes.push_back(index);

				mesh.pts.row(index)=Vector2d(i, j);
			}
		}

		mesh.pts.col(0)/=n_x_el;
		mesh.pts.col(1)/=n_y_el;

		Matrix<int, 1, 4> el;

		for(int j=0; j<n_y_el;++j)
		{
			for(int i=0; i<n_x_el;++i)
			{
				const int i1 = j*(n_x_el+1)+i;
				const int i2 = j*(n_x_el+1)+i+1;
				const int i3 = (j+1)*(n_x_el+1)+i+1;
				const int i4 = (j+1)*(n_x_el+1)+i;

				el << i1, i2, i3, i4;
				mesh.els.row(j*n_x_el+i)=el;
			}
		}

		mesh.n_x = n_x_el;
		mesh.n_y = n_y_el;

		mesh.is_volume = false;
	}

	void triangulate_quad_mesh(const Mesh &mesh, Matrix<int, Dynamic, 3> &vis_faces)
	{
		assert(mesh.els.cols()==4);
		const long n_els = mesh.els.rows();

		const long n_vis_faces = n_els*2;
		vis_faces.resize(n_vis_faces, 3);

		long index = 0;
		for (long i = 0; i < n_els; ++i)
		{
			const auto el = mesh.els.row(i);

			vis_faces.row(index++)=Vector3i(el(0),el(1),el(2));
			vis_faces.row(index++)=Vector3i(el(0),el(2),el(3));
		}
	}


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

		l2_err = sqrt(l2_err);
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



	State &State::state(){
		static State instance;

		return instance;
	}

	void State::init(const int n_x, const int n_y, const int n_z, const bool use_hex_, const int problem_num)
	{
		n_x_el = n_x;
		n_y_el = n_y;
		n_z_el = n_z;

		use_hex = use_hex_;

		problem.set_problem_num(problem_num);

		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		//UI
		///////////////////////////////////////////////////////
		auto clear_func = [&](){viewer.data.clear(); };

		auto show_mesh_func = [&](){
			viewer.data.set_mesh(mesh.pts, vis_faces);
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
			const int res = 5;
			MatrixXd pts(res*res, 2); //TODO

			auto t = VectorXd::LinSpaced(res, 0, 1);

			for(int i = 0; i < res; ++i)
			{
				for(int j = 0; j < res; ++j)
				{
					pts.row(i*res+j) = Vector2d(t(i), t(j));
				}
			}

			MatrixXd mapped, tmp;
			for(std::size_t i = 0; i < bases.size(); ++i)
			{
				const std::vector<Basis> &bs = bases[i];
				Basis::eval_geom_mapping(pts, bs, mapped);

				MatrixXd rhs_eval = MatrixXd::Zero(pts.rows(), 1);

				for(std::size_t j = 0; j < bs.size(); ++j)
				{
					const Basis &b = bs[j];

					b.basis(pts, tmp);
					rhs_eval += tmp * rhs(b.global_index());
				}

				MatrixXd toView(mapped.rows(), 3);
				toView.col(0)=mapped.col(0);
				toView.col(1)=mapped.col(1);
				toView.col(2)=rhs_eval;

				MatrixXd col;
				igl::colormap(igl::COLOR_MAP_TYPE_INFERNO, rhs_eval, 0, 1, col);

				viewer.data.add_points(toView, col);
			}
		};


		auto show_sol_func = [&](){
			const int res = 5;
			MatrixXd pts(res*res, 2); //TODO

			auto t = VectorXd::LinSpaced(res, 0, 1);

			for(int i = 0; i < res; ++i)
			{
				for(int j = 0; j < res; ++j)
				{
					pts.row(i*res+j) = Vector2d(t(i), t(j));
				}
			}

			MatrixXd mapped, tmp;
			for(std::size_t i = 0; i < bases.size(); ++i)
			{
				const std::vector<Basis> &bs = bases[i];
				Basis::eval_geom_mapping(pts, bs, mapped);

				MatrixXd sol_eval = MatrixXd::Zero(pts.rows(), 1);

				for(std::size_t j = 0; j < bs.size(); ++j)
				{
					const Basis &b = bs[j];

					b.basis(pts, tmp);
					sol_eval += tmp * sol(b.global_index());
				}

				MatrixXd toView(mapped.rows(), 3);
				toView.col(0)=mapped.col(0);
				toView.col(1)=mapped.col(1);
				toView.col(2)=sol_eval;

				MatrixXd col;
				igl::colormap(igl::COLOR_MAP_TYPE_INFERNO, sol_eval, 0, 1, col);

				viewer.data.add_points(toView, col);
			}
		};


		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		///////////////////////////////////////////////////////
		//Algo phases
		///////////////////////////////////////////////////////
		auto build_mesh_func = [&](){
			bounday_nodes.clear();
			if(use_hex)
			{
				build_hex_mesh(n_x_el, n_y_el, n_z_el, mesh, bounday_nodes);
				triangulate_hex_mesh(mesh, vis_faces);
			}
			else
			{
				build_quad_mesh(n_x_el, n_y_el, mesh, bounday_nodes);
				triangulate_quad_mesh(mesh, vis_faces);
			}

			clear_func();
			show_mesh_func();
		};

		auto build_basis_func = [&](){
			if(mesh.is_volume)
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

			clear_func();
			show_mesh_func();
			show_nodes_func();
		};


		auto compute_assembly_vals_func = [&]() {
			std::sort(bounday_nodes.begin(), bounday_nodes.end());
			compute_assembly_values(use_hex, quadrature_order, bases, values);

			clear_func();
			show_mesh_func();
			show_quadrature_func();
		};

		auto assemble_stiffness_mat_func = [&]() {
			Assembler<Laplacian> assembler;
			assembler.assemble(n_bases, values, values, stiffness);
    		// std::cout<<MatrixXd(stiffness)-MatrixXd(stiffness.transpose())<<"\n\n"<<std::endl;
    		// std::cout<<MatrixXd(stiffness).rowwise().sum()<<"\n\n"<<std::endl;
			assembler.set_identity(bounday_nodes, stiffness);
		};


		auto assemble_rhs_func = [&]() {
			Assembler<Laplacian> assembler;
			assembler.rhs(n_bases, values, values, problem, rhs);
			rhs *= -1;
			assembler.bc(bases, mesh, bounday_nodes, n_boundary_samples, problem, rhs);
    		// std::cout<<rhs<<"\n\n"<<std::endl;

			clear_func();
			show_mesh_func();
			show_rhs_func();
		};

		auto solve_problem_func = [&]() {
			BiCGSTAB<SparseMatrix<double, Eigen::RowMajor> > solver;
    		// SparseLU<SparseMatrix<double, Eigen::RowMajor> > solver;
			sol = solver.compute(stiffness).solve(rhs);

			clear_func();
			show_mesh_func();
			show_sol_func();
		};

		auto compute_errors_func = [&]() {
			compute_errors(values, values, problem, sol, l2_err, linf_err);
			std::cout<<l2_err<<" "<<linf_err<<std::endl;
		};




		viewer.callback_init = [&](igl::viewer::Viewer& viewer_)
		{
			viewer_.ngui->addWindow(Eigen::Vector2i(220,10),"PolyFEM");

			viewer_.ngui->addGroup("Settings");

			viewer_.ngui->addVariable("quad order", quadrature_order);
			viewer_.ngui->addVariable("b samples", n_boundary_samples);

			viewer_.ngui->addVariable("n_x",n_x_el);
			viewer_.ngui->addVariable("n_y",n_y_el);
			viewer_.ngui->addVariable("n_z",n_z_el);

			viewer_.ngui->addVariable("hex mesh", use_hex);
			viewer_.ngui->addVariable("spline basis", use_splines);

			viewer_.ngui->addVariable<ProblemType>("Problem",
				[&](ProblemType val) { problem.set_problem_num(val); },
				[&]() { return ProblemType(problem.problem_num()); }
				)->setItems({"Linear","Quadratic","Franke"});



			viewer_.ngui->addGroup("Runners");
			viewer_.ngui->addButton("Build  mesh", build_mesh_func);
			viewer_.ngui->addButton("Build  basis", build_basis_func);
			viewer_.ngui->addButton("Compute vals", compute_assembly_vals_func);
			viewer_.ngui->addButton("Assemble stiffness", assemble_stiffness_mat_func);
			viewer_.ngui->addButton("Assemble rhs", assemble_rhs_func);
			viewer_.ngui->addButton("Solve", solve_problem_func);
			viewer_.ngui->addButton("compute errors", compute_errors_func);


			viewer_.ngui->addGroup("Debug");
			viewer_.ngui->addButton("Clear", clear_func);
			viewer_.ngui->addButton("Show mesh", show_mesh_func);
			viewer_.ngui->addButton("Show nodes", show_nodes_func);
			viewer_.ngui->addButton("Show quadrature", show_quadrature_func);
			viewer_.ngui->addButton("Show rhs", show_rhs_func);
			viewer_.ngui->addButton("Show sol", show_sol_func);


			viewer_.screen->performLayout();

			return false;
		};

		viewer.launch();
	}

	void State::sertialize(const std::string &name)
	{

	}

}