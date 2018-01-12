#include "UIState.hpp"

#include "Mesh2D.hpp"
#include "Mesh3D.hpp"

#include "LinearElasticity.hpp"

#include "LinearSolver.hpp"

#include <igl/per_face_normals.h>
#include <igl/per_corner_normals.h>
#include <igl/triangle/triangulate.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/Timer.h>


#include <nanogui/formhelper.h>
#include <nanogui/screen.h>

#include <stdlib.h>


// ... or using a custom callback
  //       viewer_.ngui->addVariable<bool>("bool",[&](bool val) {
  //     boolVariable = val; // set
  // },[&]() {
  //     return boolVariable; // get
  // });


using namespace Eigen;



namespace poly_fem
{

	namespace
	{
		Navigation3D::Index current_3d_index;

		const std::vector<std::string> explode(const std::string &s, const char &c)
		{
			std::string buff{""};
			std::vector<std::string> v;

			for(auto n: s)
			{
				if(n != c) buff+=n; else
				if(n == c && buff != "") { v.push_back(buff); buff = ""; }
			}
			if(buff != "") v.push_back(buff);

			return v;
		}
	}

	void UIState::plot_selection_and_index(const bool recenter)
	{
		std::vector<bool> valid_elements(normalized_barycenter.rows(), false);
		auto v{explode(selected_elements, ',')};
		for(auto idx : v)
			valid_elements[atoi(idx.c_str())] = true;

		viewer.data.clear();

		if(current_visualization == Visualizing::InputMesh)
		{
			const long n_tris = show_clipped_elements(tri_pts, tri_faces, element_ranges, valid_elements, recenter);
			color_mesh(n_tris, valid_elements);
		}
		else
		{
			show_clipped_elements(vis_pts, vis_faces, vis_element_ranges, valid_elements, recenter);
		}

		if(state.mesh->is_volume())
		{
			const auto p = state.mesh->point(current_3d_index.vertex);
			const auto p1 = state.mesh->point(static_cast<Mesh3D *>(state.mesh)->switch_vertex(current_3d_index).vertex);

			viewer.data.add_points(p, MatrixXd::Zero(1, 3));
			viewer.data.add_edges(p, p1, RowVector3d(1, 1, 0));

			viewer.data.add_points(static_cast<Mesh3D *>(state.mesh)->face_barycenter(current_3d_index.face), RowVector3d(1, 0, 0));
		}
	}

	void UIState::color_mesh(const int n_tris, const std::vector<bool> &valid_elements)
	{
		const std::vector<ElementType> &ele_tag = state.mesh->elements_tag();

		Eigen::MatrixXd cols(n_tris, 3);
		cols.setZero();

		int from = 0;
		for(std::size_t i = 1; i < element_ranges.size(); ++i)
		{
			if(!valid_elements[i-1]) continue;

			const ElementType type = ele_tag[i-1];
			const int range = element_ranges[i]-element_ranges[i-1];

			switch(type)
			{
					//green
				case ElementType::RegularInteriorCube:
				cols.block(from, 1, range, 1).setOnes(); break;

					//dark green
				case ElementType::RegularBoundaryCube:
				cols.block(from, 1, range, 1).setConstant(0.5); break;

					//yellow
				case ElementType::SimpleSingularInteriorCube:
				cols.block(from, 0, range, 1).setOnes();
				cols.block(from, 1, range, 1).setOnes(); break;

					//orange
				case ElementType::SimpleSingularBoundaryCube:
				cols.block(from, 0, range, 1).setOnes();
				cols.block(from, 1, range, 1).setConstant(0.5); break;

 						//red
				case ElementType::MultiSingularInteriorCube:
				cols.block(from, 0, range, 1).setOnes(); break;

						//blue
				case ElementType::MultiSingularBoundaryCube:
				cols.block(from, 2, range, 1).setConstant(0.6); break;

				  		 //light blue
				case ElementType::BoundaryPolytope:
				case ElementType::InteriorPolytope:
				cols.block(from, 2, range, 1).setOnes();
				cols.block(from, 1, range, 1).setConstant(0.5); break;

					//grey
				case ElementType::Undefined:
				cols.block(from, 0, range, 3).setConstant(0.5); break;
			}

			from += range;
		}

		viewer.data.set_colors(cols);
	}

	long UIState::clip_elements(const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tris, const std::vector<int> &ranges, std::vector<bool> &valid_elements)
	{
		viewer.data.clear();

		valid_elements.resize(normalized_barycenter.rows());

		if(!is_slicing)
		{
			std::fill(valid_elements.begin(), valid_elements.end(), true);
			viewer.data.set_mesh(pts, tris);
			
			if(state.mesh->is_volume())
			{
				MatrixXd normals;
				igl::per_face_normals(pts, tris, normals);
				viewer.data.set_normals(normals);

				igl::per_corner_normals(pts, tris, 20, normals);
				viewer.data.set_normals(normals);
				viewer.data.set_face_based(false);
			}

			return tris.rows();
		}

		for (long i = 0; i<normalized_barycenter.rows();++i)
			valid_elements[i] = normalized_barycenter(i, slice_coord) < slice_position;

		return show_clipped_elements(pts, tris, ranges, valid_elements);
	}

	long UIState::show_clipped_elements(const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tris, const std::vector<int> &ranges, const std::vector<bool> &valid_elements, const bool recenter)
	{
		viewer.data.set_face_based(false);

		int n_vis_valid_tri = 0;

		for (long i = 0; i<normalized_barycenter.rows();++i)
		{
			if(valid_elements[i])
				n_vis_valid_tri += ranges[i+1] - ranges[i];
		}

		MatrixXi valid_tri(n_vis_valid_tri, tris.cols());

		int from = 0;
		for(std::size_t i = 1; i < ranges.size(); ++i)
		{
			if(!valid_elements[i-1]) continue;

			const int range = ranges[i]-ranges[i-1];

			valid_tri.block(from, 0, range, tri_faces.cols()) = tris.block(ranges[i-1], 0, range, tris.cols());

			from += range;
		}



		viewer.data.set_mesh(pts, valid_tri);
		
		if(state.mesh->is_volume())
		{
			MatrixXd normals;
			igl::per_face_normals(pts, valid_tri, normals);
			viewer.data.set_normals(normals);

			igl::per_corner_normals(pts, valid_tri, 20, normals);
			viewer.data.set_normals(normals);
			viewer.data.set_face_based(false);
		}

		if(recenter)
			viewer.core.align_camera_center(pts, valid_tri);
		return valid_tri.rows();
	}

	void UIState::interpolate_function(const MatrixXd &fun, MatrixXd &result)
	{
		MatrixXd tmp;

		int actual_dim = 1;
		if(state.problem.problem_num() == 3)
			actual_dim = state.mesh->is_volume() ? 3:2;

		result.resize(vis_pts.rows(), actual_dim);

		int index = 0;

		for(int i = 0; i < int(state.bases.size()); ++i)
		{
			const ElementBases &bs = state.bases[i];
			MatrixXd local_pts;

			if(state.mesh->is_cube(i))
				local_pts = local_vis_pts_quad;
			else
				local_pts = vis_pts_poly[i];

			MatrixXd local_res = MatrixXd::Zero(local_pts.rows(), actual_dim);

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];

				b.basis(local_pts, tmp);
				for(int d = 0; d < actual_dim; ++d)
				{
					for(std::size_t ii = 0; ii < b.global().size(); ++ii)
						local_res.col(d) += b.global()[ii].val * tmp * fun(b.global()[ii].index*actual_dim + d);
				}
			}

			result.block(index, 0, local_res.rows(), actual_dim) = local_res;
			index += local_res.rows();
		}
	}


	UIState::UIState()
	: state(State::state())
	{ }

	void UIState::plot_function(const MatrixXd &fun, double min, double max)
	{
		MatrixXd col;
		std::vector<bool> valid_elements;

		if(state.problem.problem_num() == 3)
		{
			const MatrixXd ffun = (fun.array()*fun.array()).rowwise().sum().sqrt(); //norm of displacement, maybe replace with stress
			// const MatrixXd ffun = fun.col(1); //y component

			// LinearElasticity lin_elast;
			// MatrixXd ffun(vis_pts.rows(), 1);

			// int size = 1;
			// if(state.problem.problem_num() == 3)
			// 	size = state.mesh->is_volume() ? 3:2;

			// MatrixXd stresses;
			// int counter = 0;
			// for(int i = 0; i < int(state.bases.size()); ++i)
			// {
			// 	const ElementBases &bs = state.bases[i];

			// 	MatrixXd local_pts;

			// 	if(is_quad(bs))
			// 		local_pts = local_vis_pts_quad;
			// 	else if(is_tri(bs))
			// 		local_pts = local_vis_pts_tri;
			// 	else{
			// 		local_pts = vis_pts_poly[i];
			// 	}
			// 	lin_elast.compute_von_mises_stresses(size, bs, local_pts, fun, stresses);
			// 	ffun.block(counter, 0, stresses.rows(), stresses.cols()) = stresses;
			// 	counter += stresses.rows();
			// }

			if(min < max)
				igl::colormap(color_map, ffun, min, max, col);
			else
				igl::colormap(color_map, ffun, true, col);

			MatrixXd tmp = vis_pts;

			for(long i = 0; i < fun.cols(); ++i) //apply displacement
				tmp.col(i) += fun.col(i);

			clip_elements(tmp, vis_faces, vis_element_ranges, valid_elements);
		}
		else
		{

			if(min < max)
				igl::colormap(color_map, fun, min, max, col);
			else
				igl::colormap(color_map, fun, true, col);

			if(state.mesh->is_volume())
				clip_elements(vis_pts, vis_faces, vis_element_ranges, valid_elements);
			else
			{
				MatrixXd tmp;
				tmp.resize(fun.rows(),3);
				tmp.col(0)=vis_pts.col(0);
				tmp.col(1)=vis_pts.col(1);
				tmp.col(2)=fun;
				clip_elements(tmp, vis_faces, vis_element_ranges, valid_elements);
			}
		}

		viewer.data.set_colors(col);
	}


	UIState &UIState::ui_state(){
		static UIState instance;

		return instance;
	}

	void UIState::init(const std::string &mesh_path, const int n_refs, const int problem_num)
	{
		state.init(mesh_path, n_refs, problem_num);

		auto clear_func = [&](){ viewer.data.clear(); };

		auto show_mesh_func = [&](){
			clear_func();
			current_visualization = Visualizing::InputMesh;

			std::vector<bool> valid_elements;
			const long n_tris = clip_elements(tri_pts, tri_faces, element_ranges, valid_elements);

			color_mesh(n_tris, valid_elements);

			MatrixXd p0, p1;
			state.mesh->get_edges(p0, p1);
			viewer.data.add_edges(p0, p1, MatrixXd::Zero(1, 3));

			for(int i = 0; i < state.mesh->n_faces(); ++i)
			{
				MatrixXd p = state.mesh->face_barycenter(i);
				viewer.data.add_label(p.transpose(), std::to_string(i));
			}

			// for(int i = 0; i < state.mesh->n_elements(); ++i)
			// {
			// 	MatrixXd p = state.mesh->cell_barycenter(i);
			// 	viewer.data.add_label(p.transpose(), std::to_string(i));
			// }

			// for(int i = 0; i < static_cast<Mesh3D *>(state.mesh)->n_pts(); ++i)
			// {
			// 	MatrixXd p; static_cast<Mesh3D *>(state.mesh)->point(i, p);
			// 	viewer.data.add_label(p.transpose(), std::to_string(i));
			// }

			// for(int i = 0; i < state.mesh->n_elements(); ++i)
			// {
			// 	MatrixXd p = static_cast<Mesh2D *>(state.mesh)->node_from_face(i);
			// 	viewer.data.add_label(p.transpose(), std::to_string(i));
			// }
		};

		auto show_vis_mesh_func = [&](){
			clear_func();
			current_visualization = Visualizing::VisMesh;

			std::cout<<vis_faces.rows()<<" "<<vis_faces.cols()<<std::endl;
			std::vector<bool> valid_elements;
			clip_elements(vis_pts, vis_faces, vis_element_ranges, valid_elements);
		};

		auto show_nodes_func = [&](){
			for(std::size_t i = 0; i < state.bases.size(); ++i)
			{
				const ElementBases &basis = state.bases[i];

				for(std::size_t j = 0; j < basis.bases.size(); ++j)
				{
					for(std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
					{
						const Local2Global &l2g = basis.bases[j].global()[kk];
						int g_index = l2g.index;

						if(state.problem.problem_num() == 3)
							g_index *= 2;

						MatrixXd node = l2g.node;
						MatrixXd col = MatrixXd::Zero(l2g.node.rows(), 3);

						if(std::find(state.boundary_nodes.begin(), state.boundary_nodes.end(), g_index) != state.boundary_nodes.end())
							col.col(0).setOnes();
						else
							col.col(1).setOnes();


						viewer.data.add_points(node, col);
						viewer.data.add_label(node.transpose(), std::to_string(g_index));
					}
				}
			}
		};

		auto show_quadrature_func = [&](){
			for(std::size_t i = 0; i < state.values.size(); ++i)
			{
				const ElementAssemblyValues &vals = state.values[i];
				if(state.mesh->is_volume())
					viewer.data.add_points(vals.val, vals.quadrature.points);
				else
					viewer.data.add_points(vals.val, MatrixXd::Zero(vals.val.rows(), 3));

				// for(long j = 0; j < vals.val.rows(); ++j)
					// viewer.data.add_label(vals.val.row(j), std::to_string(j));
			}
		};

		auto show_rhs_func = [&](){
			current_visualization = Visualizing::Rhs;
			MatrixXd global_rhs;
			state.interpolate_function(state.rhs, local_vis_pts_quad, global_rhs);

			plot_function(global_rhs, 0, 1);
		};


		auto show_sol_func = [&](){
			current_visualization = Visualizing::Solution;
			MatrixXd global_sol;
			interpolate_function(state.sol, global_sol);
			plot_function(global_sol);
		};


		auto show_error_func = [&]()
		{
			current_visualization = Visualizing::Error;
			MatrixXd global_sol;
			interpolate_function(state.sol, global_sol);

			MatrixXd exact_sol;
			state.problem.exact(vis_pts, exact_sol);

			const MatrixXd err = (global_sol - exact_sol).array().abs();
			plot_function(err);
		};


		auto show_basis_func = [&]()
		{
			if(vis_basis < 0 || vis_basis >= state.n_bases) return;

			current_visualization = Visualizing::VisBasis;

			MatrixXd fun = MatrixXd::Zero(state.n_bases, 1);
			fun(vis_basis) = 1;

			MatrixXd global_fun;
			interpolate_function(fun, global_fun);
			// global_fun /= 100;


			std::cout<<global_fun.minCoeff()<<" "<<global_fun.maxCoeff()<<std::endl;
			plot_function(global_fun);
		};

		auto linear_reproduction_func = [&]()
		{
			auto ff = [](double x, double y) {return -0.1 + .3*x - .5*y;};

			MatrixXd fun = MatrixXd::Zero(state.n_bases, 1);

			for(std::size_t i = 0; i < state.bases.size(); ++i)
			{
				const ElementBases &basis = state.bases[i];
				if(!basis.has_parameterization) continue;
				for(std::size_t j = 0; j < basis.bases.size(); ++j)
				{
					for(std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
					{
						const Local2Global &l2g = basis.bases[j].global()[kk];
						const int g_index = l2g.index;

						const MatrixXd node = l2g.node;
						// std::cout<<node<<std::endl;
						fun(g_index) = ff(node(0),node(1));
					}
				}
			}

			MatrixXd tmp;
			interpolate_function(fun, tmp);

			MatrixXd exact_sol(vis_pts.rows(), 1);
			for(long i = 0; i < vis_pts.rows(); ++i)
				exact_sol(i) =  ff(vis_pts(i, 0),vis_pts(i, 1));

			const MatrixXd global_fun = (exact_sol - tmp).array().abs();

			std::cout<<global_fun.minCoeff()<<" "<<global_fun.maxCoeff()<<std::endl;
			plot_function(global_fun);
		};


		auto build_vis_mesh_func = [&]()
		{
			vis_element_ranges.clear();


			vis_faces_poly.clear();
			vis_pts_poly.clear();

			igl::Timer timer; timer.start();
			std::cout<<"Building vis mesh..."<<std::flush;

			const double area_param = 0.00001*state.mesh->n_elements();

			std::stringstream buf;
			buf.precision(100);
			buf.setf(std::ios::fixed, std::ios::floatfield);

			if(state.mesh->is_volume())
			{
				buf<<"Qpq1.414a"<<area_param;
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

				MatrixXi tets;
				igl::copyleft::tetgen::tetrahedralize(pts, faces, buf.str(), local_vis_pts_quad, tets, local_vis_faces_quad);
			}
			else
			{
				buf<<"Qqa"<<area_param;
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
					igl::triangle::triangulate(pts, E, H, buf.str(), local_vis_pts_quad, local_vis_faces_quad);
				}
				// {
				// 	MatrixXd pts(3,2); pts <<
				// 	0,0,
				// 	1,0,
				// 	0,1;

				// 	MatrixXi E(3,2); E <<
				// 	0,1,
				// 	1,2,
				// 	2,0;

				// 	igl::triangle::triangulate(pts, E, MatrixXd(0,2), buf.str(), local_vis_pts_tri, local_vis_faces_tri);
				// }
			}

			const auto &current_bases = state.iso_parametric ? state.bases : state.geom_bases;
			int faces_total_size = 0, points_total_size = 0;
			vis_element_ranges.push_back(0);

			for(int i = 0; i < int(current_bases.size()); ++i)
			{
				const ElementBases &bs = current_bases[i];

				if(state.mesh->is_cube(i)){
					faces_total_size   += local_vis_faces_quad.rows();
					points_total_size += local_vis_pts_quad.rows();
				}
				// else if(is_tri(bs))
				// {
				// 	faces_total_size   += local_vis_faces_tri.rows();
				// 	points_total_size += local_vis_pts_tri.rows();
				// }
				else
				{
					if(state.mesh->is_volume())
					{
						vis_pts_poly[i] = state.polys_3d[i].first;
						vis_faces_poly[i] = state.polys_3d[i].second;

						faces_total_size   += vis_faces_poly[i].rows();
						points_total_size += vis_pts_poly[i].rows();
					}
					else
					{
						MatrixXd poly = state.polys[i];
						MatrixXi E(poly.rows(),2);
						for(int e = 0; e < int(poly.rows()); ++e)
						{
							E(e, 0) = e;
							E(e, 1) = (e+1) % poly.rows();
						}

						igl::triangle::triangulate(poly, E, MatrixXd(0,2), "Qpqa0.0001", vis_pts_poly[i], vis_faces_poly[i]);

						faces_total_size   += vis_faces_poly[i].rows();
						points_total_size += vis_pts_poly[i].rows();
					}
				}

				vis_element_ranges.push_back(faces_total_size);
			}

			vis_pts.resize(points_total_size, local_vis_pts_quad.cols());
			vis_faces.resize(faces_total_size, 3);

			MatrixXd mapped, tmp;
			int face_index = 0, point_index = 0;
			for(int i = 0; i < int(current_bases.size()); ++i)
			{
				const ElementBases &bs = current_bases[i];
				if(state.mesh->is_cube(i))
				{
					bs.eval_geom_mapping(local_vis_pts_quad, mapped);
					vis_faces.block(face_index, 0, local_vis_faces_quad.rows(), 3) = local_vis_faces_quad.array() + point_index;
					face_index += local_vis_faces_quad.rows();

					vis_pts.block(point_index, 0, mapped.rows(), mapped.cols()) = mapped;
					point_index += mapped.rows();
				}
				// else if(is_tri(bs))
				// {
				// 	bs.eval_geom_mapping(local_vis_pts_tri, mapped);
				// 	vis_faces.block(face_index, 0, local_vis_faces_tri.rows(), 3) = local_vis_faces_tri.array() + point_index;

				// 	face_index += local_vis_faces_tri.rows();

				// 	vis_pts.block(point_index, 0, mapped.rows(), mapped.cols()) = mapped;
				// 	point_index += mapped.rows();
				// }
				else{
					vis_faces.block(face_index, 0, vis_faces_poly[i].rows(), 3) = vis_faces_poly[i].array() + point_index;

					face_index += vis_faces_poly[i].rows();

					vis_pts.block(point_index, 0, vis_pts_poly[i].rows(), vis_pts_poly[i].cols()) = vis_pts_poly[i];
					point_index += vis_pts_poly[i].rows();
				}
			}

			assert(point_index == vis_pts.rows());
			assert(face_index == vis_faces.rows());

			if(state.mesh->is_volume())
			{
				//reverse all faces
				for(long i = 0; i < vis_faces.rows(); ++i)
				{
					const int v0 = vis_faces(i, 0);
					const int v1 = vis_faces(i, 1);
					const int v2 = vis_faces(i, 2);

					int tmpc = vis_faces(i, 2);
					vis_faces(i, 2) = vis_faces(i, 1);
					vis_faces(i, 1) = tmpc;
				}
			}
			else
			{
				Matrix2d mmat;
				for(long i = 0; i < vis_faces.rows(); ++i)
				{
					const int v0 = vis_faces(i, 0);
					const int v1 = vis_faces(i, 1);
					const int v2 = vis_faces(i, 2);

					mmat.row(0) = vis_pts.row(v2) - vis_pts.row(v0);
					mmat.row(1) = vis_pts.row(v1) - vis_pts.row(v0);

					if(mmat.determinant() > 0)
					{
						int tmpc = vis_faces(i, 2);
						vis_faces(i, 2) = vis_faces(i, 1);
						vis_faces(i, 1) = tmpc;
					}
				}
			}

			timer.stop();
			std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

			if(skip_visualization) return;

			clear_func();
			show_vis_mesh_func();
		};


		auto load_mesh_func = [&](){
			element_ranges.clear();
			vis_element_ranges.clear();

			vis_faces_poly.clear();
			vis_pts_poly.clear();

			state.load_mesh();
			state.compute_mesh_stats();
			state.mesh->triangulate_faces(tri_faces, tri_pts, element_ranges);
			state.mesh->compute_element_barycenters(normalized_barycenter);

			// std::cout<<"normalized_barycenter\n"<<normalized_barycenter<<"\n\n"<<std::endl;
			for(long i = 0; i < normalized_barycenter.cols(); ++i){
				normalized_barycenter.col(i) = MatrixXd(normalized_barycenter.col(i).array() - normalized_barycenter.col(i).minCoeff());
				normalized_barycenter.col(i) /= normalized_barycenter.col(i).maxCoeff();
			}

			// std::cout<<"normalized_barycenter\n"<<normalized_barycenter<<"\n\n"<<std::endl;

			if(skip_visualization) return;

			clear_func();
			show_mesh_func();
			viewer.core.align_camera_center(tri_pts);
		};

		auto build_basis_func = [&](){
			state.build_basis();

			if(skip_visualization) return;
			clear_func();
			show_mesh_func();
			show_nodes_func();
		};


		auto compute_assembly_vals_func = [&]() {
			state.compute_assembly_vals();

			if(skip_visualization) return;
			// clear_func();
			// show_mesh_func();
			// show_quadrature_func();
		};

		auto assemble_stiffness_mat_func = [&]() {
			state.assemble_stiffness_mat();
		};


		auto assemble_rhs_func = [&]() {
			state.assemble_rhs();

			// std::cout<<state.rhs<<std::endl;

			// if(skip_visualization) return;
			// clear_func();
			// show_rhs_func();
		};

		auto solve_problem_func = [&]() {
			state.solve_problem();
			// state.solve_problem_old();

			if(skip_visualization) return;
			clear_func();
			show_sol_func();
		};

		auto compute_errors_func = [&]() {
			state.compute_errors();

			if(skip_visualization) return;
			clear_func();
			show_error_func();
		};


		auto update_slices = [&]() {
			clear_func();
			switch(current_visualization)
			{
				case Visualizing::InputMesh: show_mesh_func(); break;
				case Visualizing::VisMesh: show_vis_mesh_func(); break;
				case Visualizing::Solution: show_sol_func(); break;
				case Visualizing::Rhs: break;
				case Visualizing::Error: show_error_func(); break;
				case Visualizing::VisBasis: show_basis_func(); break;
			}
		};

		enum Foo : int { A=0 };

		viewer.callback_init = [&](igl::viewer::Viewer& viewer_)
		{
			viewer_.ngui->addWindow(Eigen::Vector2i(220,10),"PolyFEM");

			viewer_.ngui->addGroup("Settings");

			viewer_.ngui->addVariable("quad order", state.quadrature_order);
			viewer_.ngui->addVariable("discr order", state.discr_order);
			viewer_.ngui->addVariable("b samples", state.n_boundary_samples);

			viewer_.ngui->addVariable("lambda", state.lambda);
			viewer_.ngui->addVariable("mu", state.mu);

			viewer_.ngui->addVariable("mesh path", state.mesh_path);
			viewer_.ngui->addButton("browse...", [&]() {
				std::string path = nanogui::file_dialog({
					{ "HYBRID", "General polyhedral mesh" }, { "OBJ", "Obj 2D mesh" }
				}, false);

				if (!path.empty())
					state.mesh_path = path;

			});
			viewer_.ngui->addVariable("n refs", state.n_refs);
			viewer_.ngui->addVariable("refinenemt t", state.refinenemt_location);

			viewer_.ngui->addVariable("spline basis", state.use_splines);


			viewer_.ngui->addVariable<igl::ColorMapType>("Colormap", color_map)->setItems({"inferno", "jet", "magma", "parula", "plasma", "viridis"});

			viewer_.ngui->addVariable<ProblemType>("Problem",
				[&](ProblemType val) { state.problem.set_problem_num(val); },
				[&]() { return ProblemType(state.problem.problem_num()); }
				)->setItems({"Linear","Quadratic","Franke", "Elastic", "Zero BC"});


			auto solvers = LinearSolver::availableSolvers();
			if (state.solver_type.empty()) {
				state.solver_type = LinearSolver::defaultSolver();
			}
			viewer_.ngui->addVariable<Foo>("Solver",
				[&,solvers](Foo i) { state.solver_type = solvers[i]; },
				[&,solvers]() { return (Foo) std::distance(solvers.begin(),
					std::find(solvers.begin(), solvers.end(), state.solver_type)); }
				)->setItems(solvers);

			auto precond = LinearSolver::availablePrecond();
			if (state.precond_type.empty()) {
				state.precond_type = LinearSolver::defaultPrecond();
			}
			viewer_.ngui->addVariable<Foo>("Precond",
				[&,precond](Foo i) { state.precond_type = precond[i]; },
				[&,precond]() { return (Foo) std::distance(precond.begin(),
					std::find(precond.begin(), precond.end(), state.precond_type)); }
				)->setItems(precond);

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
				compute_assembly_vals_func();

				if(!skip_visualization)
					build_vis_mesh_func();

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

			viewer_.ngui->addButton("Show linear r", linear_reproduction_func);

			viewer_.ngui->addVariable("basis num",vis_basis);
			viewer_.ngui->addButton("Show basis", show_basis_func);

			viewer_.ngui->addGroup("Slicing");
			viewer_.ngui->addVariable<int>("coord",[&](int val) {
				slice_coord = val;
				if(is_slicing)
					update_slices();
			},[&]() {
				return slice_coord;
			});
			viewer_.ngui->addVariable<float>("pos",[&](float val) {
				slice_position = val;
				if(is_slicing)
					update_slices();
			},[&]() {
				return slice_position;
			});

			viewer_.ngui->addButton("+0.1", [&](){ slice_position += 0.1; if(is_slicing) update_slices();});
			viewer_.ngui->addButton("-0.1", [&](){ slice_position -= 0.1; if(is_slicing) update_slices();});

			viewer_.ngui->addVariable<bool>("enable",[&](bool val) {
				is_slicing = val;
				update_slices();
			},[&]() {
				return is_slicing;
			});

			// viewer_.ngui->addGroup("Stats");
			// viewer_.ngui->addVariable("NNZ", Type &value)

			viewer_.ngui->addGroup("Selection");
			viewer_.ngui->addVariable("element ids", selected_elements);
			viewer_.ngui->addButton("Show", [&]{
				if(state.mesh->is_volume())
				{
					auto v{explode(selected_elements, ',')};
					current_3d_index = static_cast<Mesh3D *>(state.mesh)->get_index_from_element(atoi(v.front().c_str()), 1, 0);
					std::cout<<"e:"<<current_3d_index.element<<" f:"<<current_3d_index.face<<" e:"<<current_3d_index.edge<<" v:"<<current_3d_index.vertex<<std::endl;
				}

				plot_selection_and_index(true);
			});

			viewer_.ngui->addButton("Switch vertex", [&]{
				if(state.mesh->is_volume())
				{
					current_3d_index = static_cast<Mesh3D *>(state.mesh)->switch_vertex(current_3d_index);
					std::cout<<"e:"<<current_3d_index.element<<" f:"<<current_3d_index.face<<" e:"<<current_3d_index.edge<<" v:"<<current_3d_index.vertex<<std::endl;
				}

				plot_selection_and_index();
			});

			viewer_.ngui->addButton("Switch edge", [&]{
				if(state.mesh->is_volume())
				{
					current_3d_index = static_cast<Mesh3D *>(state.mesh)->switch_edge(current_3d_index);
					std::cout<<"e:"<<current_3d_index.element<<" f:"<<current_3d_index.face<<" e:"<<current_3d_index.edge<<" v:"<<current_3d_index.vertex<<std::endl;
				}

				plot_selection_and_index();
			});

			viewer_.ngui->addButton("Switch face", [&]{
				if(state.mesh->is_volume())
				{
					current_3d_index = static_cast<Mesh3D *>(state.mesh)->switch_face(current_3d_index);
					std::cout<<"e:"<<current_3d_index.element<<" f:"<<current_3d_index.face<<" e:"<<current_3d_index.edge<<" v:"<<current_3d_index.vertex<<std::endl;
				}

				plot_selection_and_index();
			});

			viewer_.ngui->addButton("Switch element", [&]{
				if(state.mesh->is_volume())
				{
					current_3d_index = static_cast<Mesh3D *>(state.mesh)->switch_element(current_3d_index);
					selected_elements += ","+std::to_string(current_3d_index.element);
					std::cout<<"e:"<<current_3d_index.element<<" f:"<<current_3d_index.face<<" e:"<<current_3d_index.edge<<" v:"<<current_3d_index.vertex<<std::endl;
				}

				plot_selection_and_index();
			});

			viewer_.ngui->addButton("Save selection", [&]{
				if(state.mesh->is_volume())
				{
					auto v{explode(selected_elements, ',')};
					std::set<int> idx;
					for(auto s : v)
						idx.insert(atoi(s.c_str()));

					std::vector<int> idx_v(idx.begin(), idx.end());

					static_cast<Mesh3D *>(state.mesh)->save(idx_v, 2, "mesh.HYBRID");
				}
			});

			viewer_.screen->performLayout();

			return false;
		};

		viewer.core.set_rotation_type(igl::viewer::ViewerCore::RotationType::ROTATION_TYPE_TRACKBALL);
		viewer.launch();
	}

	void UIState::sertialize(const std::string &name)
	{

	}

}
