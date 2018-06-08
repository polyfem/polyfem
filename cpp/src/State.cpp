#include "State.hpp"

#include "Mesh2D.hpp"
#include "Mesh3D.hpp"

#include "FEBasis2d.hpp"
#include "FEBasis3d.hpp"

#include "SpectralBasis2d.hpp"

#include "SplineBasis2d.hpp"
#include "SplineBasis3d.hpp"

#include "EdgeSampler.hpp"
#include "BoundarySampler.hpp"

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
#include "MeshUtils.hpp"

#include "NLProblem.hpp"
#include "SparseNewtonDescentSolver.hpp"

#include "auto_bases.hpp"

#ifdef USE_TBB
#include <tbb/task_scheduler_init.h>
#endif

#include <igl/Timer.h>
#include <igl/serialize.h>
#include <igl/remove_unreferenced.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/isolines.h>
#include <igl/write_triangle_mesh.h>

#include <igl/AABB.h>
#include <igl/in_element.h>
#include <igl/barycentric_coordinates.h>


#include <unsupported/Eigen/SparseExtra>

#include <iostream>
#include <algorithm>
#include <memory>

#include "autodiff.h"
DECLARE_DIFFSCALAR_BASE();


using namespace Eigen;


namespace poly_fem
{
	namespace
	{
		template<typename V1, typename  V2>
		double angle2(const V1 &v1, const V2 &v2)
		{
			assert(v1.size() == 2);
			assert(v2.size() == 2);
			return std::abs(atan2(v1(0)*v2(1) - v1(1)*v2(0), v1.dot(v2)));
		}

		template<typename V1, typename  V2>
		double angle3(const V1 &v1, const V2 &v2)
		{
			assert(v1.size() == 3);
			assert(v2.size() == 3);
			return std::abs(atan2(v1.cross(v2).norm(), v1.dot(v2)));
		}
	}

	State::State()
	{
		problem = ProblemFactory::factory().get_problem("Linear");
	}

	void State::compute_mesh_size(const Mesh &mesh_in, const std::vector< ElementBases > &bases_in, const int n_samples)
	{
		Eigen::MatrixXd samples_simplex, samples_cube, mapped, p0, p1, p;

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

		if(mesh_in.is_volume()){
			EdgeSampler::sample_3d_simplex(n_samples, samples_simplex);
			EdgeSampler::sample_3d_cube(n_samples, samples_cube);
		}
		else{
			EdgeSampler::sample_2d_simplex(n_samples, samples_simplex);
			EdgeSampler::sample_2d_cube(n_samples, samples_cube);
		}


		int n = 0;
		for(std::size_t i = 0; i < bases_in.size(); ++i){
			if(mesh_in.is_polytope(i)) continue;
			int n_edges;

			if(mesh_in.is_simplex(i)){
				n_edges = mesh_in.is_volume() ? 6 : 3;
				bases_in[i].eval_geom_mapping(samples_simplex, mapped);
			}
			else {
				n_edges = mesh_in.is_volume() ? 12 : 4;
				bases_in[i].eval_geom_mapping(samples_cube, mapped);
			}

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

	void State::save_json()
	{
		const std::string out_path = args["output"];
		if(!out_path.empty()){
			std::ofstream out(out_path);
			save_json(out);
			out.close();
		}
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
		j["discr_order_min"] = disc_orders.minCoeff();
		j["discr_order_max"] = disc_orders.maxCoeff();
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

		j["num_p1"] = (disc_orders.array() == 1).count();
		j["num_p2"] = (disc_orders.array() == 2).count();
		j["num_p3"] = (disc_orders.array() == 3).count();
		j["num_p4"] = (disc_orders.array() == 4).count();
		j["num_p5"] = (disc_orders.array() == 5).count();

		j["mesh_size"] = mesh_size;
		j["max_angle"] = max_angle;

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
		j["time_computing_poly_basis"] = computing_poly_basis_time;
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

#ifdef USE_TBB
		j["num_threads"] = tbb::task_scheduler_init::default_num_threads();

#else
		j["num_threads"] = 1;
#endif

		j["formulation"] = formulation();


		out << j.dump(4) << std::endl;

		std::cout<<"done"<<std::endl;
	}

	double get_opt_p(bool h1_formula, double B,
		double h_ref, int p_ref, double rho_ref,
		double h, double rho)
	{
		const double ptmp = h1_formula ?
				(std::log(B*std::pow(h_ref, p_ref + 1)*rho     / (h *rho_ref))            /std::log(h)):
				(std::log(B*std::pow(h_ref, p_ref + 2)*rho*rho / (h * h *rho_ref*rho_ref))/std::log(h));

		// std::cout<<ptmp<<std::endl;
		return std::min(std::max(p_ref, (int)std::round(ptmp)), autogen::MAX_P_BASES);
	}

	void State::p_refinement(const Mesh2D &mesh2d)
	{
		max_angle = 0;
		// static const int max_angles = 5;
		// static const double angles[max_angles] = {0, 170./180.*M_PI, 179./180.*M_PI, 179.9/180.* M_PI, M_PI};

		Eigen::MatrixXd p0, p1;
		mesh2d.get_edges(p0, p1);
		const auto tmp = p0-p1;
		const double h_ref = tmp.rowwise().norm().mean();
		const double B = args["B"];
		const bool h1_formula = args["h1_formula"];
		const int p_ref = args["discr_order"];
		const double rho_ref =  sqrt(3.0)/6.0*h_ref;

		// std::cout<<"h_ref "<<h_ref<<std::endl;
		// std::cout<<"edges "<<tmp.rowwise().norm()<<std::endl;

		double sigma_sum = 0;

		for(int f = 0; f < mesh2d.n_faces(); ++f)
		{
			if(!mesh2d.is_simplex(f))
				continue;

			auto v0 = mesh2d.point(mesh2d.face_vertex(f, 0));
			auto v1 = mesh2d.point(mesh2d.face_vertex(f, 1));
			auto v2 = mesh2d.point(mesh2d.face_vertex(f, 2));

			const RowVectorNd e0 = v1-v0;
			const RowVectorNd e1 = v2-v1;
			const RowVectorNd e2 = v0-v2;

			const double e0n = e0.norm();
			const double e1n = e1.norm();
			const double e2n = e2.norm();

			const double alpha0 = angle2(e0, -e2);
			const double alpha1 = angle2(e1, -e0);
			const double alpha2 = angle2(e2, -e1);

			const double P = e0n + e1n + e2n;
			const double A = std::abs(e1(0)*e2(1) - e1(1)*e2(0))/2;
			const double rho = 2*A/P;
			const double hp = std::max(e0n, std::max(e1n, e2n));

			sigma_sum += rho/hp;

			// std::cout<<"A "<<A<< " rho "<<rho<<" hp "<<hp<<std::endl;

			// const double ptmp = 0.5 * (-4 * std::log(hp) + (2*p_ref+2) * std::log(h) + 2*std::log(rho) + 2*std::log(2) + std::log(3) + 2*std::log(B))/std::log(hp);

			// const double ptmp = std::log(B*std::pow(h_ref, p_ref + 1)*rho / (hp *rho_ref))/std::log(hp);
			// const double ptmp = std::log(B*std::pow(h_ref, p_ref + 2)*rho*rho / (hp *hp *rho_ref*rho_ref))/std::log(hp);

			// const int p = std::min(std::max(p_ref, (int)std::round(ptmp)), autogen::MAX_P_BASES);
			const int p = get_opt_p(h1_formula, B, h_ref, p_ref, rho_ref, hp, rho);

			if(p > disc_orders[f])
				disc_orders[f] = p;
			auto index = mesh2d.get_index_from_face(f);

			for(int lv = 0; lv < 3; ++lv)
			{
				auto nav = mesh2d.switch_face(index);

				if(nav.face >=0)
				{
					if(p > disc_orders[nav.face])
						disc_orders[nav.face] = p;
				}

				index = mesh2d.next_around_face(index);
			}

			max_angle = std::max(max_angle, alpha0);
			max_angle = std::max(max_angle, alpha1);
			max_angle = std::max(max_angle, alpha2);
		}

		max_angle = max_angle/M_PI*180.;
		std::cout<<"using B=" << B << " with " << (h1_formula ? "H1" : "L2") <<" estimate max_angle "<<max_angle<<std::endl;
		std::cout<<"average sigma: "<<sigma_sum/mesh2d.n_faces()<<std::endl;

		std::cout<<"num_p1 " << (disc_orders.array() == 1).count()<<std::endl;
		std::cout<<"num_p2 " << (disc_orders.array() == 2).count()<<std::endl;
		std::cout<<"num_p3 " << (disc_orders.array() == 3).count()<<std::endl;
		std::cout<<"num_p4 " << (disc_orders.array() == 4).count()<<std::endl;
		std::cout<<"num_p5 " << (disc_orders.array() == 5).count()<<std::endl;
	}

	void State::p_refinement(const Mesh3D &mesh3d)
	{
		max_angle = 0;

		Eigen::MatrixXd p0, p1;
		mesh3d.get_edges(p0, p1);
		const auto tmp = p0-p1;
		const double h_ref = tmp.rowwise().norm().mean();
		const double B = args["B"];
		const bool h1_formula = args["h1_formula"];
		const int p_ref = args["discr_order"];
		const double rho_ref = sqrt(6.)/12.*h_ref;

		for(int c = 0; c < mesh3d.n_cells(); ++c)
		{
			if(!mesh3d.is_simplex(c))
				continue;


			const auto v0 = mesh3d.point(mesh3d.cell_vertex(c, 0));
			const auto v1 = mesh3d.point(mesh3d.cell_vertex(c, 1));
			const auto v2 = mesh3d.point(mesh3d.cell_vertex(c, 2));
			const auto v3 = mesh3d.point(mesh3d.cell_vertex(c, 3));

			Eigen::Matrix<double, 6, 3> e;
			e.row(0) = v0 - v1;
			e.row(1) = v1 - v2;
			e.row(2) = v2 - v0;

			e.row(3) = v0 - v3;
			e.row(4) = v1 - v3;
			e.row(5) = v2 - v3;

			Eigen::Matrix<double, 6, 1> en = e.rowwise().norm();

			Eigen::Matrix<double, 3*4, 1> alpha;
			alpha(0) = angle3(e.row(0), -e.row(1));	 	alpha(1) = angle3(e.row(1), -e.row(2));	 	alpha(2) = angle3(e.row(2), -e.row(0));
			alpha(3) = angle3(e.row(0), -e.row(4));	 	alpha(4) = angle3(e.row(4), e.row(3));	 	alpha(5) = angle3(-e.row(3), -e.row(0));
			alpha(6) = angle3(-e.row(4), -e.row(1));	alpha(7) = angle3(e.row(1), -e.row(5));	 	alpha(8) = angle3(e.row(5), e.row(4));
			alpha(9) = angle3(-e.row(2), -e.row(5));	alpha(10) = angle3(e.row(5), e.row(3));		alpha(11) = angle3(-e.row(3), e.row(2));

			const double S = (e.row(0).cross(e.row(1)).norm() + e.row(0).cross(e.row(4)).norm() + e.row(4).cross(e.row(1)).norm() + e.row(2).cross(e.row(5)).norm()) / 2;
			const double V = std::abs(e.row(3).dot(e.row(2).cross(-e.row(0))))/6;
			const double rho = 3 * V / S;
			const double hp = en.maxCoeff();

			// const double ptmp = std::log(B*std::pow(h_ref, p_ref + 1)*rho / (hp *rho_ref))/std::log(hp);
			// const double ptmp = std::log(B*std::pow(h_ref, p_ref + 2)*rho*rho / (hp *hp *rho_ref*rho_ref))/std::log(hp);
			// 0.5 * (-4*power*std::log(hp) + (2*p_ref+2) * std::log(h) + 2*power*std::log(rho) + 3*std::log(2) + std::log(3) + 2*std::log(B)) / std::log(hp);
			// const int p = std::min(std::max(p_ref, (int)std::round(ptmp)), autogen::MAX_P_BASES);
			const int p = get_opt_p(h1_formula, B, h_ref, p_ref, rho_ref, hp, rho);


			if(p > disc_orders[c])
				disc_orders[c] = p;

			for(int le = 0; le < 6; ++le)
			{
				const int e_id = mesh3d.cell_edge(c, le);
				const auto cells = mesh3d.edge_neighs(e_id);

				for(auto c_id : cells)
				{
					if(p > disc_orders[c_id])
						disc_orders[c_id] = p;
				}
			}


			max_angle = std::max(max_angle, alpha.maxCoeff());
		}

		max_angle = max_angle/M_PI*180.;
		std::cout<<"using B=" << B << " with " << (h1_formula ? "H1" : "L2") <<" estimate max_angle "<<max_angle<<std::endl;

		std::cout<<"num_p1 " << (disc_orders.array() == 1).count()<<std::endl;
		std::cout<<"num_p2 " << (disc_orders.array() == 2).count()<<std::endl;
		std::cout<<"num_p3 " << (disc_orders.array() == 3).count()<<std::endl;
		std::cout<<"num_p4 " << (disc_orders.array() == 4).count()<<std::endl;
		std::cout<<"num_p5 " << (disc_orders.array() == 5).count()<<std::endl;

	}

	void State::interpolate_boundary_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, MatrixXd &result)
	{
		assert(mesh->is_volume());

		const Mesh3D &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());

		Eigen::MatrixXd points;
		Eigen::VectorXd weights;

		int actual_dim = 1;
		if(!problem->is_scalar())
			actual_dim = 3;

		igl::AABB<Eigen::MatrixXd, 3> tree;
		tree.init(pts, faces);

		const auto &gbases = iso_parametric() ? bases : geom_bases;
		result.resize(faces.rows(), actual_dim);
		result.setConstant(std::numeric_limits<double>::quiet_NaN());

		int counter = 0;

		for(int e = 0; e < mesh3d.n_elements(); ++e)
		{
			const ElementBases &gbs = gbases[e];
			const ElementBases &bs = bases[e];

			for(int lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
			{
				const int face_id = mesh3d.cell_face(e, lf);
				if(!mesh3d.is_boundary_face(face_id))
					continue;

				BoundarySampler::quadrature_for_tri_face(lf, 4, points, weights);
				weights *= mesh3d.tri_area(face_id);

				ElementAssemblyValues vals;
				vals.compute(e, true, points, bs, gbs);
				Eigen::Vector3d loc_val; loc_val.setZero();

				// UIState::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(1,0,0));

				const auto nodes = bs.local_nodes_for_primitive(face_id, mesh3d);

				for(long n = 0; n < nodes.size(); ++n)
				{
					// const auto &b = bs.bases[nodes(n)];
					const AssemblyValues &v = vals.basis_values[nodes(n)];
					for(int d = 0; d < actual_dim; ++d)
					{
						for(size_t g = 0; g < v.global.size(); ++g)
						{
							loc_val(d) +=  (v.global[g].val * v.val.array() * fun(v.global[g].index*actual_dim + d) * weights.array()).sum();
						}
					}
				}

				int I;
				Eigen::RowVector3d C;
				const Eigen::RowVector3d bary = mesh3d.face_barycenter(face_id);

				const double dist = tree.squared_distance(pts, faces, bary, I, C);
				assert(dist < 1e-16);
				// std::cout<<face_id<<" - "<<I<<": "<<dist<<" -> "<<bary<<std::endl;
				assert(std::isnan(result(I, 0)));
				result.row(I) = loc_val;
				++counter;
			}
		}

		assert(counter == result.rows());
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

	void State::load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd&)> &boundary_marker)
	{
		bases.clear();
		geom_bases.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();
		parent_elements.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		n_bases = 0;



		igl::Timer timer; timer.start();
		std::cout<<"Loading mesh..."<<std::flush;
		mesh = Mesh::create(meshin);
		if (!mesh) {
			return;
		}

		if(args["normalize_mesh"])
			mesh->normalize();

		mesh->refine(args["n_refs"], args["refinenemt_location"], parent_elements);

		mesh->compute_boundary_ids(boundary_marker);

		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

		RefElementSampler::sampler().init(mesh->is_volume(), mesh->n_elements(), args["vismesh_rel_area"]);

		disc_orders.resize(mesh->n_elements());
		disc_orders.setConstant(args["discr_order"]);
	}

	void State::load_mesh()
	{
		bases.clear();
		geom_bases.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
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

		// mesh->set_tag(1712, ElementType::InteriorPolytope);

		const std::string bc_tag_path = args["bc_tag"];
		if(bc_tag_path.empty())
			mesh->compute_boundary_ids();
		else
			mesh->load_boundary_ids(bc_tag_path);


		timer.stop();
		std::cout<<" took "<<timer.getElapsedTime()<<"s"<<std::endl;

		RefElementSampler::sampler().init(mesh->is_volume(), mesh->n_elements(), args["vismesh_rel_area"]);

		disc_orders.resize(mesh->n_elements());
		disc_orders.setConstant(args["discr_order"]);



		// const double poly_percentage = 0.05;
		const double poly_percentage = 0;
		const double perturb_t = 0.3;

		if(poly_percentage > 0)
		{
			const int n_poly = std::max(1., mesh->n_elements()*poly_percentage);
			int counter = 0;
			srand(11);

			for(int trial = 0; trial < n_poly*10; ++trial)
			{
				int el_id = rand() % mesh->n_elements();

				auto tags = mesh->elements_tag();

				if(mesh->is_volume())
				{
					assert(false);
				}
				else
				{
					const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
					auto index = tmp_mesh.get_index_from_face(el_id);

					bool stop = false;

					for(int i = 0; i < tmp_mesh.n_face_vertices(el_id); ++i)
					{
						if(tmp_mesh.is_boundary_edge(index.edge))
						{
							stop = true;
							break;
						}

						const auto neigh_index = tmp_mesh.switch_face(index);
						if(tags[neigh_index.face] != ElementType::RegularInteriorCube)
						{
							stop = true;
							break;
						}

						const auto f1 = tmp_mesh.switch_face(tmp_mesh.switch_edge(neigh_index						 )).face;
						const auto f2 = tmp_mesh.switch_face(tmp_mesh.switch_edge(tmp_mesh.switch_vertex(neigh_index))).face;
						if((f1 >= 0 && tags[f1] != ElementType::RegularInteriorCube) || (f2 >= 0 && tags[f2] != ElementType::RegularInteriorCube ))
						{
							stop = true;
							break;
						}

						index = tmp_mesh.next_around_face(index);
					}

					if(stop) continue;
				}

				mesh->set_tag(el_id, ElementType::InteriorPolytope);
				++counter;

				mesh->update_elements_tag();

				if(counter >= n_poly)
					break;

			}


			if(perturb_t > 0)
			{
				if(mesh->is_volume())
				{
					assert(false);
				}
				else
				{
					Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
					for(int el_id = 0; el_id < tmp_mesh.n_elements(); ++el_id)
					{
						if(!tmp_mesh.is_polytope(el_id))
							continue;

						const int rand_index = rand() % tmp_mesh.n_face_vertices(el_id);
						auto index = tmp_mesh.get_index_from_face(el_id);
						for(int r = 0; r < rand_index; ++r)
							index = tmp_mesh.next_around_face(index);

						const auto v1 = tmp_mesh.point(index.vertex);
						const auto v2 = tmp_mesh.point(tmp_mesh.next_around_face(tmp_mesh.next_around_face(index)).vertex);


						const double t = perturb_t + ((double) rand() / (RAND_MAX)) * 0.2 - 0.1;
						const RowVectorNd v = t * v1 + (1-t) * v2;
						tmp_mesh.set_point(index.vertex, v);
					}
				}
			}
		}

		// for(int i = 8; i < 16; ++i)
			// mesh->set_tag(i, ElementType::SimpleSingularInteriorCube);

		//TODO
		// disc_orders(18) = 4;
		// disc_orders(2) = 4;
		// disc_orders(4) = 4;
		// disc_orders(20) = 4;
		// disc_orders(14) = 4;
		// disc_orders(28) = 4;
		// disc_orders(12) = 4;
		// disc_orders(27) = 4;
		// disc_orders(11) = 4;
		// disc_orders(23) = 4;
		// disc_orders(25) = 4;
		// disc_orders(5) = 4;
		// disc_orders(21) = 4;
	}

	void State::compute_mesh_stats()
	{
		bases.clear();
		geom_bases.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
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
		local_neumann_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();
		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);

		n_bases = 0;


		igl::Timer timer; timer.start();
		std::cout<<"Building "<< (iso_parametric()? "isoparametric":"not isoparametric") <<" basis..."<<std::flush;

		local_boundary.clear();
		local_neumann_boundary.clear();
		std::map<int, InterfaceData> poly_edge_to_data_geom; //temp dummy variable

		const int base_p = args["discr_order"];
		disc_orders.setConstant(base_p);

		if(args["use_p_ref"])
		{
			if(mesh->is_volume())
				p_refinement(*dynamic_cast<Mesh3D *>(mesh.get()));
			else
				p_refinement(*dynamic_cast<Mesh2D *>(mesh.get()));

			std::cout<<"min p: " << disc_orders.minCoeff() << " max p: " << disc_orders.maxCoeff()<<std::endl;
		}

		if(mesh->is_volume())
		{
			const Mesh3D &tmp_mesh = *dynamic_cast<Mesh3D *>(mesh.get());
			if(args["use_spline"])
			{
				if(!iso_parametric())
					FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, geom_bases, local_boundary, poly_edge_to_data_geom);

				n_bases = SplineBasis3d::build_bases(tmp_mesh, args["quadrature_order"], bases, local_boundary, poly_edge_to_data);

				if(iso_parametric() &&  args["fit_nodes"])
					SplineBasis3d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if (!iso_parametric())
					FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], 1, geom_bases, local_boundary, poly_edge_to_data_geom);

				n_bases = FEBasis3d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, bases, local_boundary, poly_edge_to_data);
			}
		}
		else
		{
			const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
			if(args["use_spline"])
			{

				if(!iso_parametric())
					FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, geom_bases, local_boundary, poly_edge_to_data_geom);

				n_bases = SplineBasis2d::build_bases(tmp_mesh, args["quadrature_order"], bases, local_boundary, poly_edge_to_data);

				if(iso_parametric() && args["fit_nodes"])
					SplineBasis2d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if(!iso_parametric())
					FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], 1, geom_bases, local_boundary, poly_edge_to_data_geom);

				n_bases = FEBasis2d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, bases, local_boundary, poly_edge_to_data);
				// n_bases = SpectralBasis2d::build_bases(tmp_mesh, args["quadrature_order"], disc_orders, bases, geom_bases, local_boundary);
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
				// std::cout<<"Basis "<< i << ( parent_elements.size() > 0 ? (" -> " + std::to_string(parent_elements[i])) : "") << " has negative volume P" <<disc_orders(i)<<std::endl;

				// std::cout<<mesh->point(dynamic_cast<Mesh2D *>(mesh.get())->face_vertex(i, 0))<<std::endl;
				// std::cout<<mesh->point(dynamic_cast<Mesh2D *>(mesh.get())->face_vertex(i, 1))<<std::endl;
				// std::cout<<mesh->point(dynamic_cast<Mesh2D *>(mesh.get())->face_vertex(i, 2))<<std::endl;
				++n_flipped;
			}
		}

		// dynamic_cast<Mesh3D *>(mesh.get())->save({56}, 1, "mesh.HYBRID");

		// std::sort(flipped_elements.begin(), flipped_elements.end());
		// auto it = std::unique(flipped_elements.begin(), flipped_elements.end());
		// flipped_elements.resize(std::distance(flipped_elements.begin(), it));


		problem->setup_bc(*mesh, bases, local_boundary, boundary_nodes, local_neumann_boundary);

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
		computing_poly_basis_time = timer.getElapsedTime();
		std::cout<<" took "<<computing_poly_basis_time<<"s"<<std::endl;
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
		rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs);

		timer.stop();
		assigning_rhs_time = timer.getElapsedTime();
		std::cout<<" took "<<assigning_rhs_time<<"s"<<std::endl;
	}

	void State::solve_problem()
	{
		sol.resize(0, 0);

		igl::Timer timer; timer.start();
		std::cout<<"Solving " << formulation() <<"... "<<std::flush;


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

			std::cout<<"Solver error: "<<(A*sol-b).norm()<<std::endl;
		}
		else
		{
			int steps = 20;
			RhsAssembler rhs_assembler(*mesh, n_bases, mesh->dimension(), bases, iso_parametric() ? bases : geom_bases, formulation(), *problem);
			VectorXd tmp_sol;
			auto rhs_old  = rhs;
			for(int n = 1; n <=steps; ++n)
			{
				const double t = double(n)/double(steps);

				rhs = rhs_old * t;
				NLProblem nl_problem(rhs_assembler, t);
				if(n == 1)
					tmp_sol = nl_problem.initial_guess();

				// {
				// 	// tmp_sol.setZero();
				// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad, expected_grad;
				// 	nl_problem.gradient(tmp_sol, actual_grad);

				// 	Eigen::SparseMatrix<double> hessian;
				// 	Eigen::MatrixXd expected_hessian;
				// 	nl_problem.hessian(tmp_sol, hessian);
				// 	nl_problem.finiteGradient(tmp_sol, expected_grad, 0);

				// 	Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);

				// 	for(int i = 0; i < actual_hessian.rows(); ++i)
				// 	{
				// 		double hhh = 1e-7;
				// 		VectorXd xp = tmp_sol; xp(i) += hhh;
				// 		VectorXd xm = tmp_sol; xm(i) -= hhh;

				// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
				// 		nl_problem.gradient(xp, tmp_grad_p);

				// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
				// 		nl_problem.gradient(xm, tmp_grad_m);

				// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m)/hhh/2.;

				// 		const double vp = nl_problem.value(xp);
				// 		const double vm = nl_problem.value(xm);

				// 		const double fd = (vp-vm)/hhh/2.;
				// 		const double  diff = std::abs(actual_grad(i) - fd);
				// 		if(diff > 1e-6)
				// 			std::cout<<"diff grad "<<i<<": "<<actual_grad(i)<<" vs "<<fd <<" error: " <<diff<<std::endl;

				// 		for(int j = 0; j < actual_hessian.rows(); ++j)
				// 		{
				// 			const double diff = std::abs(actual_hessian(i,j) - fd_h(j));

				// 			if(diff > 1e-6)
				// 				std::cout<<"diff H "<<i<<", "<<j<<": "<<actual_hessian(i,j)<<" vs "<<fd_h(j)<<" error: " <<diff<<std::endl;

				// 		}
				// 	}
				// 	exit(0);

				// 	// std::cout<<"diff grad "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
				// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
				// }

				cppoptlib::SparseNewtonDescentSolver<NLProblem> solver(n == 1);
				solver.minimize(nl_problem, tmp_sol);
				solver.getInfo(solver_info);
				std::cout<<n<<"/"<<steps<<std::endl;
			}

			rhs = rhs_old;

			const int full_size 	= n_bases*mesh->dimension();
			const int reduced_size 	= n_bases*mesh->dimension() - boundary_nodes.size();

			NLProblem::reduced_to_full_aux(full_size, reduced_size, tmp_sol, false, sol);
		}

		timer.stop();
		solving_time = timer.getElapsedTime();
		std::cout<<" took "<<solving_time<<"s"<<std::endl;
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
		// double pred_norm = 0;

		static const int p = 8;

		// Eigen::MatrixXd err_per_el(n_el, 5);

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

			auto err = (v_exact-v_approx).eval().rowwise().norm().eval();
			const auto err_grad = (v_exact_grad - v_approx_grad).eval().rowwise().norm().eval();

			// for(long i = 0; i < err.size(); ++i)
				// errors.push_back(err(i));

			linf_err = max(linf_err, err.maxCoeff());
			grad_max_err = max(linf_err, err_grad.maxCoeff());

			// {
			// 	const auto &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());
			// 	const auto v0 = mesh3d.point(mesh3d.cell_vertex(e, 0));
			// 	const auto v1 = mesh3d.point(mesh3d.cell_vertex(e, 1));
			// 	const auto v2 = mesh3d.point(mesh3d.cell_vertex(e, 2));
			// 	const auto v3 = mesh3d.point(mesh3d.cell_vertex(e, 3));

			// 	Eigen::Matrix<double, 6, 3> ee;
			// 	ee.row(0) = v0 - v1;
			// 	ee.row(1) = v1 - v2;
			// 	ee.row(2) = v2 - v0;

			// 	ee.row(3) = v0 - v3;
			// 	ee.row(4) = v1 - v3;
			// 	ee.row(5) = v2 - v3;

			// 	Eigen::Matrix<double, 6, 1> en = ee.rowwise().norm();

			// 	// Eigen::Matrix<double, 3*4, 1> alpha;
			// 	// alpha(0) = angle3(e.row(0), -e.row(1));	 	alpha(1) = angle3(e.row(1), -e.row(2));	 	alpha(2) = angle3(e.row(2), -e.row(0));
			// 	// alpha(3) = angle3(e.row(0), -e.row(4));	 	alpha(4) = angle3(e.row(4), e.row(3));	 	alpha(5) = angle3(-e.row(3), -e.row(0));
			// 	// alpha(6) = angle3(-e.row(4), -e.row(1));	alpha(7) = angle3(e.row(1), -e.row(5));	 	alpha(8) = angle3(e.row(5), e.row(4));
			// 	// alpha(9) = angle3(-e.row(2), -e.row(5));	alpha(10) = angle3(e.row(5), e.row(3));		alpha(11) = angle3(-e.row(3), e.row(2));

			// 	const double S = (ee.row(0).cross(ee.row(1)).norm() + ee.row(0).cross(ee.row(4)).norm() + ee.row(4).cross(ee.row(1)).norm() + ee.row(2).cross(ee.row(5)).norm()) / 2;
			// 	const double V = std::abs(ee.row(3).dot(ee.row(2).cross(-ee.row(0))))/6;
			// 	const double rho = 3 * V / S;
			// 	const double hp = en.maxCoeff();
			// 	const int pp = disc_orders(e);
			// 	const int p_ref = args["discr_order"];

			// 	err_per_el(e, 0) = err.mean();
			// 	err_per_el(e, 1) = err.maxCoeff();
			// 	err_per_el(e, 2) = std::pow(hp, pp+1)/(rho/hp); // /std::pow(average_edge_length, p_ref+1) * (sqrt(6)/12);
			// 	err_per_el(e, 3) = rho/hp;
			// 	err_per_el(e, 4) = (vals.det.array() * vals.quadrature.weights.array()).sum();

			// 	// pred_norm += (pow(std::pow(hp, pp+1)/(rho/hp),p) * vals.det.array() * vals.quadrature.weights.array()).sum();
			// }

			l2_err += (err.array() * err.array() 			* vals.det.array() * vals.quadrature.weights.array()).sum();
			h1_err += (err_grad.array() * err_grad.array() 	* vals.det.array() * vals.quadrature.weights.array()).sum();
			lp_err += (err.array().pow(p) 					* vals.det.array() * vals.quadrature.weights.array()).sum();
		}

		h1_semi_err = sqrt(fabs(h1_err));
		h1_err = sqrt(fabs(l2_err) + fabs(h1_err));
		l2_err = sqrt(fabs(l2_err));

		lp_err = pow(fabs(lp_err), 1./p);

		// pred_norm = pow(fabs(pred_norm), 1./p);

		timer.stop();
		computing_errors_time = timer.getElapsedTime();
		std::cout<<" took "<<computing_errors_time<<"s"<<std::endl;

		std::cout << "-- L2 error: " << l2_err << std::endl;
		std::cout << "-- Lp error: " << lp_err << std::endl;
		std::cout << "-- H1 error: " << h1_err << std::endl;
		std::cout << "-- H1 semi error: " << h1_semi_err << std::endl;
		// std::cout << "-- Perd norm: " << pred_norm << std::endl;

		std::cout << "\n --Linf error: " << linf_err << std::endl;
		std::cout << "-- grad max error: " << grad_max_err << std::endl;
		// std::cout<<l2_err<<" "<<linf_err<<" "<<lp_err<<std::endl;

		// {
		// 	std::ofstream out("errs.txt");
		// 	out<<err_per_el;
		// 	out.close();
		// }
	}

	State &State::state(){
		static State instance;

		return instance;
	}

	void State::init(const json &args_in)
	{
		this->args = {
			{"mesh", ""},
			{"bc_tag", ""},
			{"n_refs", 0},
			{"vismesh_rel_area", 0.00001},
			{"refinenemt_location", 0.5},
			{"n_boundary_samples", 10},
			{"problem", "Franke"},
			{"normalize_mesh", true},

			{"scalar_formulation", "Laplacian"},
			{"tensor_formulation", "LinearElasticity"},

			{"B", 3},
			{"h1_formula", false},

			{"quadrature_order", 12},
			{"discr_order", 1},
			{"boundary_samples", 10},
			{"use_p_ref", false},
			{"use_spline", false},
			{"iso_parametric", false},
			{"integral_constraints", 2},

			{"fit_nodes", false},

			{"n_harmonic_samples", 10},

			{"solver_type", LinearSolver::defaultSolver()},
			{"precond_type", LinearSolver::defaultPrecond()},

			{"solver_params", json({})},

			{"params", {
				{"lambda", 0.32967032967032966},
				{"mu", 0.3846153846153846},
				{"k", 1.0},
				{"elasticity_tensor", json({})},
				// {"young", 1.0},
				// {"nu", 0.3},
				{"alphas", {2.13185026692482, -0.600299816209491}},
				{"mus", {0.00407251192475097, 0.000167202574129608}},
				{"Ds", {9.4979, 1000000}}
			}},

			{"problem_params", json({})},

			{"output", ""},

			{"export", {
				{"vis_mesh", ""},
				{"wire_mesh", ""},
				{"iso_mesh", ""},
			}}
		};

		this->args.merge_patch(args_in);


		problem = ProblemFactory::factory().get_problem(args["problem"]);
		problem->set_parameters(args["problem_params"]);
	}

	void State::export_data()
	{
		// Export vtu mesh of solution + wire mesh of deformed input
		// + mesh colored with the bases
		const std::string vis_mesh_path  = args["export"]["vis_mesh"];
		const std::string wire_mesh_path = args["export"]["wire_mesh"];
		const std::string iso_mesh_path = args["export"]["iso_mesh"];

		if (!vis_mesh_path.empty()) {
			save_vtu(vis_mesh_path);
		}
		if (!wire_mesh_path.empty()) {
			save_wire(wire_mesh_path);
		}
		if (!iso_mesh_path.empty()) {
			save_wire(iso_mesh_path, true);
		}
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

		Eigen::MatrixXd discr(pts_total_size, 1);

		Eigen::MatrixXd mapped, tmp;
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
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
				pts_index += mapped.rows();
			}
			else if(mesh->is_cube(i))
			{
				bs.eval_geom_mapping(sampler.cube_points(), mapped);
				tets.block(tet_index, 0, sampler.cube_volume().rows(), tets.cols()) = sampler.cube_volume().array() + pts_index;
				tet_index += sampler.cube_volume().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
				pts_index += mapped.rows();
			}
		}

		assert(pts_index == points.rows());
		assert(tet_index == tets.rows());

		Eigen::MatrixXd fun, exact_fun, err;

		interpolate_function(pts_index, sol, fun);

		if(problem->has_exact_sol()){
			problem->exact(points, exact_fun);
			err = (fun - exact_fun).eval().rowwise().norm();
		}

		VTUWriter writer;

		if(fun.cols() != 1 && !mesh->is_volume())
		{
			fun.conservativeResize(fun.rows(), 3);
			fun.col(2).setZero();

			exact_fun.conservativeResize(exact_fun.rows(), 3);
			exact_fun.col(2).setZero();
		}

		writer.add_field("solution", fun);
		writer.add_field("discr", discr);
		if(problem->has_exact_sol()){
			writer.add_field("exact", exact_fun);
			writer.add_field("error", err);
		}


		if(fun.cols() != 1)
		{
			Eigen::MatrixXd scalar_val;
			compute_scalar_value(pts_index, sol, scalar_val);
			writer.add_field("scalar_value", scalar_val);
		}

		writer.write_tet_mesh(path, points, tets);
	}

	void State::save_wire(const std::string &name, bool isolines) {
		const auto &sampler = RefElementSampler::sampler();

		const auto &current_bases = iso_parametric() ? bases : geom_bases;
		int seg_total_size = 0;
		int pts_total_size = 0;
		int faces_total_size = 0;

		for(size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if(mesh->is_simplex(i)) {
				pts_total_size += sampler.simplex_points().rows();
				seg_total_size += sampler.simplex_edges().rows();
				faces_total_size += sampler.simplex_faces().rows();
			} else if(mesh->is_cube(i)) {
				pts_total_size += sampler.cube_points().rows();
			}
		}

		Eigen::MatrixXd points(pts_total_size, mesh->dimension());
		Eigen::MatrixXi edges(seg_total_size, 2);
		Eigen::MatrixXi faces(faces_total_size, 3);
		points.setZero();

		MatrixXd mapped, tmp;
		int seg_index = 0, pts_index = 0, face_index = 0;
		for(size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if(mesh->is_simplex(i))
			{
				bs.eval_geom_mapping(sampler.simplex_points(), mapped);
				edges.block(seg_index, 0, sampler.simplex_edges().rows(), edges.cols()) = sampler.simplex_edges().array() + pts_index;
				seg_index += sampler.simplex_edges().rows();

				faces.block(face_index, 0, sampler.simplex_faces().rows(), 3) = sampler.simplex_faces().array() + pts_index;
				face_index += sampler.simplex_faces().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				pts_index += mapped.rows();
			}
			else if(mesh->is_cube(i))
			{
				bs.eval_geom_mapping(sampler.cube_points(), mapped);
				edges.block(seg_index, 0, sampler.simplex_edges().rows(), edges.cols()) = sampler.simplex_edges().array() + pts_index;
				seg_index += sampler.simplex_edges().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				pts_index += mapped.rows();
			}
		}

		assert(pts_index == points.rows());
		assert(face_index == faces.rows());

		if(mesh->is_volume())
		{
			//reverse all faces
			for(long i = 0; i < faces.rows(); ++i)
			{
				const int v0 = faces(i, 0);
				const int v1 = faces(i, 1);
				const int v2 = faces(i, 2);

				int tmpc = faces(i, 2);
				faces(i, 2) = faces(i, 1);
				faces(i, 1) = tmpc;
			}
		}
		else
		{
			Matrix2d mmat;
			for(long i = 0; i < faces.rows(); ++i)
			{
				const int v0 = faces(i, 0);
				const int v1 = faces(i, 1);
				const int v2 = faces(i, 2);

				mmat.row(0) = points.row(v2) - points.row(v0);
				mmat.row(1) = points.row(v1) - points.row(v0);

				if(mmat.determinant() > 0)
				{
					int tmpc = faces(i, 2);
					faces(i, 2) = faces(i, 1);
					faces(i, 1) = tmpc;
				}
			}
		}

		Eigen::MatrixXd fun, exact_fun, err;

		interpolate_function(pts_index, sol, fun);

		if (problem->has_exact_sol()) {
			problem->exact(points, exact_fun);
			err = (fun - exact_fun).eval().rowwise().norm();
		}

		if (fun.cols() != 1 && !mesh->is_volume()) {
			fun.conservativeResize(fun.rows(), 3);
			fun.col(2).setZero();

			exact_fun.conservativeResize(exact_fun.rows(), 3);
			exact_fun.col(2).setZero();
		}

		if(!mesh->is_volume())
		{
			points.conservativeResize(points.rows(), 3);
			points.col(2).setZero();
		}

		// writer.add_field("solution", fun);
		// if (problem->has_exact_sol()) {
		// 	writer.add_field("exact", exact_fun);
		// 	writer.add_field("error", err);
		// }

		// if (fun.cols() != 1) {
		// 	Eigen::MatrixXd scalar_val;
		// 	compute_scalar_value(pts_index, sol, scalar_val);
		// 	writer.add_field("scalar_value", scalar_val);
		// }

		if (fun.cols() != 1) {
			assert(points.rows() == fun.rows());
			assert(points.cols() == fun.cols());
			points += fun;
		} else {
			if (isolines)
				points.col(2) += fun;
		}

		if (isolines) {
			Eigen::MatrixXd isoV;
			Eigen::MatrixXi isoE;
			igl::isolines(points, faces, Eigen::VectorXd(fun), 20, isoV, isoE);
			igl::write_triangle_mesh("foo.obj", points, faces);
			points = isoV;
			edges = isoE;
		}

		Eigen::MatrixXd V;
		Eigen::MatrixXi E;
		Eigen::VectorXi I, J;
		igl::remove_unreferenced(points, edges, V, E, I);
		igl::remove_duplicate_vertices(V, E, 1e-14, points, I, J, edges);

		// Remove loops
		int last = edges.rows() - 1;
		int new_size = edges.rows();
		for (int i = 0; i <= last; ++i) {
			if (edges(i, 0) == edges(i, 1)) {
				edges.row(i) = edges.row(last);
				--last;
				--i;
				--new_size;
			}
		}
		edges.conservativeResize(new_size, edges.cols());

		save_edges(name, points, edges);
	}

	// void State::compute_poly_basis_error(const std::string &path)
	// {



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
