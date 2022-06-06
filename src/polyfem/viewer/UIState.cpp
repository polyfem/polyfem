#include "UIState.hpp"

#include <polyfem/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh3D/Mesh3D.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/problem/ElasticProblem.hpp>
#include <polyfem/utils/RefElementSampler.hpp>

// #include <polyfem/LinearSolver.hpp>
#include <polyfem/utils/EdgeSampler.hpp>

#include <polyfem/utils/Logger.hpp>

#include <igl/per_face_normals.h>
#include <igl/per_corner_normals.h>
#include <igl/Timer.h>
#include <igl/isolines.h>

#include <cstdlib>
#include <fstream>

#ifdef __APPLE__
const int line_width = 1;
#else
const int line_width = 2;
#endif

// ... or using a custom callback
//       viewer_.ngui->addVariable<bool>("bool",[&](bool val) {
//     boolVariable = val; // set
// },[&]() {
//     return boolVariable; // get
// });

using namespace Eigen;
using namespace polyfem::utils;

// int offscreen_screenshot(igl::opengl::glfw::Viewer &viewer, const std::string &path);

// void add_spheres(igl::opengl::glfw::Viewer &viewer0, const Eigen::MatrixXd &P, double radius) {
// 	Eigen::MatrixXd V = viewer0.data().V, VS, VN;
// 	Eigen::MatrixXi F = viewer0.data().F, FS;
// 	igl::read_triangle_mesh(POLYFEM_MESH_PATH "sphere.ply", VS, FS);

// 	Eigen::RowVector3d minV = VS.colwise().minCoeff();
// 	Eigen::RowVector3d maxV = VS.colwise().maxCoeff();
// 	VS.rowwise() -= minV + 0.5 * (maxV - minV);
// 	VS /= (maxV - minV).maxCoeff();
// 	VS *= 2.0 * radius;

// 	Eigen::MatrixXd C = viewer0.data().F_material_ambient.leftCols(3);
// 	C *= 10;

// 	int nv = V.rows();
// 	int nf = 0;
// 	V.conservativeResize(V.rows() + P.rows() * VS.rows(), V.cols());
// 	F.conservativeResize(nf + P.rows() * FS.rows(), F.cols());
// 	C.conservativeResize(C.rows() + P.rows() * FS.rows(), C.cols());
// 	for (int i = 0; i < P.rows(); ++i) {
// 		V.middleRows(nv, VS.rows()) = VS.rowwise() + P.row(i);
// 		F.middleRows(nf, FS.rows()) = FS.array() + nv;
// 		C.middleRows(nf, FS.rows()).rowwise() = Eigen::RowVector3d(142, 68, 173)/255.;
// 		nv += VS.rows();
// 		nf += FS.rows();
// 	}

// 	igl::per_corner_normals(V, F, 20.0, VN);

// 	C = Eigen::RowVector3d(142, 68, 173)/255.;

// 	igl::opengl::glfw::Viewer viewer;
// 	viewer.data().set_mesh(V, F);
// 	// viewer.data().add_points(P, Eigen::Vector3d(0,1,1).transpose());
// 	viewer.data().set_normals(VN);
// 	viewer.data().set_face_based(false);
// 	viewer.data().set_colors(C);
// 	viewer.data().lines = viewer0.data().lines;
// 	viewer.data().show_lines = false;
// 	viewer.data().line_width = line_width;
// 	// viewer.core().background_color.setOnes();
// 	// viewer.core().set_rotation_type(igl::opengl::ViewerCore::RotationType::ROTATION_TYPE_TRACKBALL);

// 	// #ifdef IGL_VIEWER_WITH_NANOGUI
// 	// viewer.callback_init = [&](igl::opengl::glfw::Viewer& viewer_) {
// 	// 	viewer_.ngui->addButton("Save screenshot", [&] {
// 	// 		// Allocate temporary buffers
// 	// 		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(6400, 4000);
// 	// 		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(6400, 4000);
// 	// 		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(6400, 4000);
// 	// 		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(6400, 4000);

// 	// 		// Draw the scene in the buffers
// 	// 		viewer_.core.draw_buffer(viewer.data(),viewer.opengl,false,R,G,B,A);
// 	// 		A.setConstant(255);

// 	// 		// Save it to a PNG
// 	// 		igl::png::writePNG(R,G,B,A,"foo.png");
// 	// 	});
// 	// 	viewer_.ngui->addButton("Load", [&] {
// 	// 		igl::deserialize(viewer.core, "core", "viewer.core");
// 	// 	});
// 	// 	viewer_.ngui->addButton("Save", [&] {
// 	// 		igl::serialize(viewer.core, "core", "viewer.core");
// 	// 	});
// 	// 	viewer_.screen->performLayout();
// 	// 	return false;
// 	// };
// 	// #endif

// 	viewer.launch();
// }

namespace polyfem
{
	using namespace assembler;
	using namespace mesh;

	void UIState::get_plot_edges(const Mesh &mesh, const std::vector<ElementBases> &bases, const std::vector<ElementBases> &gbases, const int n_samples, const std::vector<bool> &valid_elements, const Visualizations &layer, Eigen::MatrixXd &pp0, Eigen::MatrixXd &pp1)
	{
		Eigen::MatrixXd samples_simplex, samples_cube, mapped, p0, p1;
		std::vector<Eigen::MatrixXd> p0v, p1v;
		std::vector<AssemblyValues> tmp_val;

		std::vector<bool> valid_polytopes(valid_elements.size(), false);
		const int actual_dim = mesh.dimension();

		if (mesh.is_volume())
		{
			EdgeSampler::sample_3d_simplex(n_samples, samples_simplex);
			EdgeSampler::sample_3d_cube(n_samples, samples_cube);
		}
		else
		{
			EdgeSampler::sample_2d_simplex(n_samples, samples_simplex);
			EdgeSampler::sample_2d_cube(n_samples, samples_cube);
		}

		for (std::size_t i = 0; i < bases.size(); ++i)
		{
			if (!valid_elements[i])
				continue;

			if (mesh.is_polytope(i))
			{
				valid_polytopes[i] = true;
				continue;
			}

			auto samples = mesh.is_simplex(i) ? samples_simplex : samples_cube;

			const int n_edges = mesh.is_simplex(i) ? (mesh.is_volume() ? 6 : 3) : (mesh.is_volume() ? 12 : 4);

			Eigen::MatrixXd result(samples.rows(), samples.cols());
			result.setZero();

			if (!state.problem->is_scalar() && layer == Visualizations::Solution)
			{
				const ElementBases &bs = bases[i];
				bs.evaluate_bases(samples, tmp_val);

				for (std::size_t j = 0; j < bs.bases.size(); ++j)
				{
					const Basis &b = bs.bases[j];
					const auto &tmp = tmp_val[j].val;

					for (std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						for (int d = 0; d < actual_dim; ++d)
						{
							result.col(d) += b.global()[ii].val * tmp * state.sol(b.global()[ii].index * actual_dim + d);
						}
					}
				}
			}

			gbases[i].eval_geom_mapping(samples, mapped);

			for (int j = 0; j < n_edges; ++j)
			{
				for (int k = 0; k < n_samples - 1; ++k)
				{
					p0v.push_back(mapped.row(j * n_samples + k) + result.row(j * n_samples + k));
					p1v.push_back(mapped.row(j * n_samples + k + 1) + result.row(j * n_samples + k + 1));
				}
			}
		}

		mesh.get_edges(p0, p1, valid_polytopes);

		pp0.resize(p0.rows() + p0v.size(), mesh.dimension());
		pp1.resize(p1.rows() + p1v.size(), mesh.dimension());

		for (size_t i = 0; i < p1v.size(); ++i)
		{
			pp0.row(i) = p0v[i];
			pp1.row(i) = p1v[i];
		}

		pp0.bottomRows(p0.rows()) = p0;
		pp1.bottomRows(p1.rows()) = p1;
	}

	void UIState::plot_selection_and_index(const bool recenter)
	{
		std::vector<bool> valid_elements(normalized_barycenter.rows(), false);
		for (auto idx : selected_elements)
		{
			valid_elements[idx] = true;
		}

		data(Visualizations::NavigationIndex).clear();

		// if(layer == Visualizations::InputMesh)
		{
			// const long n_tris = show_clipped_elements(tri_pts, tri_faces, element_ranges, valid_elements, false, Visualizations::NavigationIndex, recenter);
			// color_mesh(n_tris, valid_elements, Visualizations::NavigationIndex);
		}
		// else
		{
			const long n_tris = show_clipped_elements(vis_pts, vis_faces, vis_element_ranges, valid_elements, false, Visualizations::NavigationIndex, recenter);
			color_mesh(n_tris, valid_elements, Visualizations::NavigationIndex);
		}

		if (state.mesh->is_volume())
		{
			const auto p = state.mesh->point(current_3d_index.vertex);
			const auto p1 = state.mesh->point(dynamic_cast<Mesh3D *>(state.mesh.get())->switch_vertex(current_3d_index).vertex);

			data(Visualizations::NavigationIndex).add_points(p, MatrixXd::Zero(1, 3));
			data(Visualizations::NavigationIndex).add_edges(p, p1, RowVector3d(1, 1, 0));

			data(Visualizations::NavigationIndex).add_points(state.mesh->face_barycenter(current_3d_index.face), RowVector3d(1, 0, 0));
		}
		else
		{
			const auto p = state.mesh->point(current_2d_index.vertex);
			const auto p1 = state.mesh->point(dynamic_cast<Mesh2D *>(state.mesh.get())->switch_vertex(current_2d_index).vertex);

			data(Visualizations::NavigationIndex).add_points(p, MatrixXd::Zero(1, 3));
			data(Visualizations::NavigationIndex).add_edges(p, p1, RowVector3d(1, 1, 0));

			data(Visualizations::NavigationIndex).add_points(state.mesh->face_barycenter(current_2d_index.face), RowVector3d(1, 0, 0));
		}
	}

	void UIState::color_mesh(const int n_tris, const std::vector<bool> &valid_elements, const Visualizations &layer)
	{
		const std::vector<ElementType> &ele_tag = state.mesh->elements_tag();

		Eigen::MatrixXd cols(n_tris, 3);
		cols.setZero();

		int from = 0;
		if (layer == Visualizations::DiscrMesh)
		{
			Eigen::MatrixXd tmp(5, 3);
			tmp << 255, 234, 167,
				250, 177, 160,
				255, 118, 117,
				253, 121, 168,
				232, 67, 147;

			tmp /= 255;

			for (std::size_t i = 1; i < element_ranges.size(); ++i)
			{
				if (!valid_elements[i - 1])
					continue;

				const ElementType type = ele_tag[i - 1];
				const int range = element_ranges[i] - element_ranges[i - 1];

				for (int c = 0; c < 3; ++c)
					cols.block(from, c, range, 1).setConstant(tmp(std::min(5, state.disc_orders(i - 1)) - 1, c));
				from += range;
			}
		}
		else if (layer == Visualizations::NavigationIndex)
		{
			cols.col(1).setOnes();
		}
		else
		{
			for (std::size_t i = 1; i < element_ranges.size(); ++i)
			{
				if (!valid_elements[i - 1])
					continue;

				const ElementType type = ele_tag[i - 1];
				const int range = element_ranges[i] - element_ranges[i - 1];

				switch (type)
				{
					// violet
				case ElementType::Simplex:
					cols.block(from, 0, range, 1).setConstant(155. / 255.);
					cols.block(from, 1, range, 1).setConstant(89. / 255.);
					cols.block(from, 2, range, 1).setConstant(182. / 255.);
					break;

					// dark green
				case ElementType::RegularInteriorCube:
				case ElementType::RegularBoundaryCube:
					cols.block(from, 0, range, 1).setConstant(30. / 255.);
					cols.block(from, 1, range, 1).setConstant(174. / 255.);
					cols.block(from, 2, range, 1).setConstant(96. / 255.);
					break;

					// orange
				case ElementType::SimpleSingularInteriorCube:
				case ElementType::SimpleSingularBoundaryCube:
				case ElementType::MultiSingularInteriorCube:
				case ElementType::MultiSingularBoundaryCube:
				case ElementType::InterfaceCube:
					cols.block(from, 0, range, 1).setConstant(231. / 255.);
					cols.block(from, 1, range, 1).setConstant(76. / 255.);
					cols.block(from, 2, range, 1).setConstant(60. / 255.);
					break;

					// light blue
				case ElementType::BoundaryPolytope:
				case ElementType::InteriorPolytope:
					cols.block(from, 0, range, 1).setConstant(52. / 255.);
					cols.block(from, 1, range, 1).setConstant(152. / 255.);
					cols.block(from, 2, range, 1).setConstant(219. / 255.);
					break;

					// grey
				case ElementType::Undefined:
					cols.block(from, 0, range, 3).setConstant(0.5);
					break;
				}

				from += range;
			}
		}

		data(layer).set_colors(cols);
		data(layer).show_overlay = 1;
		data(layer).show_faces = 1;

		viewer.core().lighting_factor = (light_enabled ? 1.f : 0.f);
		// if(!light_enabled){
		// 	data(layer).F_material_specular.setZero();
		// 	data(layer).V_material_specular.setZero();
		// 	data(layer).dirty |= igl::opengl::MeshGL::DIRTY_DIFFUSE;

		// 	data(layer).V_material_ambient *= 2;
		// 	data(layer).F_material_ambient *= 2;
		// }
	}

	long UIState::clip_elements(const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tris, const std::vector<int> &ranges, std::vector<bool> &valid_elements, const bool map_edges, const Visualizations &layer)
	{
		data(layer).clear();

		valid_elements.resize(normalized_barycenter.rows());

		if (!is_slicing)
		{
			std::fill(valid_elements.begin(), valid_elements.end(), true);
			data(layer).set_mesh(pts, tris);

			viewer.core().lighting_factor = (light_enabled ? 1.f : 0.f);
			// if(!light_enabled){
			// 	data(layer).F_material_specular.setZero();
			// 	data(layer).V_material_specular.setZero();
			// 	data(layer).V_material_ambient *= 4;
			// 	data(layer).F_material_ambient *= 4;

			// 	data(layer).dirty |= igl::opengl::MeshGL::DIRTY_DIFFUSE;
			// }

			if (state.mesh->is_volume())
			{
				MatrixXd normals;
				igl::per_face_normals(pts, tris, normals);
				data(layer).set_normals(normals);

				igl::per_corner_normals(pts, tris, 20, normals);
				data(layer).set_normals(normals);
				data(layer).set_face_based(false);
			}

			MatrixXd p0, p1;

			if (map_edges)
			{
				const auto &current_bases = state.iso_parametric() ? state.bases : state.geom_bases;
				get_plot_edges(*state.mesh, state.bases, current_bases, 20, valid_elements, layer, p0, p1);
			}
			else
			{
				state.mesh->get_edges(p0, p1);
			}

			data(layer).line_width = line_width;
			data(layer).add_edges(p0, p1, MatrixXd::Zero(1, 3));
			data(layer).show_lines = 0;

			return tris.rows();
		}

		for (long i = 0; i < normalized_barycenter.rows(); ++i)
			valid_elements[i] = normalized_barycenter(i, slice_coord) < slice_position;

		return show_clipped_elements(pts, tris, ranges, valid_elements, map_edges, layer);
	}

	long UIState::show_clipped_elements(const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tris, const std::vector<int> &ranges, const std::vector<bool> &valid_elements, const bool map_edges, const Visualizations &layer, const bool recenter)
	{
		data(layer).set_face_based(false);

		int n_vis_valid_tri = 0;

		for (long i = 0; i < normalized_barycenter.rows(); ++i)
		{
			if (valid_elements[i])
				n_vis_valid_tri += ranges[i + 1] - ranges[i];
		}

		MatrixXi valid_tri(n_vis_valid_tri, tris.cols());

		int from = 0;
		for (std::size_t i = 1; i < ranges.size(); ++i)
		{
			if (!valid_elements[i - 1])
				continue;

			const int range = ranges[i] - ranges[i - 1];

			valid_tri.block(from, 0, range, tri_faces.cols()) = tris.block(ranges[i - 1], 0, range, tris.cols());

			from += range;
		}

		data(layer).set_mesh(pts, valid_tri);

		viewer.core().lighting_factor = (light_enabled ? 1.f : 0.f);
		// if(!light_enabled){
		// 	data(layer).F_material_specular.setZero();
		// 	data(layer).V_material_specular.setZero();
		// 	data(layer).dirty |= igl::opengl::MeshGL::DIRTY_DIFFUSE;
		// 	data(layer).V_material_ambient *= 2;
		// 	data(layer).F_material_ambient *= 2;
		// }

		if (state.mesh->is_volume())
		{
			MatrixXd normals;
			igl::per_face_normals(pts, valid_tri, normals);
			data(layer).set_normals(normals);

			igl::per_corner_normals(pts, valid_tri, 20, normals);
			data(layer).set_normals(normals);
			data(layer).set_face_based(false);
		}

		if (recenter)
			viewer.core().align_camera_center(pts, valid_tri);

		MatrixXd p0, p1;
		if (map_edges)
		{
			const auto &current_bases = state.iso_parametric() ? state.bases : state.geom_bases;
			get_plot_edges(*state.mesh, state.bases, current_bases, 20, valid_elements, layer, p0, p1);
		}
		else
		{
			state.mesh->get_edges(p0, p1, valid_elements);
		}

		data(layer).line_width = line_width;
		data(layer).add_edges(p0, p1, MatrixXd::Zero(1, 3));
		data(layer).show_lines = 0;

		return valid_tri.rows();
	}

	void UIState::interpolate_function(const MatrixXd &fun, MatrixXd &result)
	{
		// MatrixXd tmp;
		std::vector<AssemblyValues> tmp_val;

		int actual_dim = 1;
		if (!state.problem->is_scalar())
			actual_dim = state.mesh->dimension();

		result.resize(vis_pts.rows(), actual_dim);

		int index = 0;
		const auto &sampler = state.ref_element_sampler;

		for (int i = 0; i < int(state.bases.size()); ++i)
		{
			const ElementBases &bs = state.bases[i];
			MatrixXd local_pts;

			if (state.mesh->is_simplex(i))
				local_pts = sampler.simplex_points();
			else if (state.mesh->is_cube(i))
				local_pts = sampler.cube_points();
			else
				local_pts = vis_pts_poly[i];

			MatrixXd local_res = MatrixXd::Zero(local_pts.rows(), actual_dim);
			bs.evaluate_bases(local_pts, tmp_val);
			for (std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				const auto &tmp = tmp_val[j].val;

				for (int d = 0; d < actual_dim; ++d)
				{
					for (std::size_t ii = 0; ii < b.global().size(); ++ii)
						local_res.col(d) += b.global()[ii].val * tmp * fun(b.global()[ii].index * actual_dim + d);
				}
			}

			result.block(index, 0, local_res.rows(), actual_dim) = local_res;
			index += local_res.rows();
		}
	}

	void UIState::interpolate_grad_function(const MatrixXd &fun, MatrixXd &result)
	{
		// MatrixXd tmp;
		std::vector<AssemblyValues> tmp_val;

		int actual_dim = 1;
		if (!state.problem->is_scalar())
			actual_dim *= state.mesh->dimension();

		result.resize(vis_pts.rows(), actual_dim * state.mesh->dimension());

		int index = 0;
		std::vector<Eigen::MatrixXd> j_g_mapping;
		std::vector<Eigen::MatrixXd> grads;

		const auto &sampler = state.ref_element_sampler;

		for (int i = 0; i < int(state.bases.size()); ++i)
		{
			const ElementBases &bs = state.bases[i];
			MatrixXd local_pts;

			if (state.mesh->is_simplex(i))
				local_pts = sampler.simplex_points();
			else if (state.mesh->is_cube(i))
				local_pts = sampler.cube_points();
			else
				local_pts = vis_pts_poly[i];

			bs.eval_geom_mapping_grads(local_pts, j_g_mapping);
			MatrixXd local_res = MatrixXd::Zero(local_pts.rows(), actual_dim * state.mesh->dimension());
			grads.resize(state.mesh->dimension());

			for (int c = 0; c < state.mesh->dimension(); ++c)
				grads[c].resize(j_g_mapping.size(), bs.bases.size());

			for (std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				for (size_t n = 0; n < j_g_mapping.size(); ++n)
				{
					Eigen::RowVectorXd grad(state.mesh->dimension());
					bs.evaluate_grads(local_pts, tmp_val);
					for (int c = 0; c < state.mesh->dimension(); ++c)
					{
						// bs.evaluate_grads(local_pts, c, tmp);
						// grad(c) = tmp(n, j);
						grad(c) = tmp_val[j].grad(n, c);
					}

					grad = grad * j_g_mapping[n].inverse().transpose();

					for (int c = 0; c < state.mesh->dimension(); ++c)
						grads[c](n, j) = grad(c);
				}
			}

			for (int c = 0; c < state.mesh->dimension(); ++c)
			{
				// bs.evaluate_grads(local_pts, c, tmp);
				for (std::size_t j = 0; j < bs.bases.size(); ++j)
				{
					const Basis &b = bs.bases[j];

					for (int d = 0; d < actual_dim; ++d)
					{
						for (std::size_t ii = 0; ii < b.global().size(); ++ii)
							local_res.col(c * actual_dim + d) += b.global()[ii].val * grads[c].col(j) * fun(b.global()[ii].index * actual_dim + d);
					}
				}
			}

			result.block(index, 0, local_res.rows(), local_res.cols()) = local_res;

			index += local_res.rows();
		}
	}

	igl::opengl::ViewerData &UIState::data(const Visualizations &layer)
	{
		size_t index = viewer.mesh_index(layer);
		assert(viewer.data_list[index].id == layer);
		return viewer.data_list[index];
	}

	void UIState::reset_flags(const Visualizations &layer, bool clear)
	{
		if (clear)
			data(layer).clear();

		data(layer).show_overlay = 1;
		data(layer).show_faces = 1;
		data(layer).show_lines = 1;
		data(layer).show_vertex_labels = 0;
		data(layer).show_face_labels = 0;
	}

	void UIState::hide_data(const Visualizations &layer)
	{
		if (vis_flags[layer].empty())
		{
			vis_flags[layer] = {
				data(layer).show_overlay,
				data(layer).show_faces,
				data(layer).show_lines,
				data(layer).show_vertex_labels,
				data(layer).show_face_labels};
		}
		data(layer).show_overlay = 0;
		data(layer).show_faces = 0;
		data(layer).show_lines = 0;
		data(layer).show_vertex_labels = 0;
		data(layer).show_face_labels = 0;
	}

	void UIState::show_data(const Visualizations &layer)
	{
		const auto &flags = vis_flags[layer];

		data(layer).show_overlay = flags[0];
		data(layer).show_faces = flags[1];
		data(layer).show_lines = flags[2];
		data(layer).show_vertex_labels = flags[3];
		data(layer).show_face_labels = flags[4];
	}

	UIState::UIState()
	{
		for (int i = 0; i < Visualizations::TotalVisualizations; ++i)
		{
			if (i > 0)
				viewer.append_mesh();
			viewer.data_list[i].id = i;
		}

		vis_flags.resize(Visualizations::TotalVisualizations);

		available_visualizations.resize(Visualizations::TotalVisualizations);
		available_visualizations.setConstant(false);

		visible_visualizations.resize(Visualizations::TotalVisualizations);
		visible_visualizations.setConstant(false);

		viewer.selected_data_index = 0;
	}

	void UIState::plot_function(const MatrixXd &fun, const Visualizations &layer, double min, double max)
	{
		if (show_funs_in_3d)
			light_enabled = true;
		else
			light_enabled = state.mesh->is_volume();

		MatrixXd col;
		std::vector<bool> valid_elements;
		const auto &sampler = state.ref_element_sampler;

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		if (fun.cols() != 1)
		{
			MatrixXd ffun(vis_pts.rows(), 1);
			// int size = state.mesh->dimension();

			const auto &assembler = state.assembler;

			MatrixXd stresses;
			int counter = 0;
			for (int i = 0; i < int(state.bases.size()); ++i)
			{
				const ElementBases &bs = state.bases[i];
				const ElementBases &gbs = gbases[i];
				MatrixXd local_pts;

				if (state.mesh->is_simplex(i))
					local_pts = sampler.simplex_points();
				else if (state.mesh->is_cube(i))
					local_pts = sampler.cube_points();
				else
					local_pts = vis_pts_poly[i];

				assembler.compute_scalar_value(state.formulation(), i, bs, gbs, local_pts, state.sol, stresses);

				ffun.block(counter, 0, stresses.rows(), 1) = stresses;
				counter += stresses.rows();
			}

			if (min < max)
				igl::colormap(color_map, ffun, min, max, col);
			else
				igl::colormap(color_map, ffun, true, col);

			if (min < max)
			{
				min_val = min;
				max_val = max;
			}
			else
			{
				min_val = ffun.minCoeff();
				max_val = ffun.maxCoeff();
			}

			MatrixXd ttmp = vis_pts;

			if (assembler.is_solution_displacement(state.formulation()))
			{
				// apply displacement
				for (long i = 0; i < fun.cols(); ++i)
					ttmp.col(i) += fun.col(i);
			}

			MatrixXd tmp(ttmp.rows(), 3);
			tmp.setZero();
			for (long i = 0; i < ttmp.cols(); ++i)
				tmp.col(i) = ttmp.col(i);

			clip_elements(tmp, vis_faces, vis_element_ranges, valid_elements, true, layer);

			data(layer).show_overlay = 0;
			// if(show_isolines && fun.cols() != 3)
			// {
			// 	Eigen::MatrixXd isoV;
			// 	Eigen::MatrixXi isoE;
			// 	igl::isolines(tmp, vis_faces, Eigen::VectorXd(ffun), 20, isoV, isoE);
			// 	data(layer).set_edges(isoV,isoE,Eigen::RowVector3d(0,0,0));
			// }
		}
		else
		{

			if (min < max)
				igl::colormap(color_map, fun, min, max, col);
			else
				igl::colormap(color_map, fun, true, col);

			if (state.mesh->is_volume())
				clip_elements(vis_pts, vis_faces, vis_element_ranges, valid_elements, true, layer);
			else
			{
				MatrixXd tmp;
				tmp.resize(fun.rows(), 3);
				tmp.col(0) = vis_pts.col(0);
				tmp.col(1) = vis_pts.col(1);
				if (show_funs_in_3d)
					tmp.col(2) = fun;
				else
					tmp.col(2).setZero();
				clip_elements(tmp, vis_faces, vis_element_ranges, valid_elements, true, layer);

				if (show_isolines)
				{
					Eigen::MatrixXd isoV;
					Eigen::MatrixXi isoE;
					igl::isolines(tmp, vis_faces, Eigen::VectorXd(fun), 20, isoV, isoE);
					data(layer).set_edges(isoV, isoE, Eigen::RowVector3d(0, 0, 0));
					data(layer).show_overlay = 1;
				}
				else
					data(layer).show_overlay = 0;
			}

			if (min < max)
			{
				min_val = min;
				max_val = max;
			}
			else
			{
				min_val = fun.minCoeff();
				max_val = fun.maxCoeff();
			}
		}

		data(layer).set_colors(col);

		viewer.core().lighting_factor = (light_enabled ? 1.f : 0.f);
		// if(!light_enabled){
		// 	data(layer).F_material_specular.setZero();
		// 	data(layer).V_material_specular.setZero();

		// 	data(layer).V_material_ambient *= 2;
		// 	data(layer).F_material_ambient *= 2;
		// 	data(layer).dirty |= igl::opengl::MeshGL::DIRTY_DIFFUSE;
		// }
	}

	UIState &UIState::ui_state()
	{
		static UIState instance;

		return instance;
	}

	void UIState::clear()
	{
		visible_visualizations.setConstant(false);
		viewer.selected_data_index = 0;
		for (int i = 0; i < Visualizations::TotalVisualizations; ++i)
		{
			hide_data(static_cast<Visualizations>(i));
		}
	}

	void UIState::show_sidesets()
	{
		if (!state.mesh)
		{
			return;
		}

		Eigen::MatrixXd pts;
		Eigen::MatrixXi faces;
		Eigen::MatrixXd sidesets;
		Eigen::MatrixXd col;
		state.get_sidesets(pts, faces, sidesets);
		if (sidesets.size() != 0)
			igl::colormap(color_map, sidesets, true, col);

		if (visible_visualizations(Visualizations::Sidesets) && !available_visualizations[Visualizations::Sidesets])
		{
			reset_flags(Visualizations::Sidesets);
		}

		if (state.mesh->is_volume())
		{
			data(Visualizations::Sidesets).set_mesh(pts, faces);
			if (col.size() != 0)
				data(Visualizations::Sidesets).set_colors(col);
			Eigen::MatrixXd p0, p1;
			state.mesh->get_edges(p0, p1);

			data(Visualizations::Sidesets).line_width = line_width;
			data(Visualizations::Sidesets).add_edges(p0, p1, MatrixXd::Zero(1, 3));
			data(Visualizations::Sidesets).show_lines = 0;
		}
		else
		{
			data(Visualizations::Sidesets).show_lines = 1;
			data(Visualizations::Sidesets).line_width = line_width;
			if (col.size() != 0)
				data(Visualizations::Sidesets).set_edges(pts, faces, col);
		}

		if (visible_visualizations(Visualizations::Sidesets) && !available_visualizations[Visualizations::Sidesets])
		{
			available_visualizations[Visualizations::Sidesets] = true;
			vis_flags[Visualizations::Sidesets].clear();
			hide_data(Visualizations::Sidesets);
		}

		if (visible_visualizations(Visualizations::Sidesets))
			show_data(Visualizations::Sidesets);
	}

	void UIState::show_mesh()
	{
		if (!state.mesh)
		{
			return;
		}

		if (visible_visualizations(Visualizations::InputMesh) && !available_visualizations[Visualizations::InputMesh])
		{
			reset_flags(Visualizations::InputMesh);
			std::vector<bool> valid_elements;
			const long n_tris = clip_elements(tri_pts, tri_faces, element_ranges, valid_elements, false, Visualizations::InputMesh);
			color_mesh(n_tris, valid_elements, Visualizations::InputMesh);

			available_visualizations[Visualizations::InputMesh] = true;
			vis_flags[Visualizations::InputMesh].clear();
			hide_data(Visualizations::InputMesh);
		}

		if (visible_visualizations(Visualizations::InputMesh))
			show_data(Visualizations::InputMesh);

		if (visible_visualizations(Visualizations::DiscrMesh) && !available_visualizations[Visualizations::DiscrMesh])
		{
			reset_flags(Visualizations::DiscrMesh);
			std::vector<bool> valid_elements;
			const long n_tris = clip_elements(tri_pts, tri_faces, element_ranges, valid_elements, false, Visualizations::DiscrMesh);
			color_mesh(n_tris, valid_elements, Visualizations::DiscrMesh);

			available_visualizations[Visualizations::DiscrMesh] = true;
			vis_flags[Visualizations::DiscrMesh].clear();
			hide_data(Visualizations::DiscrMesh);
		}

		if (visible_visualizations(Visualizations::DiscrMesh))
			show_data(Visualizations::DiscrMesh);

		// for(int i = 0; i < state.mesh->n_faces(); ++i)
		// {
		// 	MatrixXd p = state.mesh->face_barycenter(i);
		// 	data(Visualizations::InputMesh).add_label(p.transpose(), std::to_string(i));
		// }

		// for(int i = 0; i < state.mesh->n_edges(); ++i)
		// {
		// 	MatrixXd p = state.mesh->edge_barycenter(i);
		// 	data(Visualizations::InputMesh).add_label(p.transpose(), std::to_string(i));
		// }

		// TODO Text is impossible to hide :(
		// visible_visualizations(Visualizations::ElementId) = true;
		if (visible_visualizations(Visualizations::ElementId) && !available_visualizations[Visualizations::ElementId])
		{
			reset_flags(Visualizations::ElementId);
			available_visualizations[Visualizations::ElementId] = true;
			for (int i = 0; i < state.mesh->n_elements(); ++i)
			{
				MatrixXd p = state.mesh->is_volume() ? state.mesh->cell_barycenter(i) : state.mesh->face_barycenter(i);
				data(Visualizations::ElementId).add_label(p.transpose(), std::to_string(i));
			}

			vis_flags[Visualizations::ElementId].clear();
			hide_data(Visualizations::ElementId);
		}

		if (visible_visualizations(Visualizations::ElementId))
			show_data(Visualizations::ElementId);

		if (visible_visualizations(Visualizations::VertexId) && !available_visualizations[Visualizations::VertexId])
		{
			reset_flags(Visualizations::VertexId);
			available_visualizations[Visualizations::VertexId] = true;

			for (int i = 0; i < state.mesh->n_vertices(); ++i)
			{
				const auto p = state.mesh->point(i);
				data(Visualizations::VertexId).add_label(p.transpose(), std::to_string(i));
			}

			vis_flags[Visualizations::VertexId].clear();
			hide_data(Visualizations::VertexId);
		}

		if (visible_visualizations(Visualizations::VertexId))
			show_data(Visualizations::VertexId);
	}

	void UIState::show_vis_mesh()
	{
		if (!state.mesh)
		{
			return;
		}

		if (!visible_visualizations(Visualizations::VisMesh))
			return;

		if (!available_visualizations[Visualizations::VisMesh])
		{
			reset_flags(Visualizations::VisMesh);
			std::vector<bool> valid_elements;
			clip_elements(vis_pts, vis_faces, vis_element_ranges, valid_elements, true, Visualizations::VisMesh);
			data(Visualizations::VisMesh).show_lines = 1;
			available_visualizations[Visualizations::VisMesh] = true;

			// data(Visualizations::VisMesh).add_points(state.mesh->face_barycenter(3314), Eigen::RowVector3d(1,0,0));
			// data(Visualizations::VisMesh).add_points(state.mesh->face_barycenter(3443), Eigen::RowVector3d(1,0,0));
			vis_flags[Visualizations::VisMesh].clear();
			hide_data(Visualizations::VisMesh);
		}

		if (visible_visualizations(Visualizations::VisMesh))
			show_data(Visualizations::VisMesh);
	}

	void UIState::show_nodes()
	{
		// return;
		if (!state.mesh)
		{
			return;
		}

		// const auto &current_bases = state.iso_parametric() ? state.bases : state.geom_bases;
		const auto &current_bases = state.bases;

		MatrixXd col(1, 3);
		if (!available_visualizations[Visualizations::BNodes])
		{
			reset_flags(Visualizations::BNodes);

			if (state.local_neumann_boundary.size() < 3500)
			{
				col << 0.5, 0.5, 0.5;
				for (const auto &lb : state.local_neumann_boundary)
				{
					const int e = lb.element_id();
					const ElementBases &bs = current_bases[e];

					for (int i = 0; i < lb.size(); ++i)
					{
						const int primitive_global_id = lb.global_primitive_id(i);
						const auto nodes = bs.local_nodes_for_primitive(primitive_global_id, *state.mesh);

						for (long n = 0; n < nodes.size(); ++n)
						{
							const auto &b = bs.bases[nodes(n)];
							for (size_t g = 0; g < b.global().size(); ++g)
							{
								const Local2Global &l2g = b.global()[g];
								int g_index = l2g.index;

								int ddim = 1;
								if (!state.problem->is_scalar())
								{
									g_index *= state.mesh->dimension();
									ddim = state.mesh->dimension();
								}

								bool show_node = true;

								for (int d = 0; d < ddim; ++d)
								{
									bool found = std::find(state.boundary_nodes.begin(), state.boundary_nodes.end(), g_index + d) != state.boundary_nodes.end();

									if (found)
									{
										show_node = false;
										break;
									}
								}

								if (show_node)
								{
									MatrixXd node = l2g.node;
									data(Visualizations::BNodes).add_points(node, col);

									// TODO text is impossible to hide :(
									//  data(Visualizations::NodesId).add_label(node.transpose(), std::to_string(l2g.index));
								}
							}
						}
					}
				}
			}

			int shown_boundaries = 0;
			col << 0.5, 0.5, 0.5;
			for (std::size_t i = 0; i < current_bases.size(); ++i)
			{
				const ElementBases &basis = current_bases[i];
				Eigen::MatrixXd P(basis.bases.size(), 3);

				for (std::size_t j = 0; j < basis.bases.size(); ++j)
				{
					for (std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
					{
						const Local2Global &l2g = basis.bases[j].global()[kk];
						int g_index = l2g.index;
						col << 0, 0, 0;
						bool is_boundary = false;

						if (!state.problem->is_scalar())
						{
							g_index *= state.mesh->dimension();

							for (int d = 0; d < state.mesh->dimension(); ++d)
							{
								const auto loc_d_b = std::find(state.boundary_nodes.begin(), state.boundary_nodes.end(), g_index + d) != state.boundary_nodes.end();
								is_boundary = is_boundary || loc_d_b;
								if (loc_d_b)
									col(d) = 1;
							}
						}
						else
						{
							is_boundary = std::find(state.boundary_nodes.begin(), state.boundary_nodes.end(), g_index) != state.boundary_nodes.end();
						}

						if (is_boundary)
						{
							MatrixXd node = l2g.node;
							data(Visualizations::BNodes).add_points(node, col);
							++shown_boundaries;
						}
					}
				}

				// if(shown_boundaries > 4500)
				// 	break;
			}

			available_visualizations[Visualizations::BNodes] = true;
			vis_flags[Visualizations::BNodes].clear();
			hide_data(Visualizations::BNodes);
		}

		if (visible_visualizations(Visualizations::BNodes))
			show_data(Visualizations::BNodes);

		if (!available_visualizations[Visualizations::BPNodes])
		{
			reset_flags(Visualizations::BPNodes);

			int shown_boundaries = 0;
			col << 1, 0.5, 0.5;
			for (std::size_t i = 0; i < state.pressure_bases.size(); ++i)
			{
				const ElementBases &basis = state.pressure_bases[i];
				Eigen::MatrixXd P(basis.bases.size(), 3);

				for (std::size_t j = 0; j < basis.bases.size(); ++j)
				{
					for (std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
					{
						const Local2Global &l2g = basis.bases[j].global()[kk];
						int g_index = l2g.index;
						bool is_boundary = false;

						g_index += state.mesh->dimension() * state.n_bases;

						is_boundary = std::find(state.boundary_nodes.begin(), state.boundary_nodes.end(), g_index) != state.boundary_nodes.end();

						if (is_boundary)
						{
							MatrixXd node = l2g.node;
							data(Visualizations::BPNodes).add_points(node, col);
							++shown_boundaries;
						}
					}
				}

				// if(shown_boundaries > 4500)
				// 	break;
			}

			available_visualizations[Visualizations::BPNodes] = true;
			vis_flags[Visualizations::BPNodes].clear();
			hide_data(Visualizations::BPNodes);
		}

		if (visible_visualizations(Visualizations::BPNodes))
			show_data(Visualizations::BPNodes);

		if (state.n_pressure_bases <= 3500)
		{
			if ((visible_visualizations(Visualizations::PNodes) || visible_visualizations(Visualizations::NodesId)) && !available_visualizations[Visualizations::PNodes])
			{
				reset_flags(Visualizations::PNodes);

				col << 142. / 255., 68. / 255., 173. / 255.;

				for (std::size_t i = 0; i < state.pressure_bases.size(); ++i)
				{
					const ElementBases &basis = state.pressure_bases[i];
					Eigen::MatrixXd P(basis.bases.size(), 3);

					for (std::size_t j = 0; j < basis.bases.size(); ++j)
					{
						for (std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
						{
							const Local2Global &l2g = basis.bases[j].global()[kk];
							int g_index = l2g.index;

							MatrixXd node = l2g.node;
							data(Visualizations::PNodes).add_points(node, col);

							// TODO text is impossible to hide :(
							//  data(Visualizations::NodesId).add_label(node.transpose(), std::to_string(g_index));
						}
					}
				}

				available_visualizations[Visualizations::PNodes] = true;
				vis_flags[Visualizations::PNodes].clear();
				hide_data(Visualizations::PNodes);
			}

			if (visible_visualizations(Visualizations::PNodes))
				show_data(Visualizations::PNodes);
		}

		if (state.n_bases > 3500)
			return;

		if ((visible_visualizations(Visualizations::Nodes) || visible_visualizations(Visualizations::NodesId)) && !available_visualizations[Visualizations::Nodes])
		{
			reset_flags(Visualizations::Nodes);

			col << 142. / 255., 68. / 255., 173. / 255.;

			for (std::size_t i = 0; i < current_bases.size(); ++i)
			{
				const ElementBases &basis = current_bases[i];
				// Eigen::MatrixXd P(basis.bases.size(), 3);

				for (std::size_t j = 0; j < basis.bases.size(); ++j)
				{
					for (std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
					{
						const Local2Global &l2g = basis.bases[j].global()[kk];
						int g_index = l2g.index;

						if (!state.problem->is_scalar())
							g_index *= state.mesh->dimension();

						MatrixXd node = l2g.node;
						data(Visualizations::Nodes).add_points(node, col);

						// TODO text is impossible to hide :(
						//  data(Visualizations::NodesId).add_label(node.transpose(), std::to_string(g_index));
					}
				}
			}

			available_visualizations[Visualizations::Nodes] = true;
			vis_flags[Visualizations::Nodes].clear();
			hide_data(Visualizations::Nodes);
		}

		if (visible_visualizations(Visualizations::Nodes))
			show_data(Visualizations::Nodes);

		// TODO text is impossible to hide :(
		//  if(visible_visualizations(Visualizations::NodesId))
		//  show_data(Visualizations::NodesId);
	}

	void UIState::show_error()
	{
		if (!state.mesh)
		{
			return;
		}
		if (state.sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}
		if (!state.problem->has_exact_sol())
		{
			return;
		}
		double tend = state.args.value("tend", 1.0); // default=1
		if (tend <= 0)
			tend = 1;

		if (visible_visualizations(Visualizations::Error) && !available_visualizations[Visualizations::Error])
		{
			reset_flags(Visualizations::Error);
			MatrixXd global_sol;
			MatrixXd exact_sol;

			interpolate_function(state.sol, global_sol);
			state.problem->exact(vis_pts, tend, exact_sol);

			const MatrixXd err = (global_sol - exact_sol).eval().rowwise().norm();
			plot_function(err, Visualizations::Error);
			data(Visualizations::Error).show_lines = 0;

			available_visualizations[Visualizations::Error] = true;
			vis_flags[Visualizations::Error].clear();
			hide_data(Visualizations::Error);
		}

		if (visible_visualizations(Visualizations::ErrorGrad) && !available_visualizations[Visualizations::ErrorGrad])
		{
			reset_flags(Visualizations::ErrorGrad);
			MatrixXd global_sol;
			MatrixXd exact_sol;

			interpolate_grad_function(state.sol, global_sol);
			state.problem->exact_grad(vis_pts, tend, exact_sol);

			const MatrixXd err = (global_sol - exact_sol).eval().rowwise().norm();
			plot_function(err, Visualizations::ErrorGrad);
			data(Visualizations::ErrorGrad).show_lines = 0;

			available_visualizations[Visualizations::ErrorGrad] = true;
			vis_flags[Visualizations::ErrorGrad].clear();
			hide_data(Visualizations::ErrorGrad);
		}

		if (visible_visualizations(Visualizations::ErrorGrad))
			show_data(Visualizations::ErrorGrad);
		if (visible_visualizations(Visualizations::Error))
			show_data(Visualizations::Error);
	}

	void UIState::show_basis()
	{
		int actual_dim = 1;
		if (!state.problem->is_scalar())
			actual_dim = state.mesh->dimension();

		if (!state.mesh)
		{
			return;
		}
		if (vis_basis < 0 || vis_basis >= state.n_bases * actual_dim)
			return;
		if (tri_faces.size() <= 0)
			return;

		available_visualizations(Visualizations::VisBasis) = false;
		if (visible_visualizations(Visualizations::VisBasis) && !available_visualizations[Visualizations::VisBasis])
		{
			reset_flags(Visualizations::VisBasis);
			MatrixXd fun = MatrixXd::Zero(state.n_bases * actual_dim, 1);
			fun(vis_basis) = 1;

			MatrixXd global_fun;
			interpolate_function(fun, global_fun);

			// std::cout<<global_fun.minCoeff()<<" "<<global_fun.maxCoeff()<<std::endl;
			plot_function(global_fun.col(0), Visualizations::VisBasis);
			available_visualizations[Visualizations::VisBasis] = true;
			vis_flags[Visualizations::VisBasis].clear();
			hide_data(Visualizations::VisBasis);
		}

		if (visible_visualizations(Visualizations::VisBasis))
			show_data(Visualizations::VisBasis);
	}

	void UIState::show_sol()
	{
		if (!state.mesh)
		{
			return;
		}

		if (visible_visualizations(Visualizations::Solution) && !available_visualizations[Visualizations::Solution])
		{
			reset_flags(Visualizations::Solution);
			MatrixXd global_sol;
			interpolate_function(state.sol, global_sol);
			plot_function(global_sol, Visualizations::Solution);
			data(Visualizations::Solution).show_lines = 0;

			available_visualizations[Visualizations::Solution] = true;
			vis_flags[Visualizations::Solution].clear();
			hide_data(Visualizations::Solution);
		}

		if (visible_visualizations(Visualizations::Solution))
			show_data(Visualizations::Solution);
	}

	void UIState::show_linear_reproduction()
	{
		auto ff = [](double x, double y) { return -0.1 + .3 * x - .5 * y; };

		MatrixXd fun = MatrixXd::Zero(state.n_bases, 1);

		for (std::size_t i = 0; i < state.bases.size(); ++i)
		{
			const ElementBases &basis = state.bases[i];
			if (!basis.has_parameterization)
				continue;
			for (std::size_t j = 0; j < basis.bases.size(); ++j)
			{
				for (std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
				{
					const Local2Global &l2g = basis.bases[j].global()[kk];
					const int g_index = l2g.index;

					const MatrixXd node = l2g.node;
					fun(g_index) = ff(node(0), node(1));
				}
			}
		}

		MatrixXd tmp;
		interpolate_function(fun, tmp);

		MatrixXd exact_sol(vis_pts.rows(), 1);
		for (long i = 0; i < vis_pts.rows(); ++i)
			exact_sol(i) = ff(vis_pts(i, 0), vis_pts(i, 1));

		const MatrixXd global_fun = (exact_sol - tmp).array().abs();

		// std::cout<<global_fun.minCoeff()<<" "<<global_fun.maxCoeff()<<std::endl;

		available_visualizations[Visualizations::Debug] = true;
		vis_flags[Visualizations::Debug].clear();
		hide_data(Visualizations::Debug);

		plot_function(global_fun, Visualizations::Debug);

		show_data(Visualizations::Debug);
	}

	void UIState::show_quadratic_reproduction()
	{
		auto ff = [](double x, double y) {
			const double v = (2 * y - 0.9);
			return v * v;
		};

		auto ff1 = [](double x, double y) {
			Eigen::RowVector2d res;
			res(0) = 0;
			res(1) = 8 * y - 3.6;
			return res;
		};

		bool show_grad = true;

		MatrixXd fun = MatrixXd::Zero(state.n_bases, 1);

		for (std::size_t i = 0; i < state.bases.size(); ++i)
		{
			const ElementBases &basis = state.bases[i];
			if (!basis.has_parameterization)
				continue;
			for (std::size_t j = 0; j < basis.bases.size(); ++j)
			{
				for (std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
				{
					const Local2Global &l2g = basis.bases[j].global()[kk];
					const int g_index = l2g.index;

					const MatrixXd node = l2g.node;
					fun(g_index) = ff(node(0), node(1));
				}
			}
		}

		clear();
		visible_visualizations.setConstant(false);
		visible_visualizations(Visualizations::Debug) = true;
		viewer.selected_data_index = Visualizations::Debug;

		reset_flags(Visualizations::Debug);
		MatrixXd global_sol;
		MatrixXd exact_sol;

		if (show_grad)
			interpolate_grad_function(fun, global_sol);
		else
			interpolate_function(fun, global_sol);

		exact_sol.resize(vis_pts.rows(), show_grad ? 2 : 1);
		for (long i = 0; i < vis_pts.rows(); ++i)
		{
			if (show_grad)
				exact_sol.row(i) = ff1(vis_pts(i, 0), vis_pts(i, 1));
			else
				exact_sol(i) = ff(vis_pts(i, 0), vis_pts(i, 1));
		}

		const MatrixXd err = (global_sol - exact_sol).eval().rowwise().norm();
		// std::cout<<err.minCoeff()<<" "<<err.maxCoeff()<<std::endl;

		plot_function(err, Visualizations::Debug);
		data(Visualizations::Debug).show_lines = 0;

		available_visualizations[Visualizations::Debug] = true;
		vis_flags[Visualizations::Debug].clear();
		hide_data(Visualizations::Debug);
		show_data(Visualizations::Debug);
	}

	void UIState::build_vis_mesh()
	{
		if (!state.mesh)
		{
			return;
		}
		vis_element_ranges.clear();

		available_visualizations.block(Visualizations::VisMesh, 0, Visualizations::TotalVisualizations - Visualizations::VisMesh, 1).setConstant(false);

		vis_faces_poly.clear();
		vis_pts_poly.clear();

		igl::Timer timer;
		timer.start();
		logger().info("Building vis mesh...");

		const auto &current_bases = state.iso_parametric() ? state.bases : state.geom_bases;
		int faces_total_size = 0, points_total_size = 0;
		vis_element_ranges.push_back(0);

		const auto &sampler = state.ref_element_sampler;

		for (int i = 0; i < int(current_bases.size()); ++i)
		{
			const ElementBases &bs = current_bases[i];

			if (state.mesh->is_simplex(i))
			{
				faces_total_size += sampler.simplex_faces().rows();
				points_total_size += sampler.simplex_points().rows();
			}
			else if (state.mesh->is_cube(i))
			{
				faces_total_size += sampler.cube_faces().rows();
				points_total_size += sampler.cube_points().rows();
			}
			else
			{
				if (state.mesh->is_volume())
				{
					sampler.sample_polyhedron(state.polys_3d[i].first, state.polys_3d[i].second, vis_pts_poly[i], vis_faces_poly[i]);

					faces_total_size += vis_faces_poly[i].rows();
					points_total_size += vis_pts_poly[i].rows();
				}
				else
				{
					const MatrixXd &poly = state.polys[i];
					sampler.sample_polygon(poly, vis_pts_poly[i], vis_faces_poly[i]);

					faces_total_size += vis_faces_poly[i].rows();
					points_total_size += vis_pts_poly[i].rows();
				}
			}

			vis_element_ranges.push_back(faces_total_size);
		}

		vis_pts.resize(points_total_size, sampler.cube_points().cols());
		vis_faces.resize(faces_total_size, 3);

		MatrixXd mapped, tmp;
		int face_index = 0, point_index = 0;
		for (int i = 0; i < int(current_bases.size()); ++i)
		{
			const ElementBases &bs = current_bases[i];
			if (state.mesh->is_simplex(i))
			{
				bs.eval_geom_mapping(sampler.simplex_points(), mapped);
				vis_faces.block(face_index, 0, sampler.simplex_faces().rows(), 3) = sampler.simplex_faces().array() + point_index;

				face_index += sampler.simplex_faces().rows();

				vis_pts.block(point_index, 0, mapped.rows(), mapped.cols()) = mapped;
				point_index += mapped.rows();
			}
			else if (state.mesh->is_cube(i))
			{
				bs.eval_geom_mapping(sampler.cube_points(), mapped);
				vis_faces.block(face_index, 0, sampler.cube_faces().rows(), 3) = sampler.cube_faces().array() + point_index;
				face_index += sampler.cube_faces().rows();

				vis_pts.block(point_index, 0, mapped.rows(), mapped.cols()) = mapped;
				point_index += mapped.rows();
			}
			else
			{
				bs.eval_geom_mapping(vis_pts_poly[i], mapped);
				vis_faces.block(face_index, 0, vis_faces_poly[i].rows(), 3) = vis_faces_poly[i].array() + point_index;

				face_index += vis_faces_poly[i].rows();

				vis_pts.block(point_index, 0, vis_pts_poly[i].rows(), vis_pts_poly[i].cols()) = mapped;
				point_index += mapped.rows();
			}
		}

		assert(point_index == vis_pts.rows());
		assert(face_index == vis_faces.rows());

		if (state.mesh->is_volume())
		{
			// reverse all faces
			for (long i = 0; i < vis_faces.rows(); ++i)
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
			for (long i = 0; i < vis_faces.rows(); ++i)
			{
				const int v0 = vis_faces(i, 0);
				const int v1 = vis_faces(i, 1);
				const int v2 = vis_faces(i, 2);

				mmat.row(0) = vis_pts.row(v2) - vis_pts.row(v0);
				mmat.row(1) = vis_pts.row(v1) - vis_pts.row(v0);

				if (mmat.determinant() > 0)
				{
					int tmpc = vis_faces(i, 2);
					vis_faces(i, 2) = vis_faces(i, 1);
					vis_faces(i, 1) = tmpc;
				}
			}
		}

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		if (skip_visualization)
			return;

		clear();
		visible_visualizations.setConstant(false);
		visible_visualizations(Visualizations::VisMesh) = true;
		viewer.selected_data_index = Visualizations::VisMesh;

		show_vis_mesh();
	}

	void UIState::load_mesh()
	{
		if (!state.has_mesh() && febio_file.empty())
		{
			viewer.open_dialog_load_mesh();
		}
		available_visualizations.setConstant(false);

		element_ranges.clear();
		vis_element_ranges.clear();

		vis_faces_poly.clear();
		vis_pts_poly.clear();
		available_visualizations.setConstant(false);
		vis_flags.clear();
		vis_flags.resize(Visualizations::TotalVisualizations);

		if (febio_file.empty())
			state.load_mesh();
		else
			state.load_febio(febio_file, in_args);
		state.compute_mesh_stats();
		state.mesh->triangulate_faces(tri_faces, tri_pts, element_ranges);
		state.mesh->compute_element_barycenters(normalized_barycenter);

		for (long i = 0; i < normalized_barycenter.cols(); ++i)
		{
			normalized_barycenter.col(i) = MatrixXd(normalized_barycenter.col(i).array() - normalized_barycenter.col(i).minCoeff());
			normalized_barycenter.col(i) /= normalized_barycenter.col(i).maxCoeff();
		}

		if (skip_visualization)
			return;

		if (!state.mesh->is_volume())
		{
			light_enabled = false;

			viewer.core().set_rotation_type(igl::opengl::ViewerCore::RotationType::ROTATION_TYPE_NO_ROTATION);
		}
		else
		{
			viewer.core().set_rotation_type(igl::opengl::ViewerCore::RotationType::ROTATION_TYPE_TRACKBALL);
		}

		clear();
		visible_visualizations.setConstant(false);
		visible_visualizations(Visualizations::InputMesh) = true;
		visible_visualizations(Visualizations::Sidesets) = true;
		viewer.selected_data_index = Visualizations::InputMesh;
		show_mesh();
		show_sidesets();
		viewer.core().align_camera_center(tri_pts);
	}

	void UIState::build_basis()
	{
		if (!state.mesh)
		{
			return;
		}
		state.build_basis();
		available_visualizations.block(Visualizations::DiscrMesh, 0, Visualizations::TotalVisualizations - Visualizations::DiscrMesh, 1).setConstant(false);

		if (skip_visualization)
			return;
		clear();
		visible_visualizations.setConstant(false);
		visible_visualizations(Visualizations::DiscrMesh) = true;
		visible_visualizations(Visualizations::Nodes) = true;
		visible_visualizations(Visualizations::BNodes) = true;
		viewer.selected_data_index = Visualizations::DiscrMesh;
		show_mesh();
		show_sidesets();
		show_nodes();
	}

	void UIState::assemble_stiffness_mat()
	{
		if (!state.mesh)
		{
			return;
		}
		state.assemble_stiffness_mat();
	}

	void UIState::assemble_rhs()
	{
		if (!state.mesh)
		{
			return;
		}
		state.assemble_rhs();
	}

	void UIState::solve_problem()
	{
		if (!state.mesh)
		{
			return;
		}
		state.solve_problem();

		available_visualizations.block(Visualizations::Solution, 0, Visualizations::TotalVisualizations - Visualizations::Solution, 1).setConstant(false);

		if (skip_visualization)
			return;
		clear();
		visible_visualizations.setConstant(false);
		visible_visualizations(Visualizations::Solution) = true;
		viewer.selected_data_index = Visualizations::Solution;
		show_sol();
	}

	void UIState::compute_errors()
	{
		if (!state.mesh)
		{
			return;
		}
		state.compute_errors();

		if (!state.problem->has_exact_sol())
			return;

		available_visualizations.block(Visualizations::Error, 0, Visualizations::TotalVisualizations - Visualizations::Error, 1).setConstant(false);

		if (skip_visualization)
			return;
		clear();
		visible_visualizations.setConstant(false);
		visible_visualizations(Visualizations::Error) = true;
		viewer.selected_data_index = Visualizations::Error;
		show_error();
	}

	void UIState::update_slices()
	{
		for (int i = 0; i < visible_visualizations.size(); ++i)
		{
			if (!visible_visualizations(i))
				continue;

			available_visualizations(i) = false;
			switch (i)
			{
			case Visualizations::InputMesh:
				show_mesh();
				break;
			case Visualizations::VisMesh:
				show_vis_mesh();
				break;
			case Visualizations::Solution:
				show_sol();
				break;
			case Visualizations::Error:
				show_error();
				break;
			case Visualizations::VisBasis:
				show_basis();
				break;
			}
		}
	}

	void UIState::redraw()
	{
		for (int i = 0; i < Visualizations::TotalVisualizations; ++i)
		{
			hide_data(static_cast<Visualizations>(i));
		}

		show_mesh();
		show_sidesets();
		show_vis_mesh();
		show_nodes();
		show_sol();
		show_error();
		show_basis();
	}

	void UIState::launch(const std::string &log_file, int log_level, const bool is_quiet, const json &args, const std::string &febio_filei)
	{
		state.init_logger(log_file, log_level, is_quiet);
		state.init(args);

		// viewer.core().background_color.setOnes();
		febio_file = febio_filei;
		in_args = args;

		if (screenshot.empty())
		{
			viewer.core().is_animating = true;
			viewer.plugins.push_back(this);
			viewer.launch();
		}
		else
		{
			load_mesh();
			// offscreen_screenshot(viewer, screenshot);
		}
	}

	void UIState::sertialize(const std::string &name)
	{
	}

} // namespace polyfem

#include <GLFW/glfw3.h>

// #ifndef __APPLE__
// #  define GLEW_STATIC
// #  include <GL/glew.h>
// #endif

// #ifdef __APPLE__
// #   include <OpenGL/gl3.h>
// #   define __gl_h_ /* Prevent inclusion of the old gl.h */
// #else
// #   include <GL/gl.h>
// #endif

// namespace {

// 	static void my_glfw_error_callback(int error, const char* description)
// 	{
// 		fputs(description, stderr);
// 		fputs("\n", stderr);
// 	}

// }

// int offscreen_screenshot(igl::opengl::glfw::Viewer &viewer, const std::string &path) {
// 	glfwSetErrorCallback(my_glfw_error_callback);
// 	if (!glfwInit()) {
// 		std::cout << "init failure" << std::endl;
// 		return EXIT_FAILURE;
// 	}

// 	// glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_OSMESA_CONTEXT_API);
// 	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

// 	glfwWindowHint(GLFW_SAMPLES, 8);
// 	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
// 	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

//     #ifdef __APPLE__
// 	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
// 	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//     #endif

// 	printf("create window\n");
// 	GLFWwindow* offscreen_context = glfwCreateWindow(640, 480, "", NULL, NULL);
// 	printf("create context\n");
// 	glfwMakeContextCurrent(offscreen_context);
//     #ifndef __APPLE__
// 	glewExperimental = true;
// 	GLenum err = glewInit();
// 	if(GLEW_OK != err)
// 	{
//         /* Problem: glewInit failed, something is seriously wrong. */
// 		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
// 	}
//       glGetError(); // pull and savely ignonre unhandled errors like GL_INVALID_ENUM
//       fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
//     #endif
//       viewer.data().meshgl.init();
//       viewer.core().align_camera_center(viewer.data().V, viewer.data().F);
//       viewer.init();

//       Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(6400, 4000);
//       Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(6400, 4000);
//       Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(6400, 4000);
//       Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(6400, 4000);

//     // Draw the scene in the buffers
//       viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
//       A.setConstant(255);

//     // Save it to a PNG
//       igl::png::writePNG(R,G,B,A, path);

//       return 0;
//   }
