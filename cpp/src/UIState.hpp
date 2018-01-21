#ifndef UI_UISTATE_HPP
#define UI_UISTATE_HPP

#include "State.hpp"

#include <igl/colormap.h>
#include <igl/viewer/Viewer.h>
#include <map>


namespace poly_fem
{
	class UIState
	{
	private:
		UIState();

		enum Visualizing
		{
			InputMesh,
			VisMesh,
			Solution,
			Rhs,
			Error,
			VisBasis
		};

	public:
		static UIState &ui_state();

		void init(const std::string &mesh_path, const int n_refs, const int problem_num_);

		void sertialize(const std::string &name);

		bool skip_visualization = false;
		int vis_basis = 0;

		igl::viewer::Viewer viewer;

		std::vector<int> element_ranges, vis_element_ranges;

		Eigen::MatrixXi tri_faces, vis_faces;
		Eigen::MatrixXd tri_pts, vis_pts;

		Eigen::MatrixXi local_vis_faces_tri, local_vis_faces_quad;
		Eigen::MatrixXd local_vis_pts_tri, local_vis_pts_quad;

		std::map<int, Eigen::MatrixXi> vis_faces_poly;
		std::map<int, Eigen::MatrixXd> vis_pts_poly;

		igl::ColorMapType color_map = igl::COLOR_MAP_TYPE_VIRIDIS;

		int slice_coord = 0;
		int is_slicing = false;
		float slice_position = 1;

		bool ambient_occlusion = false;
		bool light_enabled = true;

		Eigen::MatrixXd normalized_barycenter;
		Eigen::VectorXd ambient_occlusion_mat;

		std::string selected_elements;

		State &state;
	private:
		Visualizing current_visualization = Visualizing::InputMesh;

		bool is_quad(const ElementBases &bs) const;
		bool is_tri(const ElementBases &bs) const;

		void plot_function(const Eigen::MatrixXd &fun, double min=0, double max=-1);
		void interpolate_function(const Eigen::MatrixXd &fun, Eigen::MatrixXd &result);

		long clip_elements(const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tris, const std::vector<int> &ranges, std::vector<bool> &valid_elements, const bool map_edges);
		long show_clipped_elements(const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tris, const std::vector<int> &ranges, const std::vector<bool> &valid_elements, const bool map_edges, const bool recenter = false);
		void color_mesh(const int n_tris, const std::vector<bool> &valid_elements);
		void plot_selection_and_index(const bool recenter = false);
		void get_plot_edges(const Mesh &mesh, const std::vector< ElementBases > &bases, const int n_samples, const std::vector<bool> &valid_elements, Eigen::MatrixXd &pp0, Eigen::MatrixXd &pp1);
	};

}
#endif //UI_UISTATE_HPP

