#ifndef UI_UISTATE_HPP
#define UI_UISTATE_HPP

#include "State.hpp"

#include <igl/viewer/Viewer.h>
#include <map>


namespace poly_fem
{
	class UIState
	{
	private:
		UIState();
	public:
		static UIState &ui_state();

		void init(const std::string &mesh_path, const int n_refs, const int problem_num_);

		void sertialize(const std::string &name);

		bool skip_visualization = false;
		int vis_basis = 0;

		igl::viewer::Viewer viewer;

		Eigen::MatrixXi tri_faces, vis_faces;
		Eigen::MatrixXd tri_pts, vis_pts;

		Eigen::MatrixXi local_vis_faces_tri, local_vis_faces_quad;
		Eigen::MatrixXd local_vis_pts_tri, local_vis_pts_quad;

		std::map<int, Eigen::MatrixXi> vis_faces_poly;
		std::map<int, Eigen::MatrixXd> vis_pts_poly;

		State &state;
	private:
		void plot_function(const Eigen::MatrixXd &fun, double min=0, double max=-1);
		void interpolate_function(const Eigen::MatrixXd &fun, Eigen::MatrixXd &result);
	};

}
#endif //UI_UISTATE_HPP

