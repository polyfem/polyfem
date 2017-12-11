#ifndef UI_UISTATE_HPP
#define UI_UISTATE_HPP

#include "State.hpp"

#include <igl/viewer/Viewer.h>


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

		Eigen::MatrixXi tri_faces, local_vis_faces, vis_faces;
		Eigen::MatrixXd tri_pts, local_vis_pts, vis_pts;

	private:
		void plot_function(const Eigen::MatrixXd &fun, double min=0, double max=-1);
		State &state;
	};

}
#endif //UI_UISTATE_HPP

