#ifndef UI_UISTATE_HPP
#define UI_UISTATE_HPP

#include <polyfem/State.hpp>

#include <igl/colormap.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <map>


namespace polyfem
{
	class UIState : public igl::opengl::glfw::imgui::ImGuiMenu
	{
	private:
		UIState();

		enum Visualizations
		{
			InputMesh = 0,
			DiscrMesh,
			Nodes,
			PNodes,
			BNodes,
			VisMesh,
			Solution,
			Error,
			ErrorGrad,
			VisBasis,

			NavigationIndex,

			Debug,

			ElementId,
			VertexId,
			NodesId,

			TotalVisualizations
		};

		const std::string visualizations_texts[Visualizations::TotalVisualizations] = { "InputMesh", "Diretization", "Nodes", "PNodes", "BNodes", "VisMesh", "Solution", "Error", "ErrorGrad", "VisBasis",  "NavigationIndex", "Debug", "ElementId", "VertexId", "NodesId" };


	public:
		static UIState &ui_state();

		void launch(const std::string &log_file, int log_level, const bool is_quiet, const json &args);

		void sertialize(const std::string &name);


		bool skip_visualization = false;
		bool show_funs_in_3d = false;
		int vis_basis = 0;

		igl::opengl::glfw::Viewer viewer;

		std::vector<int> element_ranges, vis_element_ranges;

		Eigen::MatrixXi tri_faces, vis_faces;
		Eigen::MatrixXd tri_pts, vis_pts;


		std::map<int, Eigen::MatrixXi> vis_faces_poly;
		std::map<int, Eigen::MatrixXd> vis_pts_poly;

		igl::ColorMapType color_map = igl::COLOR_MAP_TYPE_VIRIDIS;
		double min_val = 0, max_val = 0;

		int slice_coord = 0;
		bool is_slicing = false;
		bool show_grad_error = false;
		float slice_position = 1;

		bool ambient_occlusion = false;
		bool light_enabled = true;
		bool show_isolines = true;

		bool show_element_id = false;
		bool show_vertex_id = false;
		bool show_node_id = false;
		// bool color_using_discr_order = false;

		std::string screenshot = "";

		Eigen::MatrixXd normalized_barycenter;
		Eigen::VectorXd ambient_occlusion_mat;

		std::vector<int> selected_elements;

		State state;

		igl::opengl::ViewerData &debug_data()
		{
			reset_flags(Visualizations::Debug, false);
			hide_data(Visualizations::Debug);
			show_data(Visualizations::Debug);

			return data(Visualizations::Debug);
		}

		bool load(std::string mesh_file_name_string) override
		{
			state.args["mesh"] = mesh_file_name_string;
			load_mesh();

			return true;
		}

		bool save(std::string mesh_file_name_string) override
		{
			state.mesh->save(mesh_file_name_string);

			return true;
		}

	protected:
		Navigation3D::Index current_3d_index;
		Navigation::Index 	current_2d_index;

		igl::opengl::ViewerData &data(const Visualizations &layer);
		Eigen::Matrix<bool, Eigen::Dynamic, 1> available_visualizations;
		Eigen::Matrix<bool, Eigen::Dynamic, 1> visible_visualizations;

		std::vector<std::vector<bool>> vis_flags;
		void reset_flags(const Visualizations &layer, bool clear = true);
		void hide_data(const Visualizations &layer);
		void show_data(const Visualizations &layer);

		void clear();
		void redraw();

		// Draw menu
		virtual void draw_menu() override;
		void draw_settings();
		void draw_debug();
		void draw_screenshot();
		void draw_elasticity_bc();

		void show_mesh();
		void show_vis_mesh();
		void show_nodes();
		void show_sol();
		void show_error();
		void show_basis();
		void show_linear_reproduction();
		void show_quadratic_reproduction();
		void build_vis_mesh();
		void load_mesh();
		void build_basis();
		void build_polygonal_basis();
		void assemble_stiffness_mat();
		void assemble_rhs();
		void solve_problem();
		void compute_errors();
		void update_slices();

		bool is_quad(const ElementBases &bs) const;
		bool is_tri(const ElementBases &bs) const;

		void plot_function(const Eigen::MatrixXd &fun, const Visualizations &layer, double min=0, double max=-1);
		void interpolate_function(const Eigen::MatrixXd &fun, Eigen::MatrixXd &result);
		void interpolate_grad_function(const Eigen::MatrixXd &fun, Eigen::MatrixXd &result);

		long clip_elements(const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tris, const std::vector<int> &ranges, std::vector<bool> &valid_elements, const bool map_edges, const Visualizations &layer);
		long show_clipped_elements(const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tris, const std::vector<int> &ranges, const std::vector<bool> &valid_elements, const bool map_edges, const Visualizations &layer, const bool recenter = false);
		void color_mesh(const int n_tris, const std::vector<bool> &valid_elements, const Visualizations &layer);
		void plot_selection_and_index(const bool recenter = false);
		void get_plot_edges(const Mesh &mesh, const std::vector< ElementBases > &bases, const std::vector< ElementBases > &gbases, const int n_samples, const std::vector<bool> &valid_elements, const Visualizations &layer, Eigen::MatrixXd &pp0, Eigen::MatrixXd &pp1);
	};

}
#endif //UI_UISTATE_HPP

