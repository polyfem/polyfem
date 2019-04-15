////////////////////////////////////////////////////////////////////////////////
#include "UIState.hpp"
#include <polyfem/LinearSolver.hpp>
#include <polyfem/StringUtils.hpp>
#include <polyfem/Mesh3D.hpp>
#include <polyfem/AssemblerUtils.hpp>

#include <igl/colormap.h>
// #include <igl/png/writePNG.h>
#include <imgui/imgui.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/gl.h>
#include <imgui/imgui_internal.h>
// #include <tinyfiledialogs.h>
#include <algorithm>

#include <GLFW/glfw3.h>
////////////////////////////////////////////////////////////////////////////////

namespace {
	void push_disabled()
	{
		ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
		ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
	}

	void pop_disabled()
	{
		ImGui::PopItemFlag();
		ImGui::PopStyleVar();
	}


	namespace FileDialog {

// -----------------------------------------------------------------------------

		// std::string openFileName(const std::string &defaultPath,
		// 	const std::vector<std::string> &filters, const std::string &desc)
		// {
		// 	int n = static_cast<int>(filters.size());
		// 	std::vector<char const *> filterPatterns(n);
		// 	for (int i = 0; i < n; ++i) {
		// 		filterPatterns[i] = filters[i].c_str();
		// 	}
		// 	char const * select = tinyfd_openFileDialog("Open File",
		// 		defaultPath.c_str(), n, filterPatterns.data(), desc.c_str(), 0);
		// 	if (select == nullptr) {
		// 		return "";
		// 	} else {
		// 		return std::string(select);
		// 	}
		// }

// -----------------------------------------------------------------------------

		// std::string saveFileName(const std::string &defaultPath,
		// 	const std::vector<std::string> &filters, const std::string &desc)
		// {
		// 	int n = static_cast<int>(filters.size());
		// 	std::vector<char const *> filterPatterns(n);
		// 	for (int i = 0; i < n; ++i) {
		// 		filterPatterns[i] = filters[i].c_str();
		// 	}
		// 	char const * select = tinyfd_saveFileDialog("Save File",
		// 		defaultPath.c_str(), n, filterPatterns.data(), desc.c_str());
		// 	if (select == nullptr) {
		// 		return "";
		// 	} else {
		// 		return std::string(select);
		// 	}
		// }

} // namespace FileDialog

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

// Draw menu
void polyfem::UIState::draw_menu() {

	// Viewer settings
	float viewer_menu_width = 180.f * hidpi_scaling() / pixel_ratio();

	ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiSetCond_Always);
	ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiSetCond_Always);
	ImGui::SetNextWindowSizeConstraints(ImVec2(viewer_menu_width, -1.0f), ImVec2(viewer_menu_width, -1.0f));
	bool _viewer_menu_visible = true;
	ImGui::Begin(
		"Viewer", &_viewer_menu_visible,
		ImGuiWindowFlags_NoSavedSettings
		| ImGuiWindowFlags_AlwaysAutoResize
		| ImGuiWindowFlags_NoMove
		);
	draw_viewer_menu();
	draw_labels_window();
	draw_screenshot();
	ImGui::End();

	// Polyfem menu
	float polyfem_menu_width = 0.8 * viewer_menu_width;
	ImGui::SetNextWindowPos(ImVec2(viewer_menu_width, 0.0f), ImGuiSetCond_Always);
	ImGui::SetNextWindowSize(ImVec2(polyfem_menu_width, 0.0f), ImGuiSetCond_Always);
	ImGui::SetNextWindowSizeConstraints(ImVec2(polyfem_menu_width, -1.0f), ImVec2(polyfem_menu_width, -1.0f));
	bool _polyfem_menu_visible = true;
	ImGui::Begin(
		"Settings", &_polyfem_menu_visible,
		ImGuiWindowFlags_NoSavedSettings
		| ImGuiWindowFlags_AlwaysAutoResize
		| ImGuiWindowFlags_NoMove
		);
	draw_settings();
	ImGui::End();

	// Debug menu
	float debug_menu_width = 0.8 * viewer_menu_width;
	// ImGui::SetNextWindowPos(ImVec2(viewer_menu_width+polyfem_menu_width, 0.0f), ImGuiSetCond_Always);
	auto canvas = ImGui::GetIO().DisplaySize;
	ImGui::SetNextWindowPos(ImVec2(canvas.x - debug_menu_width, 0.0f), ImGuiSetCond_Always);
	ImGui::SetNextWindowSize(ImVec2(debug_menu_width, 0.0f), ImGuiSetCond_Always);
	ImGui::SetNextWindowSizeConstraints(ImVec2(debug_menu_width, -1.0f), ImVec2(debug_menu_width, -1.0f));
	bool _debug_menu_visible = true;
	ImGui::Begin(
		"View", &_debug_menu_visible,
		ImGuiWindowFlags_NoSavedSettings
		| ImGuiWindowFlags_AlwaysAutoResize
		| ImGuiWindowFlags_NoMove
		);
	draw_debug();
	ImGui::End();

	// // Elasticity BC
	// float elasticity_menu_width = 0.8 * viewer_menu_width;
	// auto canvas = ImGui::GetIO().DisplaySize;
	// ImGui::SetNextWindowPos(ImVec2(canvas.x - elasticity_menu_width, 0.0f), ImGuiSetCond_Always);
	// ImGui::SetNextWindowSize(ImVec2(elasticity_menu_width, 0.0f), ImGuiSetCond_FirstUseEver);
	// ImGui::SetNextWindowSizeConstraints(ImVec2(elasticity_menu_width, -1.0f), ImVec2(elasticity_menu_width, -1.0f));
	// bool _elasticity_menu_visible = true;
	// ImGui::Begin(
	// 	"Elasticity BC", &_elasticity_menu_visible,
	// 	ImGuiWindowFlags_NoSavedSettings
	// 	| ImGuiWindowFlags_AlwaysAutoResize
	// 	| ImGuiWindowFlags_NoMove
	// 	);
	// draw_elasticity_bc();
	// ImGui::End();
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::UIState::draw_settings() {
	ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.50f);
	// ImGui::InputInt("quad", &state.quadrature_order);
	int discr_order = state.args["discr_order"];
	ImGui::InputInt("discr", &discr_order);
	state.args["discr_order"] = discr_order;

	// ImGui::InputInt("b", &state.n_boundary_samples);

	//ImGui::InputFloat("lambda", &state.lambda, 0, 0, 3);
	//ImGui::InputFloat("mu", &state.mu, 0, 0, 3);
	int n_refs = state.args["n_refs"];
	ImGui::InputInt("refs", &n_refs);
	state.args["n_refs"] = n_refs;

	// ImGui::InputFloat("refinenemt t", &state.refinenemt_location, 0, 0, 2);
	ImGui::PopItemWidth();

	ImGui::Spacing();
	// if (ImGui::Button("Browse...")) {
	// 	std::string path = FileDialog::openFileName("./.*",
	// 		{"*.HYBRID","*.mesh","*.MESH", "*.obj"} , "General polyhedral mesh");

	// 	if (!path.empty()) {
	// 		load(path)
	// 	}

	// }
	// ImGui::SameLine();
	ImGui::Text("%s", state.mesh_path().c_str());
	ImGui::Spacing();

	bool use_splines = state.args["use_spline"];
	ImGui::Checkbox("spline basis", &use_splines);
	state.args["use_spline"] = use_splines;

	bool use_p_ref = state.args["use_p_ref"];
	ImGui::Checkbox("p ref", &use_p_ref);
	state.args["use_p_ref"] = use_p_ref;

	// bool fit_nodes = state.args["fit_nodes"];
	// ImGui::Checkbox("fit nodes", &fit_nodes);
	// state.args["fit_nodes"] = fit_nodes;

	ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.50f);
	ImGui::Separator();

	// Problem type
	static std::string problem_name = state.problem->name();
	// static const char *problem_labels = "Linear\0Quadratic\0Franke\0Elastic\0Zero BC\0Franke3D\0ElasticExact\0\0";
	static const auto problem_names = polyfem::ProblemFactory::factory().get_problem_names();
	if(ImGui::BeginCombo("Problem", problem_name.c_str()))
	{
		for(auto p_name : problem_names)
		{
			bool is_selected = problem_name == p_name;

			if (ImGui::Selectable(p_name.c_str(), is_selected)){
				problem_name = p_name;
				state.problem = ProblemFactory::factory().get_problem(problem_name);
				state.problem->set_parameters(state.args["problem_params"]);
			}
			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	// if (ImGui::Combo("Problem", &problem_num, problem_labels)) {
	// 	state.problem = Problem::get_problem(static_cast<ProblemType>(problem_num));
	// 	// if(state.problem->boundary_ids().empty()) {
	// 	// 	std::fill(dirichlet_bc.begin(), dirichlet_bc.end(), true);
	// 	// } else {
	// 	// 	std::fill(dirichlet_bc.begin(), dirichlet_bc.end(), false);
	// 	// }

	// 	// for (int i = 0; i < (int) state.problem->boundary_ids().size(); ++i) {
	// 	// 	dirichlet_bc[state.problem->boundary_ids()[i]-1] = true;
	// 	// }
	// }

	// ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
	// ImGui::BeginChild("Forms", ImVec2(ImGui::GetWindowContentRegionWidth(), 100), true);
	// ImGui::BeginChild("Forms", ImVec2(ImGui::GetWindowContentRegionWidth(), 100), true);
	ImGui::Separator();


	static const auto scalar_forms = polyfem::AssemblerUtils::instance().scalar_assemblers();

	const bool is_scalar = state.problem->is_scalar();

	if(is_scalar)
	{
		if(ImGui::BeginCombo("1D-Form", state.scalar_formulation().c_str()))
		{
			for(auto f : scalar_forms)
			{
				bool is_selected = state.scalar_formulation() == f;

				if (ImGui::Selectable(f.c_str(), is_selected))
					state.args["scalar_formulation"] = f;
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
	}

	else
	{
		static const auto tensor_forms = polyfem::AssemblerUtils::instance().tensor_assemblers();
		if(ImGui::BeginCombo("nD-Form", state.tensor_formulation().c_str()))
		{
			for(auto f : tensor_forms)
			{
				bool is_selected = state.tensor_formulation() == f;

				if (ImGui::Selectable(f.c_str(), is_selected))
					state.args["tensor_formulation"] = f;
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
	}


	// if(is_mixed)
	// {
	// 	static const auto mixed_forms = polyfem::AssemblerUtils::instance().mixed_assemblers();
	// 	if(ImGui::BeginCombo("mixed", state.mixed_formulation().c_str()))
	// 	{
	// 		for(auto f : mixed_forms)
	// 		{
	// 			bool is_selected = state.mixed_formulation() == f;

	// 			if (ImGui::Selectable(f.c_str(), is_selected))
	// 				state.args["mixed_formulation"] = f;
	// 			if (is_selected)
	// 				ImGui::SetItemDefaultFocus();
	// 		}
	// 		ImGui::EndCombo();
	// 	}
	// }
	ImGui::Separator();




	// Solver type
	auto solvers = LinearSolver::availableSolvers();
	static int solver_num = std::distance(solvers.begin(), std::find(solvers.begin(), solvers.end(), state.solver_type()));
	if (ImGui::Combo("Solver", &solver_num, solvers)) {
		state.args["solver_type"] = solvers[solver_num];
	}

	// Preconditioner
	auto precond = LinearSolver::availablePrecond();
	static int precond_num =  std::distance(precond.begin(), std::find(precond.begin(), precond.end(), state.precond_type()));
	if (ImGui::Combo("Precond", &precond_num, precond)) {
		state.args["precond_type"] = precond[precond_num];
	}


	ImGui::Separator();

	// Colormap type
	static int color_map_num = static_cast<int>(color_map);
	if (ImGui::Combo("Colormap", &color_map_num, "inferno\0jet\0magma\0parula\0plasma\0viridis\0\0")) {
		color_map = static_cast<igl::ColorMapType>(color_map_num);
	}

	ImGui::PopItemWidth();
	ImGui::Checkbox("skip visualization", &skip_visualization);

	// Actions
	if (ImGui::CollapsingHeader("Actions", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Load mesh", ImVec2(-1, 0))) { load_mesh(); }
		if (ImGui::Button("Build  basis", ImVec2(-1, 0))) { build_basis(); }
		// if (ImGui::Button("Compute poly bases", ImVec2(-1, 0))) {  }
		if (ImGui::Button("Build vis mesh", ImVec2(-1, 0))) { build_vis_mesh(); }

		if (ImGui::Button("Assemble rhs", ImVec2(-1, 0))) { assemble_rhs(); }
		if (ImGui::Button("Assemble stiffness", ImVec2(-1, 0))) { assemble_stiffness_mat(); }
		if (ImGui::Button("Solve", ImVec2(-1, 0))) { solve_problem(); }
		if (ImGui::Button("Compute errors", ImVec2(-1, 0))) { compute_errors(); }

		ImGui::Spacing();
		if (ImGui::Button("Run all", ImVec2(-1, 0))) {
			load_mesh();
			build_basis();

			if(!skip_visualization)
				build_vis_mesh();

			assemble_rhs();
			assemble_stiffness_mat();
			solve_problem();
			compute_errors();
			state.save_json(std::cout);
		}
	}

	ImGui::Separator();

	ImGui::LabelText("L2 error", "L2\t%g", state.l2_err);
	ImGui::LabelText("Lp error", "Lp\t%g", state.lp_err);
	ImGui::LabelText("H1 error", "H1\t%g", state.h1_err);

	ImGui::Separator();

	static GLuint color_bar_texture = 0;
	static const int width  = ImGui::GetWindowWidth();
	static const int height = 20;
	if(color_bar_texture == 0)
	{
		Eigen::Matrix<unsigned char, Eigen::Dynamic, 4, Eigen::RowMajor> cmap(width*height, 4);


		Eigen::MatrixXd t = Eigen::VectorXd::LinSpaced(width, 0, width);
		Eigen::MatrixXd col;
		igl::colormap(color_map, t, true, col);
		assert(col.rows() == width);
		for(int i = 0; i < width; ++i)
		{
			for(int j = 0; j < height; ++j)
			{
				for(int c = 0; c < 3; ++c)
					cmap(j*width+i, c) = col(i, c)*255;
			}
		}
		cmap.col(3).setConstant(255);

		glGenTextures(1, &color_bar_texture);
		glBindTexture(GL_TEXTURE_2D, color_bar_texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, cmap.data());
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	ImGui::Image(reinterpret_cast<ImTextureID>(color_bar_texture), ImVec2(width, height));

	if(fabs(min_val) <= 1e-20)
		ImGui::Text("0");
	else
	{
		const int min_power = floor(log10(fabs(min_val)));
		ImGui::Text("%ge%d", round(min_val * pow(10, -min_power)*100)/100., min_power);
	}

	if(fabs(max_val) <= 1e-20)
	{
		ImGui::SameLine(width-10);
		ImGui::Text("0");
	}
	else
	{
		ImGui::SameLine(width-45);
		const int max_power = floor(log10(fabs(max_val)));
		ImGui::Text("%ge%d", round(max_val * pow(10, -max_power)*100)/100., max_power);
	}

}

////////////////////////////////////////////////////////////////////////////////

void polyfem::UIState::draw_debug() {
	if (ImGui::Button("Clear", ImVec2(-1, 0))) { clear(); }

    // ImGui::Columns(1, "visualizations");
	for(int i = 0; i <= polyfem::UIState::Visualizations::Debug; ++i)
	{
		// if(ImGui::Selectable(visualizations_texts[i].c_str(), viewer.selected_data_index == i))
		// {
		// 	viewer.selected_data_index = i;
		// }
		// ImGui::NextColumn();
		if(ImGui::Checkbox(visualizations_texts[i].c_str(), &visible_visualizations(i))){
			redraw();
		}
		// ImGui::NextColumn();
	}
	// ImGui::Columns(1);

	// if (ImGui::Button("Show mesh", ImVec2(-1, 0))) { show_mesh(); }
	// if (ImGui::Button("Show vis mesh", ImVec2(-1, 0))) { show_vis_mesh(); }
	// if (ImGui::Button("Show nodes", ImVec2(-1, 0))) { show_nodes(); }
	// if (ImGui::Button("Show quadrature", ImVec2(-1, 0))) { show_quadrature(); }
	// if (ImGui::Button("Show rhs", ImVec2(-1, 0))) { show_rhs(); }
	// if (ImGui::Button("Show sol", ImVec2(-1, 0))) { show_sol(); }
	// if (ImGui::Button("Show error", ImVec2(-1, 0))) { show_error(); }

	// ImGui::Checkbox("Show isolines", &show_isolines);

	// int selection_err = show_grad_error ? 1 : 0;
	// static const char *error_type = "function\0gradient\0\0";
	// if (ImGui::Combo("Error", &selection_err, error_type)) {
	// 	show_grad_error = selection_err == 1;
	// }

	ImGui::Separator();

	if (ImGui::Button("Show linear r", ImVec2(-1, 0))) { show_linear_reproduction(); }
	if (ImGui::Button("Show quadra r", ImVec2(-1, 0))) { show_quadratic_reproduction(); }

	ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.50f);
	ImGui::InputInt("Basis Num", &vis_basis);
	ImGui::PopItemWidth();
	if (ImGui::Button("Show basis", ImVec2(-1, 0))) { show_basis(); }

	// Slicing
	if (ImGui::CollapsingHeader("Slicing", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.50f);
		if (ImGui::Combo("Axis", &slice_coord, "X\0Y\0Z\0\0")) {
			if (is_slicing) { update_slices(); }
		}
		if (ImGui::InputFloat("Coord", &slice_position, 0.1f, 1.f, 3)) {
			if (is_slicing) { update_slices(); }
		}
		if (ImGui::Checkbox("Enable##Slicing", &is_slicing)) {
			update_slices();
		}
		ImGui::PopItemWidth();
	}

	if (ImGui::CollapsingHeader("Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.50f);
		static char buf[1024];
		if (ImGui::InputText("Element Ids", buf, 1024)) {
			auto v = StringUtils::split(buf, ",");
			selected_elements.resize(v.size());
			std::transform(v.begin(), v.end(), selected_elements.begin(), [](const std::string &s) { return std::stoi(s); });
		}
		ImGui::PopItemWidth();
		if (ImGui::Button("Show##Selected", ImVec2(-1, 0))) {

			if(state.mesh->is_volume())
				current_3d_index = dynamic_cast<Mesh3D *>(state.mesh.get())->get_index_from_element(selected_elements.front());
			else
				current_2d_index = dynamic_cast<Mesh2D *>(state.mesh.get())->get_index_from_face(selected_elements.front());

			plot_selection_and_index(true);
		}
		if (ImGui::Button("Switch vertex", ImVec2(-1, 0))) {
			if(state.mesh->is_volume())
				current_3d_index = dynamic_cast<Mesh3D *>(state.mesh.get())->switch_vertex(current_3d_index);
			else
				current_2d_index = dynamic_cast<Mesh2D *>(state.mesh.get())->switch_vertex(current_2d_index);

			plot_selection_and_index();
		}
		if (ImGui::Button("Switch edge", ImVec2(-1, 0))) {
			if(state.mesh->is_volume())
				current_3d_index = dynamic_cast<Mesh3D *>(state.mesh.get())->switch_edge(current_3d_index);
			else
				current_2d_index = dynamic_cast<Mesh2D *>(state.mesh.get())->switch_edge(current_2d_index);

			plot_selection_and_index();
		}
		if (ImGui::Button("Switch face", ImVec2(-1, 0))) {
			if(state.mesh->is_volume())
				current_3d_index = dynamic_cast<Mesh3D *>(state.mesh.get())->switch_face(current_3d_index);
			else{
				current_2d_index = dynamic_cast<Mesh2D *>(state.mesh.get())->switch_face(current_2d_index);
				selected_elements.push_back(current_2d_index.face);
			}

			plot_selection_and_index();
		}
		if(state.mesh && state.mesh->is_volume())
		{
			if (ImGui::Button("Switch element", ImVec2(-1, 0))) {
				current_3d_index = dynamic_cast<Mesh3D *>(state.mesh.get())->switch_element(current_3d_index);
				selected_elements.push_back(current_3d_index.element);

				plot_selection_and_index();
			}
		}
		if (ImGui::Button("Save selection", ImVec2(-1, 0))) {
			if(state.mesh->is_volume()) {
				dynamic_cast<Mesh3D *>(state.mesh.get())->save(selected_elements, 2, "mesh.HYBRID");
			}
		}
	}

}

////////////////////////////////////////////////////////////////////////////////

void polyfem::UIState::draw_screenshot() {
	if (ImGui::Button("Save Screenshot", ImVec2(-1, 0))) {
		// Allocate temporary buffers
		// Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(6400, 4000);
		// Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(6400, 4000);
		// Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(6400, 4000);
		// Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(6400, 4000);

		// // Draw the scene in the buffers
		// viewer.core.draw_buffer(viewer.data(),false,R,G,B,A);
		// A.setConstant(255);

		// // Save it to a PNG
		// std::string path = (screenshot.empty() ? "out.png" : screenshot);
		// igl::png::writePNG(R,G,B,A,path);
	}

	float w = ImGui::GetContentRegionAvailWidth();
	float p = ImGui::GetStyle().FramePadding.x;
	if (ImGui::Button("Save VTU", ImVec2((w-p)/2.f, 0))) {
		state.save_vtu("result.vtu");
	}
	ImGui::SameLine(0, p);
	if (ImGui::Button("Save Wire", ImVec2((w-p)/2.f, 0))) {
		state.save_wire("result.obj");
	}

	ImGui::Separator();

	if(ImGui::Checkbox("Show element ids", &show_element_id))
		show_mesh();

	if(ImGui::Checkbox("Show vertex ids", &show_vertex_id))
		show_mesh();

	if(ImGui::Checkbox("Show node ids", &show_node_id))
		show_nodes();

	// if(ImGui::Checkbox("Color discr order", &color_using_discr_order))
		// show_mesh();

	ImGui::Checkbox("Use 3D funs", &show_funs_in_3d);
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::UIState::draw_elasticity_bc() {
	// for(int local_id = 1; local_id <= 6; ++local_id) {
	// 	viewer_.ngui->addVariable<bool>("Dirichlet " + std::to_string(local_id),
	// 		[this,local_id](bool val) {
	// 		dirichlet_bc[local_id-1] = val;

	// 		auto &ids = state.problem->boundary_ids();
	// 		ids.clear();
	// 		for(int i=0; i < dirichlet_bc.size(); ++i)
	// 		{
	// 			if(dirichlet_bc[i])
	// 				ids.push_back(i+1);
	// 		}

	// 	},[this,local_id]() {
	// 		return dirichlet_bc[local_id-1];
	// 	});
	// }
}

