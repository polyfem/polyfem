////////////////////////////////////////////////////////////////////////////////
#include "UIState.hpp"
#include "LinearSolver.hpp"
#include "StringUtils.hpp"
#include "Mesh3D.hpp"

#include <igl/png/writePNG.h>
#include <imgui/imgui.h>
#include <tinyfiledialogs.h>
#include <algorithm>
////////////////////////////////////////////////////////////////////////////////

namespace {

	namespace FileDialog {

// -----------------------------------------------------------------------------

		std::string openFileName(const std::string &defaultPath,
			const std::vector<std::string> &filters, const std::string &desc)
		{
			int n = static_cast<int>(filters.size());
			std::vector<char const *> filterPatterns(n);
			for (int i = 0; i < n; ++i) {
				filterPatterns[i] = filters[i].c_str();
			}
			char const * select = tinyfd_openFileDialog("Open File",
				defaultPath.c_str(), n, filterPatterns.data(), desc.c_str(), 0);
			if (select == nullptr) {
				return "";
			} else {
				return std::string(select);
			}
		}

// -----------------------------------------------------------------------------

		std::string saveFileName(const std::string &defaultPath,
			const std::vector<std::string> &filters, const std::string &desc)
		{
			int n = static_cast<int>(filters.size());
			std::vector<char const *> filterPatterns(n);
			for (int i = 0; i < n; ++i) {
				filterPatterns[i] = filters[i].c_str();
			}
			char const * select = tinyfd_saveFileDialog("Save File",
				defaultPath.c_str(), n, filterPatterns.data(), desc.c_str());
			if (select == nullptr) {
				return "";
			} else {
				return std::string(select);
			}
		}

} // namespace FileDialog

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

namespace ImGui {

	static auto vector_getter = [](void* vec, int idx, const char** out_text) {
		auto& vector = *static_cast<std::vector<std::string>*>(vec);
		if (idx < 0 || idx >= static_cast<int>(vector.size())) { return false; }
		*out_text = vector.at(idx).c_str();
		return true;
	};

	bool Combo(const char* label, int* idx, std::vector<std::string>& values) {
		if (values.empty()) { return false; }
		return Combo(label, idx, vector_getter,
			static_cast<void*>(&values), values.size());
	}

	bool ListBox(const char* label, int* idx, std::vector<std::string>& values) {
		if (values.empty()) { return false; }
		return ListBox(label, idx, vector_getter,
			static_cast<void*>(&values), values.size());
	}

}

////////////////////////////////////////////////////////////////////////////////

// Draw menu
void poly_fem::UIState::draw_menu() {
	// Text labels
	draw_labels_menu();

	// Viewer settings
	float viewer_menu_width = 200.f * m_HidpiScaling / m_PixelRatio;

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
	ImGui::SetNextWindowPos(ImVec2(viewer_menu_width+polyfem_menu_width, 0.0f), ImGuiSetCond_Always);
	ImGui::SetNextWindowSize(ImVec2(debug_menu_width, 0.0f), ImGuiSetCond_Always);
	ImGui::SetNextWindowSizeConstraints(ImVec2(debug_menu_width, -1.0f), ImVec2(debug_menu_width, -1.0f));
	bool _debug_menu_visible = true;
	ImGui::Begin(
		"Debug", &_debug_menu_visible,
		ImGuiWindowFlags_NoSavedSettings
		| ImGuiWindowFlags_AlwaysAutoResize
		| ImGuiWindowFlags_NoMove
		);
	draw_debug();
	ImGui::End();

	// Elasticity BC
	float elasticity_menu_width = 0.8 * viewer_menu_width;
	auto canvas = ImGui::GetIO().DisplaySize;
	ImGui::SetNextWindowPos(ImVec2(canvas.x - elasticity_menu_width, 0.0f), ImGuiSetCond_Always);
	ImGui::SetNextWindowSize(ImVec2(elasticity_menu_width, 0.0f), ImGuiSetCond_FirstUseEver);
	ImGui::SetNextWindowSizeConstraints(ImVec2(elasticity_menu_width, -1.0f), ImVec2(elasticity_menu_width, -1.0f));
	bool _elasticity_menu_visible = true;
	ImGui::Begin(
		"Elasticity BC", &_elasticity_menu_visible,
		ImGuiWindowFlags_NoSavedSettings
		| ImGuiWindowFlags_AlwaysAutoResize
		| ImGuiWindowFlags_NoMove
		);
	draw_elasticity_bc();
	ImGui::End();
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::UIState::draw_settings() {
	ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.50f);
	ImGui::InputInt("quad", &state.quadrature_order);
	ImGui::InputInt("discr", &state.discr_order);
	ImGui::InputInt("b", &state.n_boundary_samples);

	ImGui::InputFloat("lambda", &state.lambda, 0, 0, 3);
	ImGui::InputFloat("mu", &state.mu, 0, 0, 3);

	ImGui::InputInt("refs", &state.n_refs);
	ImGui::InputFloat("refinenemt t", &state.refinenemt_location, 0, 0, 2);
	ImGui::PopItemWidth();

	ImGui::Spacing();
	if (ImGui::Button("Browse...")) {
		std::string path = FileDialog::openFileName("./.*",
			{"*.HYBRID", "*.obj"} , "General polyhedral mesh");

		if (!path.empty()) {
			state.mesh_path = path;
			load_mesh();
		}

	}
	ImGui::SameLine();
	ImGui::Text("%s", state.mesh_path.c_str());
	ImGui::Spacing();

	ImGui::Checkbox("spline basis", &state.use_splines);
	ImGui::Checkbox("fit nodes", &state.fit_nodes);

	ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.50f);
	// Colormap type
	static int color_map_num = static_cast<int>(color_map);
	if (ImGui::Combo("Colormap", &color_map_num, "inferno\0jet\0magma\0parula\0plasma\0viridis\0\0")) {
		color_map = static_cast<igl::ColorMapType>(color_map_num);
	}

	// Problem type
	static std::string problem_name = state.problem->name();
	// static const char *problem_labels = "Linear\0Quadratic\0Franke\0Elastic\0Zero BC\0Franke3D\0ElasticExact\0\0";
	static const auto problem_names = poly_fem::ProblemFactory::factory().get_problem_names();
	if(ImGui::BeginCombo("Problem", problem_name.c_str()))
	{
		for(auto p_name : problem_names)
		{
			bool is_selected = problem_name == p_name;

			if (ImGui::Selectable(p_name.c_str(), is_selected))
				problem_name = p_name;
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

	static int formulation_num = static_cast<int>(state.elastic_formulation);
	static const char *formulation_labels = "Linear\0HookeLinear\0SaintVenant\0\0";
	if (ImGui::Combo("Form", &formulation_num, formulation_labels)) {
		state.elastic_formulation = static_cast<ElasticFormulation>(formulation_num);
	}

	// Solver type
	auto solvers = LinearSolver::availableSolvers();
	if (state.solver_type.empty()) {
		state.solver_type = LinearSolver::defaultSolver();
	}
	static int solver_num = std::distance(solvers.begin(), std::find(solvers.begin(), solvers.end(), state.solver_type));
	if (ImGui::Combo("Solver", &solver_num, solvers)) {
		state.solver_type = solvers[solver_num];
	}

	// Preconditioner
	auto precond = LinearSolver::availablePrecond();
	if (state.precond_type.empty()) {
		state.precond_type = LinearSolver::defaultPrecond();
	}
	static int precond_num =  std::distance(precond.begin(),
		std::find(precond.begin(), precond.end(), state.precond_type));
	if (ImGui::Combo("Precond", &precond_num, precond)) {
		state.precond_type = precond[precond_num];
	}
	ImGui::PopItemWidth();
	ImGui::Checkbox("skip visualization", &skip_visualization);

	// Actions
	if (ImGui::CollapsingHeader("Actions", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Load mesh", ImVec2(-1, 0))) { load_mesh(); }
		if (ImGui::Button("Build  basis", ImVec2(-1, 0))) { build_basis(); }
		if (ImGui::Button("Compute poly bases", ImVec2(-1, 0))) { build_polygonal_basis(); }
		if (ImGui::Button("Build vis mesh", ImVec2(-1, 0))) { build_vis_mesh(); }

		if (ImGui::Button("Assemble rhs", ImVec2(-1, 0))) { assemble_rhs(); }
		if (ImGui::Button("Assemble stiffness", ImVec2(-1, 0))) { assemble_stiffness_mat(); }
		if (ImGui::Button("Solve", ImVec2(-1, 0))) { solve_problem(); }
		if (ImGui::Button("Compute errors", ImVec2(-1, 0))) { compute_errors(); }

		ImGui::Spacing();
		if (ImGui::Button("Run all", ImVec2(-1, 0))) {
			load_mesh();
			build_basis();
			build_polygonal_basis();

			if(!skip_visualization)
				build_vis_mesh();

			assemble_rhs();
			assemble_stiffness_mat();
			solve_problem();
			compute_errors();
			state.save_json(std::cout);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::UIState::draw_debug() {
	if (ImGui::Button("Clear", ImVec2(-1, 0))) { clear(); }
	if (ImGui::Button("Show mesh", ImVec2(-1, 0))) { show_mesh(); }
	if (ImGui::Button("Show vis mesh", ImVec2(-1, 0))) { show_vis_mesh(); }
	if (ImGui::Button("Show nodes", ImVec2(-1, 0))) { show_nodes(); }
	// if (ImGui::Button("Show quadrature", ImVec2(-1, 0))) { show_quadrature(); }
	if (ImGui::Button("Show rhs", ImVec2(-1, 0))) { show_rhs(); }
	if (ImGui::Button("Show sol", ImVec2(-1, 0))) { show_sol(); }
	if (ImGui::Button("Show error", ImVec2(-1, 0))) { show_error(); }

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
			std::transform(v.begin(), v.end(), selected_elements.begin(),
				[](const std::string &s) { return std::stoi(s); });
		}
		ImGui::PopItemWidth();
		if (ImGui::Button("Show##Selected", ImVec2(-1, 0))) {
			plot_selection_and_index(true);
		}
		if (ImGui::Button("Switch vertex", ImVec2(-1, 0))) {
			plot_selection_and_index();
		}
		if (ImGui::Button("Switch edge", ImVec2(-1, 0))) {
			plot_selection_and_index();
		}
		if (ImGui::Button("Switch face", ImVec2(-1, 0))) {
			plot_selection_and_index();
		}
		if (ImGui::Button("Switch element", ImVec2(-1, 0))) {
			plot_selection_and_index();
		}
		if (ImGui::Button("Save selection", ImVec2(-1, 0))) {
			if(state.mesh->is_volume()) {
				dynamic_cast<Mesh3D *>(state.mesh.get())->save(selected_elements, 2, "mesh.HYBRID");
			}
		}
	}

}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::UIState::draw_screenshot() {
	if (ImGui::Button("Save Screenshot", ImVec2(-1, 0))) {
		// Allocate temporary buffers
		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(6400, 4000);
		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(6400, 4000);
		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(6400, 4000);
		Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(6400, 4000);

		// Draw the scene in the buffers
		viewer.core.draw_buffer(viewer.data,viewer.opengl,false,R,G,B,A);
		A.setConstant(255);

		// Save it to a PNG
		std::string path = (screenshot.empty() ? "out.png" : screenshot);
		igl::png::writePNG(R,G,B,A,path);
	}
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::UIState::draw_elasticity_bc() {
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

