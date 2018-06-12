#include <Mesh3D.hpp>
#include <CommandLine.hpp>

#include <Common.hpp>


#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/adjacency_list.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/per_face_normals.h>
#include <imgui/imgui.h>

#include <igl/unproject_onto_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <queue>


using namespace poly_fem;
using namespace Eigen;

RowVector3d color(int bc)
{
	RowVector3d col;

	switch(bc)
	{
		case 0: col << 1,1,1; break;
		case 1: col << 1,0,0; break;
		case 2: col << 0,1,0; break;
		case 3: col << 0,0,1; break;
		case 4: col << 1,1,0; break;
		case 5: col << 1,0,1; break;
		case 6: col << 0,1,1; break;
	}

	return col;
}

int main(int argc, const char **argv)
{
	#ifndef WIN32
	setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);
#endif

	GEO::initialize();

    // Import standard command line arguments, and custom ones
	GEO::CmdLine::import_arg_group("standard");
	GEO::CmdLine::import_arg_group("pre");
	GEO::CmdLine::import_arg_group("algo");


	igl::opengl::glfw::Viewer viewer;

	CommandLine command_line;
	std::string path = "";
	command_line.add_option("-mesh", path);

	std::string bc = "";
	command_line.add_option("-bc", bc);

	command_line.parse(argc, argv);

	Mesh3D mesh;
	mesh.load(path);
	MatrixXd V, p0, p1;
	MatrixXi F;

	std::vector<int> ranges;
	mesh.get_edges(p0, p1);
	V.resize(mesh.n_vertices() + mesh.n_faces(), 3);
	for(int i = 0; i < mesh.n_vertices(); ++i)
		V.row(i) = mesh.point(i);

	int v_index = mesh.n_vertices();

	F.resize(mesh.n_faces()*4, 3);
	VectorXi boundary_2_all(mesh.n_faces()*4);
	Matrix<std::vector<int>, Dynamic, 1> all_2_local(mesh.n_faces());

	int index = 0;
	for(int f = 0; f < mesh.n_faces(); ++f)
	{
		if(!mesh.is_boundary_face(f))
			continue;

		std::vector<int> &other_faces = all_2_local(f);

		const int n_f_v = mesh.n_face_vertices(f);
		if(n_f_v == 3)
		{
			F.row(index) << mesh.face_vertex(f, 2), mesh.face_vertex(f, 1), mesh.face_vertex(f, 0);
			boundary_2_all(index) = f;
			other_faces.push_back(index);
			++index;
		}
		else
		{
			auto bary = mesh.face_barycenter(f);
			for(int j = 0; j < n_f_v; ++j)
			{
				F.row(index) << mesh.face_vertex(f, j), mesh.face_vertex(f, (j+1)%n_f_v), v_index;
				boundary_2_all(index) = f;
				other_faces.push_back(index);
				++index;
			}

			V.row(v_index) = bary;
			++v_index;
		}
	}

	F.conservativeResize(index, 3);
	V.conservativeResize(v_index, 3);
	boundary_2_all.conservativeResize(index);

	Matrix<bool, Eigen::Dynamic, 1> visited(F.rows());

	MatrixXi adj;
	igl::triangle_triangle_adjacency(F, adj);

	MatrixXd N;
	igl::per_face_normals(V, F, N);


	// Initialize white
	MatrixXd C = MatrixXd::Constant(F.rows(),3,1);
	VectorXi selected(mesh.n_faces());
	selected.setZero();

	if(!bc.empty())
	{
		std::ifstream file(bc);

		std::string line;
		int bindex = 0;
		while (std::getline(file, line))
		{
			std::istringstream iss(line);
			int v;
			iss >> v;
			selected(bindex) = v;

			for(int i : all_2_local(bindex))
				C.row(i) = color(v);

			++bindex;
		}

		assert(selected.size() == bindex);

		file.close();
	}



	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	int current_id = 1;
	bool track = false;
	float vals1[3] = { 0.0f, 0.0f, 0.0f };
	float vals2[3] = { 0.0f, 0.0f, 0.0f };
	float vals3[3] = { 0.0f, 0.0f, 0.0f };
	float vals4[3] = { 0.0f, 0.0f, 0.0f };
	float vals5[3] = { 0.0f, 0.0f, 0.0f };
	float vals6[3] = { 0.0f, 0.0f, 0.0f };

	int bc_type_1 = 0;
	int bc_type_2 = 0;
	int bc_type_3 = 0;
	int bc_type_4 = 0;
	int bc_type_5 = 0;
	int bc_type_6 = 0;



	menu.callback_draw_viewer_menu = [&]()
	{
		ImGui::RadioButton("clear", &current_id, 0);
		ImGui::Separator();
		ImGui::RadioButton("1", &current_id, 1);
		ImGui::RadioButton("2", &current_id, 2);
		ImGui::RadioButton("3", &current_id, 3);
		ImGui::RadioButton("4", &current_id, 4);
		ImGui::RadioButton("5", &current_id, 5);
		ImGui::RadioButton("6", &current_id, 6);

		ImGui::Separator();

		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.80f);
		ImGui::InputFloat3("1", vals1);
		ImGui::InputFloat3("2", vals2);
		ImGui::InputFloat3("3", vals3);
		ImGui::InputFloat3("4", vals4);
		ImGui::InputFloat3("5", vals5);
		ImGui::InputFloat3("6", vals6);
		ImGui::PopItemWidth();
		ImGui::Separator();

		ImGui::TextColored(ImVec4(color(1)(0),color(1)(1),color(1)(2),1.0f), "1"); ImGui::SameLine(); ImGui::RadioButton("Dirichlet##id1", &bc_type_1, 0); ImGui::SameLine(); ImGui::RadioButton("Neuman##id1", &bc_type_1, 1);
		ImGui::TextColored(ImVec4(color(2)(0),color(2)(1),color(2)(2),1.0f), "2"); ImGui::SameLine(); ImGui::RadioButton("Dirichlet##id2", &bc_type_2, 0); ImGui::SameLine(); ImGui::RadioButton("Neuman##id2", &bc_type_2, 1);
		ImGui::TextColored(ImVec4(color(3)(0),color(3)(1),color(3)(2),1.0f), "3"); ImGui::SameLine(); ImGui::RadioButton("Dirichlet##id3", &bc_type_3, 0); ImGui::SameLine(); ImGui::RadioButton("Neuman##id3", &bc_type_3, 1);
		ImGui::TextColored(ImVec4(color(4)(0),color(4)(1),color(4)(2),1.0f), "4"); ImGui::SameLine(); ImGui::RadioButton("Dirichlet##id4", &bc_type_4, 0); ImGui::SameLine(); ImGui::RadioButton("Neuman##id4", &bc_type_4, 1);
		ImGui::TextColored(ImVec4(color(5)(0),color(5)(1),color(5)(2),1.0f), "5"); ImGui::SameLine(); ImGui::RadioButton("Dirichlet##id5", &bc_type_5, 0); ImGui::SameLine(); ImGui::RadioButton("Neuman##id5", &bc_type_5, 1);
		ImGui::TextColored(ImVec4(color(6)(0),color(6)(1),color(6)(2),1.0f), "6"); ImGui::SameLine(); ImGui::RadioButton("Dirichlet##id6", &bc_type_6, 0); ImGui::SameLine(); ImGui::RadioButton("Neuman##id6", &bc_type_6, 1);
		ImGui::Separator();

		if(ImGui::Button("save"))
		{
			std::ofstream file;
			file.open("bc.txt");

			if(file.good())
			{
				file << selected;
			}

			file.close();

			auto dirichel = json::array();
			auto neuman = json::array();

			const json json1 = {{"id", 1}, {"value", {vals1[0], vals1[1], vals1[2]}}};
			const json json2 = {{"id", 2}, {"value", {vals2[0], vals2[1], vals2[2]}}};
			const json json3 = {{"id", 3}, {"value", {vals3[0], vals3[1], vals3[2]}}};
			const json json4 = {{"id", 4}, {"value", {vals4[0], vals4[1], vals4[2]}}};
			const json json5 = {{"id", 5}, {"value", {vals5[0], vals5[1], vals5[2]}}};
			const json json6 = {{"id", 6}, {"value", {vals6[0], vals6[1], vals6[2]}}};

			if(bc_type_1 == 0)
				dirichel.push_back(json1);
			else
				neuman.push_back(json1);

			if(bc_type_2 == 0)
				dirichel.push_back(json2);
			else
				neuman.push_back(json2);

			if(bc_type_3 == 0)
				dirichel.push_back(json3);
			else
				neuman.push_back(json3);

			if(bc_type_4 == 0)
				dirichel.push_back(json4);
			else
				neuman.push_back(json4);

			if(bc_type_5 == 0)
				dirichel.push_back(json5);
			else
				neuman.push_back(json5);

			if(bc_type_6 == 0)
				dirichel.push_back(json6);
			else
				neuman.push_back(json6);

			const json args = {
				{"dirichlet_boundary", dirichel},
				{"neumann_boundary", neuman},
			};

			file.open("setting.json");
			file << args.dump(4) << std::endl;
			file.close();
		}
	};

	auto paint = [&]() {
		int fid;
		Vector3f bc;

		// Cast a ray in the view direction starting from the mouse position
		double x = viewer.current_mouse_x;
		double y = viewer.core.viewport(3) - viewer.current_mouse_y;
		if(igl::unproject_onto_mesh(Vector2f(x,y), viewer.core.view * viewer.core.model,
			viewer.core.proj, viewer.core.viewport, V, F, fid, bc))
		{

			visited.setConstant(false);
			std::queue<int> to_visit; to_visit.push(fid);

			while(!to_visit.empty())
			{
				const int id = to_visit.front();
				to_visit.pop();

				if(visited(id))
					continue;

				visited(id) = true;

				const int real_face = boundary_2_all(id);
				selected(real_face) = current_id;
				const auto &loc_faces = all_2_local(real_face);

				const auto col = color(selected[real_face]);

				for(int i : loc_faces){
					C.row(i) = col;
				}

				assert(id<adj.size());
				// auto &neighs = adj[id];
				for(int i = 0; i < 3; ++i)
				{
					const int nid = adj(id, i);
					if(visited(nid))
						continue;

					if(std::abs(N.row(fid).dot(N.row(nid)))<0.99)
						continue;

					to_visit.push(nid);
				}
			}

			viewer.data().set_colors(C);

			track = true;
			return true;
		}
		return false;
	};

	viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int modifier)->bool
	{
		//shift
		if(modifier != 1)
			return false;

		return paint();
	};

	viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	{
		if(!track)
			return false;

		return paint();
	};

	viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	{
		track = false;
		return false;
	};

	// Show mesh
	viewer.data().add_edges(p0, p1, RowVector3d(0,0,0));
	viewer.data().set_mesh(V, F);
	viewer.data().set_colors(C);
	viewer.data().show_lines = false;
	viewer.launch();
}
