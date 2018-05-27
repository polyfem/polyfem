#include <Mesh3D.hpp>
#include <CommandLine.hpp>


#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>

#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include <igl/unproject_onto_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>


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
	V.resize(mesh.n_vertices(), 3);
	for(int i = 0; i < mesh.n_vertices(); ++i)
		V.row(i) = mesh.point(i);

	F.resize(mesh.n_faces(), 3);
	VectorXi boundary_2_all(mesh.n_faces());
	Matrix<std::vector<int>, Dynamic, 1> all_2_local(mesh.n_faces());

	int index = 0;
	for(int f = 0; f < mesh.n_faces(); ++f)
	{
		if(!mesh.is_boundary_face(f))
			continue;

		std::vector<int> &other_faces = all_2_local(f);

		F.row(index) << mesh.face_vertex(f, 2), mesh.face_vertex(f, 1), mesh.face_vertex(f, 0);
		boundary_2_all(index) = f;
		other_faces.push_back(index);
		++index;
	}

	F.conservativeResize(index, 3);
	boundary_2_all.conservativeResize(index);


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


	igl::opengl::glfw::Viewer viewer;

	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	int current_id = 1;
	bool track = false;

	menu.callback_draw_viewer_menu = [&]()
	{
		ImGui::RadioButton("clear", &current_id, 0);
		ImGui::Separator();
		ImGui::RadioButton("ID 1", &current_id, 1);
		ImGui::RadioButton("ID 2", &current_id, 2);
		ImGui::RadioButton("ID 3", &current_id, 3);
		ImGui::RadioButton("ID 4", &current_id, 4);
		ImGui::RadioButton("ID 5", &current_id, 5);
		ImGui::RadioButton("ID 6", &current_id, 6);

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
			const int real_face = boundary_2_all(fid);
			selected(real_face) = current_id;
			const auto &loc_faces = all_2_local(real_face);

			const auto col = color(selected[real_face]);

			for(int i : loc_faces){
				C.row(i) = col;
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
