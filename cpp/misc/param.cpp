#include <igl/boundary_loop.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/lscm.h>


Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd V_uv;

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{

  if (key == '1')
  {
    // Plot the 3D mesh
    viewer.data().set_mesh(V,F);
    viewer.core.align_camera_center(V,F);
  }
  else if (key == '2')
  {
    // Plot the mesh in 2D using the UV coordinates as vertex coordinates
    viewer.data().set_mesh(V_uv,F);
    viewer.core.align_camera_center(V_uv,F);
  }

  viewer.data().compute_normals();

  return false;
}

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  // Load a mesh in OFF format
  igl::readOFF("/Users/teseo/Documents/scuola/polyfem/cpp/3rdparty/libigl/tutorial/shared/lion.off", V, F);
  // igl::readOBJ("/Users/teseo/GDrive/PolyFEM/pref/experiments/bar_2d/meshes/large_angle_strip_3.obj", V, F);

  // Fix two points on the boundary
  VectorXi bnd,b(2,1);
  igl::boundary_loop(F,bnd);
  b(0) = bnd(0);
  b(1) = bnd(round(bnd.size()/2));
  MatrixXd bc(2,2);
  bc<<0,0,1,0;

  // LSCM parametrization
  igl::lscm(V,F,b,bc,V_uv);

  // Scale the uv
  V_uv *= 50;

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.data().set_uv(V_uv);
  viewer.callback_key_down = &key_down;

  // Disable wireframe
  viewer.data().show_lines = false;

  // Draw checkerboard texture
  viewer.data().show_texture = true;

  // Launch the viewer
  viewer.launch();
}


// #include <igl/boundary_loop.h>
// #include <igl/harmonic.h>
// #include <igl/map_vertices_to_circle.h>
// #include <igl/readOFF.h>
// #include <igl/readOBJ.h>
// #include <igl/opengl/glfw/Viewer.h>

// Eigen::MatrixXd V;
// Eigen::MatrixXi F;
// Eigen::MatrixXd V_uv;

// bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
// {
//   if (key == '1')
//   {
//     // Plot the 3D mesh
//     viewer.data().set_mesh(V,F);
//     viewer.core.align_camera_center(V,F);
//   }
//   else if (key == '2')
//   {
//     // Plot the mesh in 2D using the UV coordinates as vertex coordinates
//     viewer.data().set_mesh(V_uv,F);
//     viewer.core.align_camera_center(V_uv,F);
//   }

//   viewer.data().compute_normals();

//   return false;
// }

// int main(int argc, char *argv[])
// {
//   // Load a mesh in OFF format
//   // igl::readOFF("/Users/teseo/Documents/scuola/polyfem/cpp/3rdparty/libigl/tutorial/shared/camel.off", V, F);
//   // igl::readOFF("/Users/teseo/Documents/scuola/polyfem/cpp/3rdparty/libigl/tutorial/shared/lion.off", V, F);
//   // igl::readOBJ("/Users/teseo/Documents/scuola/polyfem/cpp/3rdparty/libigl/tutorial/shared/snail.obj", V, F);
//   igl::readOBJ("/Users/teseo/GDrive/PolyFEM/pref/experiments/bar_2d/meshes/large_angle_strip_7.obj", V, F);
  

//   // Find the open boundary
//   Eigen::VectorXi bnd;
//   igl::boundary_loop(F,bnd);

//   // Map the boundary to a circle, preserving edge proportions
//   Eigen::MatrixXd bnd_uv;
//   igl::map_vertices_to_circle(V,bnd,bnd_uv);

//   // Harmonic parametrization for the internal vertices
//   igl::harmonic(V,F,bnd,bnd_uv,1,V_uv);

//   // Scale UV to make the texture more clear
//   V_uv *= 5;

//   // Plot the mesh
//   igl::opengl::glfw::Viewer viewer;
//   viewer.data().set_mesh(V, F);
//   viewer.data().set_uv(V_uv);
//   viewer.callback_key_down = &key_down;

//   // Disable wireframe
//   viewer.data().show_lines = false;

//   // Draw checkerboard texture
//   viewer.data().show_texture = true;

//   // Launch the viewer
//   viewer.launch();
// }
