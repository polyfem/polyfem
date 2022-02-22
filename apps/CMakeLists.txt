cmake_minimum_required(VERSION 3.1)


if(NOT POLYFEM_NO_UI)
polyfem_add_application(bc_setter.cpp polyfem::polyfem igl::core igl::tetgen geogram CLI11::CLI11  igl::opengl_glfw_imgui)
endif()
polyfem_add_application(b_extractor.cpp polyfem::polyfem igl::core geogram CLI11::CLI11)

polyfem_add_application(extrude_mesh.cpp polyfem::polyfem CLI11::CLI11)
polyfem_add_application(convert_mesh.cpp polyfem::polyfem CLI11::CLI11)
polyfem_add_application(square_beam.cpp polyfem::polyfem CLI11::CLI11)
polyfem_add_application(tetrahedralize.cpp igl::core igl::tetgen geogram)
polyfem_add_application(tetrahedralize_hybrid.cpp polyfem::polyfem igl::core igl::tetgen geogram CLI11::CLI11)


if(NOT POLYFEM_NO_UI)
set_target_properties(bc_setter PROPERTIES FOLDER app)
endif()

set_target_properties(b_extractor PROPERTIES FOLDER app)
set_target_properties(convert_mesh PROPERTIES FOLDER app)
set_target_properties(extrude_mesh PROPERTIES FOLDER app)
set_target_properties(square_beam PROPERTIES FOLDER app)
set_target_properties(tetrahedralize PROPERTIES FOLDER app)
set_target_properties(tetrahedralize_hybrid PROPERTIES FOLDER app)

