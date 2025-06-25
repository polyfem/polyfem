list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

include(imgui)
include(tinyfd)
add_executable(polyfem_app app.cpp)
target_link_libraries(polyfem_app PRIVATE imgui::imgui polyfem::polyfem polyfem::warnings CLI11::CLI11 tinyfd::tinyfd)