cmake_minimum_required(VERSION 3.1)
project(PolyFEM)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Color output
include(UseColors)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# libigl
option(LIBIGL_USE_STATIC_LIBRARY 	"Use libigl as static library" ON)
option(LIBIGL_WITH_ANTTWEAKBAR      "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL             "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO           "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK             "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE           "Use Embree"         OFF)
option(LIBIGL_WITH_LIM              "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB           "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK            "Use MOSEK"          OFF)
option(LIBIGL_WITH_NANOGUI          "Use Nanogui menu"   ON)
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
option(LIBIGL_WITH_PNG              "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON           "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN           "Use Tetgen"         ON)
option(LIBIGL_WITH_TRIANGLE         "Use Triangle"       ON)
option(LIBIGL_WITH_VIEWER           "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML              "Use XML"            OFF)
find_package(LIBIGL REQUIRED QUIET)

# Geogram
include(geogram)

# Extra warnings
include(SetWarnings)



set(SRCFILES
	src/main.cpp

	src/utils/CommandLine.hpp
	src/utils/CommandLine.cpp

	src/State.hpp
	src/State.cpp

	src/UIState.hpp
	src/UIState.cpp

	src/mesh/Mesh.hpp
	src/mesh/Mesh.cpp
	src/mesh/navigation.hpp
	src/mesh/navigation.cpp

	src/quadrature/Quadrature.hpp
	src/quadrature/LineQuadrature.hpp
	src/quadrature/LineQuadrature.cpp
	src/quadrature/QuadQuadrature.cpp
	src/quadrature/QuadQuadrature.hpp
	src/quadrature/HexQuadrature.cpp
	src/quadrature/HexQuadrature.hpp

	src/basis/Basis.hpp
	src/basis/Basis.cpp

	src/basis/QuadraticBSpline.cpp
	src/basis/QuadraticBSpline.hpp

	src/basis/BSpline.cpp
	src/basis/BSpline.hpp

	src/basis/QuadraticTPBSpline.cpp
	src/basis/QuadraticTPBSpline.hpp

	src/basis/TPBSpline.cpp
	src/basis/TPBSpline.hpp

	src/basis/Spline2dBasis.cpp
	src/basis/Spline2dBasis.hpp

	src/basis/QuadBasis.cpp
	src/basis/QuadBasis.hpp
	src/basis/HexBasis.cpp
	src/basis/HexBasis.hpp

	src/assembler/AssemblyValues.hpp
	src/assembler/ElementAssemblyValues.hpp

	src/assembler/Assembler.hpp
	src/assembler/Laplacian.hpp
	src/assembler/Laplacian.cpp
	src/assembler/LinearElasticity.hpp
	src/assembler/LinearElasticity.cpp

	src/problem/Problem.hpp
	src/problem/Problem.cpp

	)


add_executable(${PROJECT_NAME} ${SRCFILES})
target_link_libraries(${PROJECT_NAME} igl::core igl::viewer igl::triangle igl::tetgen geogram)
target_include_directories(${PROJECT_NAME} PUBLIC src src/utils src/mesh src/quadrature src/basis src/assembler src/problem)
target_compile_options(${PROJECT_NAME} PRIVATE ${ALL_WARNINGS})

################################################################################

add_subdirectory(tests)
