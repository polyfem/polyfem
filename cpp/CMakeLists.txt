cmake_minimum_required(VERSION 3.1)
project(PolyFEM)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Color output
include(UseColors)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	foreach(config ${CMAKE_CONFIGURATION_TYPES})
		string(TOUPPER ${config} config)
		string(REPLACE /MD /MT CMAKE_C_FLAGS_${config} "${CMAKE_C_FLAGS_${config}}")
		string(REPLACE /MD /MT CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
	endforeach()
endif()
# libigl
# option(ENABLE_CHOLMOD     "Enable Cholmod library"  ON)
option(ENABLE_UMFPACK     "Enable UmfPack library"  ON)


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
# CLI11
add_library(cli11 INTERFACE)
target_include_directories(cli11 SYSTEM INTERFACE ${THIRD_PARTY_DIR}/cli11)

# Json library
add_library(json INTERFACE)
target_include_directories(json SYSTEM INTERFACE ${THIRD_PARTY_DIR}/json/src)





# Extra warnings
include(SetWarnings)


# Polyfem library
add_library(polyfem
	src/State.hpp
	src/State.cpp

	src/utils/CustomSerialization.hpp

	src/utils/CommandLine.hpp
	src/utils/CommandLine.cpp

	src/utils/QuadBoundarySampler.hpp
	src/utils/QuadBoundarySampler.cpp

	src/utils/HexBoundarySampler.hpp
	src/utils/HexBoundarySampler.cpp

	src/mesh/LocalBoundary.hpp

	src/mesh/Mesh.hpp

	src/mesh/Mesh2D.hpp
	src/mesh/Mesh2D.cpp
	src/mesh/MeshUtils.cpp
	src/mesh/MeshUtils.hpp
	src/mesh/Navigation.hpp
	src/mesh/Navigation.cpp
	src/mesh/Refinement.hpp
	src/mesh/Refinement.cpp
	src/mesh/Singularities.hpp
	src/mesh/Singularities.cpp

	src/mesh3D/Mesh3DStorage.hpp
	src/mesh3D/Mesh3D.hpp
	src/mesh3D/Mesh3D.cpp
	src/mesh3D/Navigation3D.hpp
	src/mesh3D/Navigation3D.cpp
	src/mesh3D/MeshProcessing3D.hpp
	src/mesh3D/MeshProcessing3D.cpp

	src/quadrature/Quadrature.hpp
	src/quadrature/LineQuadrature.hpp
	src/quadrature/LineQuadrature.cpp
	src/quadrature/TriQuadrature.cpp
	src/quadrature/TriQuadrature.hpp
	src/quadrature/QuadQuadrature.cpp
	src/quadrature/QuadQuadrature.hpp
	src/quadrature/PolygonQuadrature.cpp
	src/quadrature/PolygonQuadrature.hpp
	src/quadrature/HexQuadrature.cpp
	src/quadrature/HexQuadrature.hpp

	src/basis/ElementBases.hpp

	src/basis/Basis.hpp
	src/basis/Basis.cpp

	src/basis/QuadraticBSpline.cpp
	src/basis/QuadraticBSpline.hpp

	src/basis/QuadraticTPBSpline.cpp
	src/basis/QuadraticTPBSpline.hpp

	src/basis/Spline2dBasis.cpp
	src/basis/Spline2dBasis.hpp

	src/basis/TriBasis.cpp
	src/basis/TriBasis.hpp

	src/basis/QuadBasis.cpp
	src/basis/QuadBasis.hpp

	src/basis/BiharmonicBasis.cpp
	src/basis/BiharmonicBasis.hpp

	src/basis/HexBasis.cpp
	src/basis/HexBasis.hpp

	src/assembler/AssemblyValues.hpp
	src/assembler/ElementAssemblyValues.hpp
	src/assembler/ElementAssemblyValues.cpp

	src/assembler/Assembler.hpp

	src/assembler/RhsAssembler.hpp
	src/assembler/RhsAssembler.cpp

	src/assembler/Laplacian.hpp
	src/assembler/Laplacian.cpp
	src/assembler/LinearElasticity.hpp
	src/assembler/LinearElasticity.cpp

	src/problem/Problem.hpp
	src/problem/Problem.cpp
)


# if(ENABLE_CHOLMOD)
# 	find_package(Cholmod)
# 	if(CHOLMOD_FOUND)
# 		target_include_directories(polyfem PUBLIC ${CHOLMOD_INCLUDES})
# 		target_link_libraries(polyfem PUBLIC ${CHOLMOD_LIBRARIES})
# 		target_compile_definitions(polyfem PUBLIC -DPOLY_FEM_WITH_CHOLMOD)
# 	endif()
# endif()

if(ENABLE_UMFPACK)
	find_package(Umfpack)
	if(UMFPACK_FOUND)
		target_include_directories(polyfem PUBLIC ${UMFPACK_INCLUDES})
		target_link_libraries(polyfem PUBLIC ${UMFPACK_LIBRARIES})
		target_compile_definitions(polyfem PUBLIC -DPOLY_FEM_WITH_UMFPACK)
	endif()
endif()

target_link_libraries(polyfem PUBLIC geogram igl::core igl::triangle igl::tetgen igl::viewer json)
target_include_directories(polyfem PUBLIC src src/assembler src/basis src/mesh src/mesh3D src/problem src/quadrature src/utils)
target_compile_options(polyfem PUBLIC ${ALL_WARNINGS})


# Main executable
add_executable(${PROJECT_NAME}_bin
	src/main.cpp
	src/UIState.hpp
	src/UIState.cpp
)
target_link_libraries(${PROJECT_NAME}_bin PUBLIC polyfem igl::viewer)


################################################################################

add_subdirectory(tests)
