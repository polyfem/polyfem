################################################################################
cmake_minimum_required(VERSION 3.1)
project(PolyFEM)
################################################################################

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Color output
include(UseColors)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	foreach(config ${CMAKE_CONFIGURATION_TYPES})
		string(TOUPPER ${config} config)
		string(REPLACE /MD /MT CMAKE_C_FLAGS_${config} "${CMAKE_C_FLAGS_${config}}")
		string(REPLACE /MD /MT CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
	endforeach()
endif()

################################################################################

# Options for enabling/disabling optional libraries
option(ENABLE_CHOLMOD     "Enable Cholmod library"  ON)
option(ENABLE_UMFPACK     "Enable UmfPack library"  ON)
option(ENABLE_SUPERLU     "Enable SuperLU library" OFF)
option(ENABLE_MKL         "Enable MKL library"     OFF)
option(ENABLE_PARDISO     "Enable Pardiso library"  ON)

# Options for libigl modules
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

################################################################################

# Uncomment the following lines to force options which are already cached
# set(PREFER_LOCAL_LIBS  ON  CACHE BOOL "" FORCE)
# set(ENABLE_CHOLMOD     OFF CACHE BOOL "" FORCE)
# set(ENABLE_UMFPACK     OFF CACHE BOOL "" FORCE)
# set(ENABLE_SUPERLU     OFF CACHE BOOL "" FORCE)
# set(ENABLE_MKL         OFF CACHE BOOL "" FORCE)
# set(ENABLE_PARDISO     OFF CACHE BOOL "" FORCE)

################################################################################

# Polyfem library
add_library(polyfem
	src/Common.hpp

	src/State.hpp
	src/State.cpp

	src/utils/CustomSerialization.hpp

	src/utils/CommandLine.hpp
	src/utils/CommandLine.cpp

	src/utils/QuadBoundarySampler.hpp
	src/utils/QuadBoundarySampler.cpp

	src/utils/HexBoundarySampler.hpp
	src/utils/HexBoundarySampler.cpp

	src/mesh/Mesh.hpp
	src/mesh/LocalBoundary.hpp

	src/mesh2D/Mesh2D.hpp
	src/mesh2D/Mesh2D.cpp
	src/mesh2D/MeshUtils.cpp
	src/mesh2D/MeshUtils.hpp
	src/mesh2D/Navigation.hpp
	src/mesh2D/Navigation.cpp
	src/mesh2D/PolygonUtils.hpp
	src/mesh2D/PolygonUtils.cpp
	src/mesh2D/Refinement.hpp
	src/mesh2D/Refinement.cpp
	src/mesh2D/Singularities.hpp
	src/mesh2D/Singularities.cpp

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
	src/quadrature/PolyhedronQuadrature.cpp
	src/quadrature/PolyhedronQuadrature.hpp
	src/quadrature/HexQuadrature.cpp
	src/quadrature/HexQuadrature.hpp
	src/quadrature/TetQuadrature.cpp
	src/quadrature/TetQuadrature.hpp

	src/basis/ElementBases.hpp
	src/basis/ElementBases.cpp

	src/basis/Basis.hpp
	src/basis/Basis.cpp

	src/basis/InterfaceData.hpp


	src/basis/FEBasis2d.cpp
	src/basis/FEBasis2d.hpp

	src/basis/SplineBasis2d.cpp
	src/basis/SplineBasis2d.hpp

	src/basis/PolygonalBasis2d.cpp
	src/basis/PolygonalBasis2d.hpp

	src/basis/FEBasis3d.cpp
	src/basis/FEBasis3d.hpp

	src/basis/SplineBasis3d.cpp
	src/basis/SplineBasis3d.hpp

	src/basis/PolygonalBasis3d.cpp
	src/basis/PolygonalBasis3d.hpp

	src/basis/function/QuadraticBSpline.cpp
	src/basis/function/QuadraticBSpline.hpp

	src/basis/function/QuadraticBSpline2d.cpp
	src/basis/function/QuadraticBSpline2d.hpp

	src/basis/function/QuadraticBSpline3d.cpp
	src/basis/function/QuadraticBSpline3d.hpp


	src/basis/function/Harmonic.cpp
	src/basis/function/Harmonic.hpp

	src/basis/function/Biharmonic.cpp
	src/basis/function/Biharmonic.hpp


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

	src/solver/FEMSolver.cpp
	src/solver/LinearSolver.cpp
	src/solver/LinearSolverPardiso.cpp
)

target_include_directories(polyfem PUBLIC
	src
	src/assembler
	src/basis
	src/basis/function
	src/mesh
	src/mesh2D
	src/mesh3D
	src/problem
	src/quadrature
	src/solver
	src/utils
)

################################################################################
# Required libraries
################################################################################

# Eigen
# Uncomment to use the system's version of Eigen (e.g. to use Eigen 3.3)
# find_package(Eigen3 REQUIRED)
# target_link_libraries(polyfem PUBLIC Eigen3::Eigen)

# libigl
find_package(LIBIGL REQUIRED)
target_link_libraries(polyfem PUBLIC igl::core igl::triangle igl::tetgen igl::viewer)
#remove the igl::viewer for cluster deployment

# Geogram
include(geogram)
target_link_libraries(polyfem PUBLIC geogram)

# CLI11
add_library(cli11 INTERFACE)
target_include_directories(cli11 SYSTEM INTERFACE ${THIRD_PARTY_DIR}/cli11)

# Clipper
add_subdirectory(${THIRD_PARTY_DIR}/clipper)
target_link_libraries(polyfem PUBLIC polyclipping)

# Json library
add_library(json INTERFACE)
target_include_directories(json SYSTEM INTERFACE ${THIRD_PARTY_DIR}/json/src)
target_link_libraries(polyfem PUBLIC json)

################################################################################
# Optional libraries
################################################################################

# Cholmod solver
if(ENABLE_CHOLMOD)
	find_package(Cholmod)
	if(CHOLMOD_FOUND)
		target_include_directories(polyfem PUBLIC ${CHOLMOD_INCLUDES})
		target_link_libraries(polyfem PUBLIC ${CHOLMOD_LIBRARIES})
		target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_CHOLMOD)
	endif()
endif()

# MKL library
if(ENABLE_MKL)
	find_package(MKL)
	if(MKL_FOUND)
		target_include_directories(polyfem PUBLIC ${MKL_INCLUDE_DIR})
		target_link_libraries(polyfem PUBLIC ${MKL_LIBRARIES})
		target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_MKL)
	endif()
endif()

# Pardiso solver
if(ENABLE_PARDISO AND NOT (ENABLE_MKL AND MKL_FOUND))
	find_package(Pardiso)
	if(PARDISO_FOUND)
		target_link_libraries(polyfem PUBLIC ${PARDISO_LIBRARIES})
		target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_PARDISO)
	endif()
endif()

# UmfPack solver
if(ENABLE_UMFPACK)
	find_package(Umfpack)
	if(UMFPACK_FOUND)
		target_include_directories(polyfem PUBLIC ${UMFPACK_INCLUDES})
		target_link_libraries(polyfem PUBLIC ${UMFPACK_LIBRARIES})
		target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_UMFPACK)
	endif()
endif()

# SuperLU solver
if(ENABLE_SUPERLU AND NOT (ENABLE_MKL AND MKL_FOUND))
	find_package(SuperLU)
	if(SUPERLU_FOUND)
		target_include_directories(polyfem PUBLIC ${SUPERLU_INCLUDES})
		target_link_libraries(polyfem PUBLIC ${SUPERLU_LIBRARIES})
		target_compile_definitions(polyfem PUBLIC ${SUPERLU_DEFINES})
		target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_SUPERLU)
	endif()
endif()

################################################################################
# Extra compilation options
################################################################################

# Extra warnings
include(SetWarnings)
target_compile_options(polyfem PUBLIC ${ALL_WARNINGS})

# Use C++14
include(CXXFeatures)
target_compile_features(polyfem PUBLIC ${CXX14_FEATURES})

################################################################################
# Applications
################################################################################

# Main executable
add_executable(${PROJECT_NAME}_bin
	src/main.cpp
	src/UIState.hpp
	src/UIState.cpp
)
target_link_libraries(${PROJECT_NAME}_bin PUBLIC polyfem igl::viewer)


# Main executable without gui, uncomment when not debugging gui is necessary
# add_executable(${PROJECT_NAME}_exec
# 	src/main_no_gui.cpp
# )
# target_link_libraries(${PROJECT_NAME}_exec PUBLIC polyfem)

################################################################################

add_subdirectory(tests)
