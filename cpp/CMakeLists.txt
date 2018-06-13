################################################################################
cmake_minimum_required(VERSION 3.1)
project(PolyFEM)
################################################################################

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/Catch2/contrib)
# list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/sanitizers-cmake/cmake)

# Color output
include(UseColors)

# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

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
option(ENABLE_MMG         "Enable MMG library"     OFF)
option(ENABLE_TBB         "Enable TBB"		        ON)
option(ENABLE_HYPRE       "Enable hypre"	        ON)
option(SANITIZE_ADDRESS   "Sanitize Address"       OFF)
option(SANITIZE_MEMORY    "Sanitize Memory"        OFF)
option(SANITIZE_THREAD    "Sanitize Thread"        OFF)
option(SANITIZE_UNDEFINED "Sanitize Undefined"     OFF)

# Options for libigl modules
option(LIBIGL_USE_STATIC_LIBRARY          "Use libigl as static library" ON)
option(LIBIGL_WITH_ANTTWEAKBAR            "Use AntTweakBar"              OFF)
option(LIBIGL_WITH_CGAL                   "Use CGAL"                     OFF)
option(LIBIGL_WITH_COMISO                 "Use CoMiso"                   OFF)
option(LIBIGL_WITH_CORK                   "Use Cork"                     OFF)
option(LIBIGL_WITH_EMBREE                 "Use Embree"                   OFF)
option(LIBIGL_WITH_LIM                    "Use LIM"                      OFF)
option(LIBIGL_WITH_MATLAB                 "Use Matlab"                   OFF)
option(LIBIGL_WITH_MOSEK                  "Use MOSEK"                    OFF)
option(LIBIGL_WITH_OPENGL                 "Use OpenGL"                   ON)
option(LIBIGL_WITH_OPENGL_GLFW            "Use GLFW"                     ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI      "Use ImGui"                    ON)
option(LIBIGL_WITH_PNG                    "Use PNG"                      ON)
option(LIBIGL_WITH_PYTHON                 "Use Python"                   OFF)
option(LIBIGL_WITH_TETGEN                 "Use Tetgen"                   ON)
option(LIBIGL_WITH_TRIANGLE               "Use Triangle"                 ON)
option(LIBIGL_WITH_VIEWER                 "Use OpenGL viewer"            ON)
option(LIBIGL_WITH_XML                    "Use XML"                      OFF)

################################################################################

# Uncomment the following lines to force options which are already cached
# set(PREFER_LOCAL_LIBS   ON  CACHE BOOL "" FORCE)
# set(ENABLE_MKL          OFF CACHE BOOL "" FORCE)
# set(ENABLE_PARDISO      OFF CACHE BOOL "" FORCE)
# set(ENABLE_MMG          ON  CACHE BOOL "" FORCE)

# set(SANITIZE_ADDRESS   ON  CACHE BOOL "" FORCE)
# set(SANITIZE_MEMORY    ON CACHE BOOL "" FORCE)
# set(SANITIZE_THREAD    OFF CACHE BOOL "" FORCE)
# set(SANITIZE_UNDEFINED ON  CACHE BOOL "" FORCE)

# Somehow Pardiso 6 conflicts with libmetis that these guys depends on... Grr
if(ENABLE_PARDISO)
	set(ENABLE_CHOLMOD      OFF CACHE BOOL "" FORCE)
	set(ENABLE_UMFPACK      OFF CACHE BOOL "" FORCE)
	set(ENABLE_SUPERLU      OFF CACHE BOOL "" FORCE)
endif()

################################################################################

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_rhs.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_rhs.hpp
	COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/elasticity_rhs.py ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/elasticity_rhs.py)


add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_eigs.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_eigs.hpp
	COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/eigs.py ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/eigs.py)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_bases.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_bases.hpp
	COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/FEBases.py ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/FEBases.py)



# Polyfem library
set(POLYFEM_SOURCES
	src/assembler/Assembler.cpp
	src/assembler/Assembler.hpp
	src/assembler/AssemblyValues.hpp
	src/assembler/ElementAssemblyValues.cpp
	src/assembler/ElementAssemblyValues.hpp
	src/assembler/Helmholtz.cpp
	src/assembler/Helmholtz.hpp
	src/assembler/HookeLinearElasticity.cpp
	src/assembler/HookeLinearElasticity.hpp
	src/assembler/Laplacian.cpp
	src/assembler/Laplacian.hpp
	src/assembler/LinearElasticity.cpp
	src/assembler/LinearElasticity.hpp
	src/assembler/NeoHookeanElasticity.cpp
	src/assembler/NeoHookeanElasticity.hpp
	src/assembler/OgdenElasticity.cpp
	src/assembler/OgdenElasticity.hpp
	src/assembler/RhsAssembler.cpp
	src/assembler/RhsAssembler.hpp
	src/assembler/SaintVenantElasticity.cpp
	src/assembler/SaintVenantElasticity.hpp
	src/assembler/utils/AssemblerUtils.cpp
	src/assembler/utils/AssemblerUtils.hpp
	src/basis/Basis.cpp
	src/basis/Basis.hpp
	src/basis/ElementBases.cpp
	src/basis/ElementBases.hpp
	src/basis/FEBasis2d.cpp
	src/basis/FEBasis2d.hpp
	src/basis/FEBasis3d.cpp
	src/basis/FEBasis3d.hpp
	src/basis/function/QuadraticBSpline.cpp
	src/basis/function/QuadraticBSpline.hpp
	src/basis/function/QuadraticBSpline2d.cpp
	src/basis/function/QuadraticBSpline2d.hpp
	src/basis/function/QuadraticBSpline3d.cpp
	src/basis/function/QuadraticBSpline3d.hpp
	src/basis/function/RBFWithLinear.cpp
	src/basis/function/RBFWithLinear.hpp
	src/basis/function/RBFWithQuadratic.cpp
	src/basis/function/RBFWithQuadratic.hpp
	src/basis/function/RBFWithQuadraticLagrange.cpp
	src/basis/function/RBFWithQuadraticLagrange.hpp
	src/basis/InterfaceData.hpp
	src/basis/PolygonalBasis2d.cpp
	src/basis/PolygonalBasis2d.hpp
	src/basis/PolygonalBasis3d.cpp
	src/basis/PolygonalBasis3d.hpp
	src/basis/SpectralBasis2d.cpp
	src/basis/SpectralBasis2d.hpp
	src/basis/SplineBasis2d.cpp
	src/basis/SplineBasis2d.hpp
	src/basis/SplineBasis3d.cpp
	src/basis/SplineBasis3d.hpp
	src/Common.hpp
	src/mesh/LocalBoundary.hpp
	src/mesh/Mesh.cpp
	src/mesh/Mesh.hpp
	src/mesh/MeshNodes.cpp
	src/mesh/MeshNodes.hpp
	src/mesh/VTUWriter.cpp
	src/mesh/VTUWriter.hpp
	src/mesh2D/Mesh2D.cpp
	src/mesh2D/Mesh2D.hpp
	src/mesh2D/MeshUtils.cpp
	src/mesh2D/MeshUtils.hpp
	src/mesh2D/Navigation.cpp
	src/mesh2D/Navigation.hpp
	src/mesh2D/PolygonUtils.cpp
	src/mesh2D/PolygonUtils.hpp
	src/mesh2D/Refinement.cpp
	src/mesh2D/Refinement.hpp
	src/mesh2D/Singularities.cpp
	src/mesh2D/Singularities.hpp
	src/mesh3D/Mesh3D.cpp
	src/mesh3D/Mesh3D.hpp
	src/mesh3D/Mesh3DStorage.hpp
	src/mesh3D/MeshProcessing3D.cpp
	src/mesh3D/MeshProcessing3D.hpp
	src/mesh3D/Navigation3D.cpp
	src/mesh3D/Navigation3D.hpp
	src/problem/PointBasedProblem.cpp
	src/problem/PointBasedProblem.hpp
	src/problem/GenericProblem.cpp
	src/problem/GenericProblem.hpp
	src/problem/ElasticProblem.cpp
	src/problem/ElasticProblem.hpp
	src/problem/FrankeProblem.cpp
	src/problem/FrankeProblem.hpp
	src/problem/KernelProblem.cpp
	src/problem/KernelProblem.hpp
	src/problem/MiscProblem.cpp
	src/problem/MiscProblem.hpp
	src/problem/TestProblem.cpp
	src/problem/TestProblem.hpp
	src/problem/Problem.cpp
	src/problem/Problem.hpp
	src/problem/ProblemWithSolution.cpp
	src/problem/ProblemWithSolution.hpp
	src/quadrature/HexQuadrature.cpp
	src/quadrature/HexQuadrature.hpp
	src/quadrature/LineQuadrature.cpp
	src/quadrature/LineQuadrature.hpp
	src/quadrature/PolygonQuadrature.cpp
	src/quadrature/PolygonQuadrature.hpp
	src/quadrature/PolyhedronQuadrature.cpp
	src/quadrature/PolyhedronQuadrature.hpp
	src/quadrature/QuadQuadrature.cpp
	src/quadrature/QuadQuadrature.hpp
	src/quadrature/Quadrature.hpp
	src/quadrature/TetQuadrature.cpp
	src/quadrature/TetQuadrature.hpp
	src/quadrature/TriQuadrature.cpp
	src/quadrature/TriQuadrature.hpp
	src/solver/FEMSolver.cpp
	src/solver/LinearSolver.cpp
	src/solver/LinearSolverHypre.cpp
	src/solver/LinearSolverPardiso.cpp
	src/solver/NLProblem.cpp
	src/solver/NLProblem.hpp
	src/solver/SparseNewtonDescentSolver.hpp
	src/State.cpp
	src/State.hpp
	src/utils/autodiff.h
	src/utils/AutodiffTypes.hpp
	src/utils/Bessel.hpp
	src/utils/BoundarySampler.cpp
	src/utils/BoundarySampler.hpp
	src/utils/CommandLine.cpp
	src/utils/CommandLine.hpp
	src/utils/EdgeSampler.cpp
	src/utils/EdgeSampler.hpp
	src/utils/ElasticityUtils.cpp
	src/utils/ElasticityUtils.hpp
	src/utils/InterpolatedFunction.cpp
	src/utils/InterpolatedFunction.hpp
	src/utils/MatrixUtils.cpp
	src/utils/MatrixUtils.hpp
	src/utils/RefElementSampler.cpp
	src/utils/RefElementSampler.hpp
	src/utils/StringUtils.cpp
	src/utils/StringUtils.hpp
	src/utils/ExpressionValue.cpp
	src/utils/ExpressionValue.hpp
	src/utils/Types.hpp
)

add_library(polyfem
	${POLYFEM_SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_rhs.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_rhs.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_eigs.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_eigs.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_bases.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_bases.hpp

	${THIRD_PARTY_DIR}/tinyexpr/tinyexpr.c
)

target_include_directories(polyfem PUBLIC
	src
	src/autogen
	src/assembler
	src/assembler/utils
	src/basis
	src/basis/function
	src/mesh
	src/mesh2D
	src/mesh3D
	src/problem
	src/quadrature
	src/solver
	src/utils
	src/viewer

	${THIRD_PARTY_DIR}/tinyexpr/
)

foreach(source IN ITEMS ${POLYFEM_SOURCES})
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "/" "\\" group_path "${source_path}")
    source_group("${group_path}" FILES "${source}")
endforeach()

source_group("autogen" FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_rhs.cpp)
source_group("autogen" FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_rhs.hpp)
source_group("autogen" FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_eigs.cpp)
source_group("autogen" FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_eigs.hpp)
source_group("autogen" FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_bases.cpp)
source_group("autogen" FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/autogen/auto_bases.hpp)

set(MESH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/")
target_compile_definitions(polyfem PUBLIC -DPOLYFEM_MESH_PATH=\"${MESH_PATH}\")

# No limit yay
target_compile_definitions(polyfem PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=0)

# 8MB
# target_compile_definitions(polyfem PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=8388608)

if("$ENV{CLUSTER}" STREQUAL "PRINCE")
	target_compile_definitions(polyfem PUBLIC -DPOLYFEM_ON_HPC)
endif()

################################################################################
# Required libraries
################################################################################

# Sanitizer flags
# find_package(Sanitizers)
# add_sanitizers(polyfem)

# Eigen
# Uncomment to use the system's version of Eigen (e.g. to use Eigen 3.3)
# find_package(Eigen3 REQUIRED)
# target_link_libraries(polyfem PUBLIC Eigen3::Eigen)

# TBB
if(ENABLE_TBB)
	set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
	set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
	set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
	set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
	set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)

	add_subdirectory(${THIRD_PARTY_DIR}/tbb tbb)
	set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")

	target_compile_definitions(tbb_static PUBLIC -DUSE_TBB)
	target_include_directories(polyfem SYSTEM PUBLIC ${THIRD_PARTY_DIR}/tbb/include)
	target_link_libraries(polyfem PUBLIC tbb_static)
endif()

# libigl
find_package(LIBIGL REQUIRED)
target_link_libraries(polyfem PUBLIC igl::core igl::triangle igl::tetgen igl::opengl_glfw_imgui)
#remove the igl::viewer for cluster deployment

# Geogram
include(geogram)
target_link_libraries(polyfem PUBLIC geogram)

# CppNumericalSolvers
add_library(cppoptlib INTERFACE)
target_include_directories(cppoptlib SYSTEM INTERFACE ${THIRD_PARTY_DIR}/CppNumericalSolvers/include)
target_link_libraries(polyfem PUBLIC cppoptlib)

# CLI11
add_library(cli11 INTERFACE)
target_include_directories(cli11 SYSTEM INTERFACE ${THIRD_PARTY_DIR}/cli11)

# Spectra
add_library(spectra INTERFACE)
target_include_directories(spectra SYSTEM INTERFACE ${THIRD_PARTY_DIR}/spectra/include)
target_link_libraries(polyfem PUBLIC spectra)

# Clipper
add_subdirectory(${THIRD_PARTY_DIR}/clipper)
target_link_libraries(polyfem PUBLIC polyclipping)

# Json
add_library(json INTERFACE)
target_include_directories(json SYSTEM INTERFACE ${THIRD_PARTY_DIR}/json/include)
target_link_libraries(polyfem PUBLIC json)

# NanoSVG
add_library(nanosvg INTERFACE)
target_include_directories(nanosvg INTERFACE ${THIRD_PARTY_DIR}/nanosvg)

# Tiny file dialogs
add_subdirectory(${THIRD_PARTY_DIR}/tinyfiledialogs tinyfiledialogs)

# Tiny format
add_library(tinyformat INTERFACE)
target_include_directories(tinyformat SYSTEM INTERFACE ${THIRD_PARTY_DIR}/tinyformat)
target_link_libraries(polyfem PUBLIC tinyformat)

# Catch2
add_library(catch INTERFACE)
target_include_directories(catch SYSTEM INTERFACE ${THIRD_PARTY_DIR}/Catch2/single_include)

# Hypre
if(ENABLE_HYPRE)
	include(hypre)
	target_link_libraries(polyfem PUBLIC HYPRE)
	target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_HYPRE)
	# add_sanitizers(HYPRE)
endif()

#Rbf interpolation
include(rbf)
if(TARGET RBF)
	target_link_libraries(polyfem PUBLIC RBF)
	target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_RBF)
endif()



################################################################################
# Optional libraries
################################################################################

# MMG wrapper
if(ENABLE_MMG)
	include(ExternalProject)
	set(MMG_DIR "${THIRD_PARTY_DIR}/mmg")
	ExternalProject_Add(MMG_Project
		PREFIX "${MMG_DIR}"
		GIT_REPOSITORY https://github.com/MmgTools/mmg.git
		GIT_TAG "v5.3.8"

		UPDATE_COMMAND ""
		PATCH_COMMAND ""

		CMAKE_ARGS
			-DCMAKE_PREFIX_PATH=${MMG_DIR}/bin
			-DBUILD_SHARED_LIBS=OFF
			"-DCMAKE_INSTALL_PREFIX=${MMG_DIR}/"

		TEST_COMMAND "")

	set(MMG_PATH "${THIRD_PARTY_DIR}/mmg/bin/mmg3d_O3")
	target_compile_definitions(polyfem PUBLIC -DPOLYFEM_MMG_PATH=\"${MMG_PATH}\")
	target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_MMG)

	# Boost
	find_package(Boost 1.61 QUIET REQUIRED COMPONENTS filesystem system)
	target_include_directories(polyfem PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(polyfem PUBLIC ${Boost_LIBRARIES})
endif()

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
		find_package(LAPACK)
		if(LAPACK_FOUND)
			target_link_libraries(polyfem PUBLIC ${LAPACK_LIBRARIES})
		else()
			message(FATAL_ERROR "unable to find lapack")
		endif()

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
	src/viewer/UIState.cpp
	src/viewer/UIState.hpp
	src/viewer/UIMenu.cpp
)

source_group("gui" FILES src/main.cpp)
source_group("gui" FILES src/viewer/UIState.cpp)
source_group("gui" FILES src/viewer/UIState.hpp)
source_group("gui" FILES src/viewer/UIMenu.cpp)

target_include_directories(${PROJECT_NAME}_bin PUBLIC
	src/viewer
)

target_link_libraries(${PROJECT_NAME}_bin PUBLIC igl::opengl_glfw_imgui igl::png polyfem tinyfiledialogs)
target_compile_features(${PROJECT_NAME}_bin PUBLIC ${CXX14_FEATURES})

# add_sanitizers(${PROJECT_NAME}_bin)

set_target_properties(igl_triangle PROPERTIES FOLDER 3rdparty/igl)
set_target_properties(igl_opengl PROPERTIES FOLDER 3rdparty/igl)
if(TARGET igl_png)
set_target_properties(igl_png PROPERTIES FOLDER 3rdparty/igl)
endif()
set_target_properties(igl_opengl_glfw_imgui PROPERTIES FOLDER 3rdparty/igl)
set_target_properties(igl_core PROPERTIES FOLDER 3rdparty/igl)
set_target_properties(igl_tetgen PROPERTIES FOLDER 3rdparty/igl)
set_target_properties(igl_opengl_glfw PROPERTIES FOLDER 3rdparty/igl)
if(TARGET igl_stb_image)
set_target_properties(igl_stb_image PROPERTIES FOLDER 3rdparty/igl)
endif()

set_target_properties(glad PROPERTIES FOLDER 3rdparty/other)
set_target_properties(tinyfiledialogs PROPERTIES FOLDER 3rdparty/other)
set_target_properties(imgui PROPERTIES FOLDER 3rdparty/other)
set_target_properties(tetgen PROPERTIES FOLDER 3rdparty/other)
set_target_properties(triangle PROPERTIES FOLDER 3rdparty/other)
# set_target_properties(uninstall_geo PROPERTIES FOLDER 3rdparty/other)
set_target_properties(geogram PROPERTIES FOLDER 3rdparty/other)
set_target_properties(polyclipping PROPERTIES FOLDER 3rdparty/other)

# Main executable without gui, uncomment when not debugging gui is necessary
# add_executable(${PROJECT_NAME}_cmd
# 	src/main_no_gui.cpp
# )
# target_link_libraries(${PROJECT_NAME}_cmd PUBLIC polyfem)

################################################################################

add_subdirectory(misc)

################################################################################

# Unit tests
include(CTest)
enable_testing()
add_subdirectory(tests)
