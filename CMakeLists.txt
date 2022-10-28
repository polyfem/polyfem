# Detects whether this is a top-level project
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
    set(POLYFEM_TOPLEVEL_PROJECT OFF)
else()
    set(POLYFEM_TOPLEVEL_PROJECT ON)
endif()

# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.14.0")
if(POLYFEM_TOPLEVEL_PROJECT)
    cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})
else()
    # Don't use cmake_minimum_required here to avoid implicitly overriding parent policies
    if(${CMAKE_VERSION} VERSION_LESS ${REQUIRED_CMAKE_VERSION})
        message(FATAL_ERROR "CMake required version to build PolyFEM is ${REQUIRED_CMAKE_VERSION}")
    endif()
endif()

# Include user-provided default options if available. We do that before the main
# `project()` so that we can define the C/C++ compilers from the option file.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/PolyFEMOptions.cmake)
    message(STATUS "Using local options file: ${CMAKE_CURRENT_SOURCE_DIR}/PolyFEMOptions.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/PolyFEMOptions.cmake)
endif()

################################################################################
# CMake Policies
################################################################################

# cmake_policy(VERSION 3.24)
# cmake_policy(SET CMP0135 NEW) # https://cmake.org/cmake/help/latest/policy/CMP0135.html

################################################################################
project(PolyFEM DESCRIPTION "A polyvalent C++ FEM library")
################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/recipes/")

# Color output
include(UseColors)

# Prepend function
include(PrependCurrentPath)

# Polyfem utils
include(PolyfemUtils)

# Extra warnings
include(Warnings)

# Use C++11/14
include(CXXFeatures)

# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################

# Polyfem options
set(POLYFEM_SMALL_N  80 CACHE STRING "Maximum length for stack-allocated vectors (gradient + Hessian).")
set(POLYFEM_BIG_N   1000 CACHE STRING "Maximum length for stack-allocated vectors (gradient only).")

option(POLYFEM_BUILD_DOCS       "Build documentation using Doxygen" OFF)

# Polyfem options for enabling/disabling optional libraries
option(POLYFEM_REGENERATE_AUTOGEN    "Generate the python autogen files" OFF)
set(POLYFEM_THREADING "TBB" CACHE STRING "Multithreading library to use (options: CPP, TBB, NONE)")
set_property(CACHE POLYFEM_THREADING PROPERTY STRINGS "CPP" "TBB" "NONE")

option(POLYFEM_WITH_REMESHING "Uses VMTK for remeshing"                     OFF)
option(POLYFEM_WITH_TESTS     "Build tests"                                 ON)
option(POLYFEM_WITH_CLIPPER   "Use clipper, necessary for polygonal bases"  ON)

#Solver
option(POLYSOLVE_WITH_CHOLMOD          "Enable Cholmod library"            ON)
option(POLYSOLVE_WITH_UMFPACK          "Enable UmfPack library"            ON)
option(POLYSOLVE_WITH_SUPERLU          "Enable SuperLU library"            ON)
option(POLYSOLVE_WITH_MKL              "Enable MKL library"                ON)
option(POLYSOLVE_WITH_PARDISO          "Enable Pardiso library"            OFF)
option(POLYSOLVE_WITH_HYPRE            "Enable hypre"                      ON)
option(POLYSOLVE_WITH_AMGCL            "Use AMGCL"                         ON)
option(POLYSOLVE_WITH_SPECTRA          "Enable computing spectrum"         ON)
option(POLYSOLVE_LARGE_INDEX           "Build for large indices"           OFF)

if(CMAKE_OSX_ARCHITECTURES STREQUAL arm64)
    set(POLYSOLVE_WITH_CHOLMOD    OFF CACHE BOOL "Enable Cholmod library"  FORCE)
    set(POLYSOLVE_WITH_UMFPACK    OFF CACHE BOOL "Enable UmfPack library"  FORCE)
    set(POLYSOLVE_WITH_SUPERLU    OFF CACHE BOOL "Enable SuperLU library"  FORCE)
    set(POLYSOLVE_WITH_MKL        OFF CACHE BOOL "Enable MKL library"      FORCE)
endif()


# Options for libigl modules
option(LIBIGL_USE_STATIC_LIBRARY          "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR            "Use AntTweakBar"              OFF)
option(LIBIGL_WITH_CGAL                   "Use CGAL"                     OFF)
option(LIBIGL_WITH_COMISO                 "Use CoMiso"                   OFF)
option(LIBIGL_WITH_CORK                   "Use Cork"                     OFF)
option(LIBIGL_WITH_EMBREE                 "Use Embree"                   OFF)
option(LIBIGL_WITH_LIM                    "Use LIM"                      OFF)
option(LIBIGL_WITH_MATLAB                 "Use Matlab"                   OFF)
option(LIBIGL_WITH_MOSEK                  "Use MOSEK"                    OFF)
option(LIBIGL_WITH_OPENGL                 "Use OpenGL"                   OFF)
option(LIBIGL_WITH_OPENGL_GLFW            "Use GLFW"                     OFF)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI      "Use ImGui"                    OFF)
option(LIBIGL_WITH_PNG                    "Use PNG"                      OFF)
option(LIBIGL_WITH_PYTHON                 "Use Python"                   OFF)
option(LIBIGL_WITH_TRIANGLE               "Use Triangle"                 OFF)
option(LIBIGL_WITH_VIEWER                 "Use OpenGL viewer"            OFF)
option(LIBIGL_WITH_XML                    "Use XML"                      OFF)
option(LIBIGL_WITH_TETGEN "Use Tetgen" OFF)

################################################################################


# set(POLYFEM_SMALL_N 105 CACHE STRING "" FORCE)

# Sanitizers
include(sanitizers)


################################################################################
# Polyfem library
################################################################################

# add_library() can only be called without any source since CMake 3.11 ...
add_library(polyfem)
add_library(polyfem::polyfem ALIAS polyfem)

# Must be call in the same "CMakeLists.txt" where add_library(polyfem ...) is called
polyfem_autogen(polyfem eigs.py auto_eigs)
polyfem_autogen(polyfem p_bases.py auto_p_bases)
polyfem_autogen(polyfem q_bases.py auto_q_bases)
polyfem_autogen(polyfem elasticity_rhs.py auto_elasticity_rhs)

# Public include directory for polyfem
target_include_directories(polyfem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Use C++17
target_compile_features(polyfem PUBLIC ${CXX17_FEATURES})

# No limit yay
target_compile_definitions(polyfem PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=0)

# 8MB
# target_compile_definitions(polyfem PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=8388608)

# Max stack-size small vectors (for gradient and Hessian)
target_compile_definitions(polyfem PUBLIC -DPOLYFEM_SMALL_N=${POLYFEM_SMALL_N})
target_compile_definitions(polyfem PUBLIC -DPOLYFEM_BIG_N=${POLYFEM_BIG_N})

target_compile_definitions(polyfem PUBLIC POLYFEM_INPUT_SPEC="${CMAKE_SOURCE_DIR}/input-spec.json")

if (MSVC)
    add_compile_options(/bigobj)
endif ()

################################################################################
# Required libraries
################################################################################

# Eigen
include(eigen)
target_link_libraries(polyfem PUBLIC Eigen3::Eigen)

# polysolve
include(polysolver)
target_link_libraries(polyfem PUBLIC polysolve)

include(ipc_toolkit)
target_link_libraries(polyfem PUBLIC ipc::toolkit)

# Threading
if(POLYFEM_THREADING STREQUAL "CPP")
    target_compile_definitions(polyfem PUBLIC POLYFEM_WITH_CPP_THREADS)
    # These definitions help avoid problems with GCC using TBB backend
    target_compile_definitions(polyfem PUBLIC PSTL_USE_PARALLEL_POLICIES=0)
    target_compile_definitions(polyfem PUBLIC _GLIBCXX_USE_TBB_PAR_BACKEND=0)
elseif(POLYFEM_THREADING STREQUAL "TBB")
    include(onetbb)
    target_link_libraries(polyfem PUBLIC TBB::tbb)
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_TBB)
endif()

# libigl
include(libigl)
target_link_libraries(polyfem PUBLIC igl::core)

# BVH
include(BVH)
target_link_libraries(polyfem PUBLIC BVH_lib)

# MSHIO
include(mshio)
target_link_libraries(polyfem PUBLIC mshio)

# Geogram
include(geogram)
target_link_libraries(polyfem PUBLIC geogram)

#spdlog
include(spdlog)
target_link_libraries(polyfem PUBLIC spdlog::spdlog)

# CppNumericalSolvers
include(cppoptlib)
target_link_libraries(polyfem PUBLIC cppoptlib)

# LBFGSpp
include(LBFGSpp)
target_link_libraries(polyfem PUBLIC LBFGSpp::LBFGSpp)

# TinyExpr library
include(tinyexpr)
target_link_libraries(polyfem PUBLIC tinyexpr::tinyexpr)

# TinyXML library
include(tinyxml)
target_link_libraries(polyfem PUBLIC tinyxml2)


# jse library
include(jse)
target_link_libraries(polyfem PUBLIC jse::jse)


if(POLYFEM_WITH_REMESHING)
    # wmtk library
    include(wmtk)
    target_link_libraries(polyfem PUBLIC wmtk::toolkit)
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_REMESHING)
endif()

# HighFive + HDF5 library
include(hdf5)
target_link_libraries(polyfem PUBLIC hdf5::hdf5)

include(high_five)
target_link_libraries(polyfem PUBLIC HighFive::HighFive)

# natsort library
include(natsort)
target_link_libraries(polyfem PUBLIC natsort::natsort)

# GLOB library
include(glob)
target_link_libraries(polyfem PUBLIC Glob::Glob)

# Extra warnings (include this at the end to have highest priority)
target_link_libraries(polyfem PRIVATE polyfem::warnings)

################################################################################
# Optional libraries
################################################################################

# # MMG wrapper
# if(POLYFEM_WITH_MMG)
#     include(ExternalProject)
#     set(MMG_DIR "${POLYFEM_DATA_DIR}/mmg")
#     ExternalProject_Add(MMG_Project
#         PREFIX "${MMG_DIR}"
#         GIT_REPOSITORY https://github.com/MmgTools/mmg.git
#         GIT_TAG "v5.3.8"

#         UPDATE_COMMAND ""
#         PATCH_COMMAND ""

#         CMAKE_ARGS
#             -DCMAKE_PREFIX_PATH=${MMG_DIR}/bin
#             -DBUILD_SHARED_LIBS=OFF
#             "-DCMAKE_INSTALL_PREFIX=${MMG_DIR}/"

#         TEST_COMMAND "")

#     set(MMG_PATH "${POLYFEM_DATA_DIR}/mmg/bin/mmg3d_O3")
#     target_compile_definitions(polyfem PUBLIC -DPOLYFEM_MMG_PATH=\"${MMG_PATH}\")
#     target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_MMG)

#     # Boost
#     find_package(Boost 1.61 QUIET REQUIRED COMPONENTS filesystem system)
#     target_include_directories(polyfem PUBLIC ${Boost_INCLUDE_DIRS})
#     target_link_libraries(polyfem PUBLIC ${Boost_LIBRARIES})
# endif()

if(POLYFEM_WITH_CLIPPER)
    include(clipper)
    target_link_libraries(polyfem PUBLIC clipper::clipper)
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_CLIPPER)
endif()

include(finite_diff)
target_link_libraries(polyfem PUBLIC finitediff::finitediff)

################################################################################
# Polyfem binary
################################################################################

# Main executable
if(POLYFEM_TOPLEVEL_PROJECT)
    add_executable(${PROJECT_NAME}_bin src/polyfem/main.cpp)
    add_executable(shape_optimization src/polyfem/main_shape_opt.cpp)
    add_executable(trajectory_optimization src/polyfem/main_trajectory_opt.cpp)
    add_executable(topology_optimization src/polyfem/main_top_opt.cpp)
    add_executable(homogenization src/polyfem/main_homogenization.cpp)

    target_compile_features(${PROJECT_NAME}_bin PUBLIC ${CXX14_FEATURES})
    include(cli11)
    target_link_libraries(${PROJECT_NAME}_bin
        PUBLIC
            polyfem::polyfem
            polyfem::warnings
            CLI11::CLI11
    )

    target_compile_features(shape_optimization PUBLIC ${CXX14_FEATURES})
    target_link_libraries(shape_optimization
        PUBLIC
            polyfem::polyfem
            polyfem::warnings
            CLI11::CLI11
    )

    target_compile_features(trajectory_optimization PUBLIC ${CXX14_FEATURES})
    target_link_libraries(trajectory_optimization
        PUBLIC
            polyfem::polyfem
            polyfem::warnings
            CLI11::CLI11
    )

    target_compile_features(topology_optimization PUBLIC ${CXX14_FEATURES})
    target_link_libraries(topology_optimization
        PUBLIC
            polyfem::polyfem
            polyfem::warnings
            CLI11::CLI11
    )

    target_compile_features(homogenization PUBLIC ${CXX14_FEATURES})
    target_link_libraries(homogenization
        PUBLIC
            polyfem::polyfem
            polyfem::warnings
            CLI11::CLI11
    )

    if(NOT (${CMAKE_VERSION} VERSION_LESS "3.6.0"))
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME}_bin)
    endif()

    if(POLYFEM_BUILD_DOCS)
        # check if Doxygen is installed
        find_package(Doxygen)
        if (DOXYGEN_FOUND)
            add_custom_target(polyfem_doc_doxygen ALL
                COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Generating API documentation with Doxygen"
                VERBATIM)
        else (DOXYGEN_FOUND)
            message("Doxygen need to be installed to generate the doxygen documentation")
        endif (DOXYGEN_FOUND)
    endif()
endif()

################################################################################
# Subdirectories
################################################################################

# Other sources
add_subdirectory(src/polyfem)

# Compile extras only if this is a top-level project
if(POLYFEM_TOPLEVEL_PROJECT)
    # Unit tests
    if(POLYFEM_WITH_TESTS)
        include(CTest)
        enable_testing()
        add_subdirectory(tests)
    endif()
endif()

################################################################################
# CUDA
################################################################################

if(IPC_TOOLKIT_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "No CUDA support found!")
  endif()

  set_target_properties(polyfem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # Nvidia RTX8000 -> compute_75
  # Nvidia V100 -> compute_70
  # Nvidia 1080/1080Ti -> compute_61
  # Nvidia 3080Ti -> compute_86
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 86)
  endif()
  set_target_properties(polyfem PROPERTIES CUDA_ARCHITECTURES "70;75;86")
  set_target_properties(PolyFEM_bin PROPERTIES CUDA_ARCHITECTURES "70;75;86")
endif()
