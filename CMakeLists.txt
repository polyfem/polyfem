# Detects whether this is a top-level project
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
    set(POLYFEM_TOPLEVEL_PROJECT OFF)
else()
    set(POLYFEM_TOPLEVEL_PROJECT ON)
endif()

# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.25.0")
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

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  option(POLYFEM_WITH_CCACHE "Enable ccache when building PolyFEM" ${POLYFEM_TOPLEVEL_PROJECT})
else()
  option(POLYFEM_WITH_CCACHE "Enable ccache when building PolyFEM" OFF)
endif()
if(POLYFEM_WITH_CCACHE AND CCACHE_PROGRAM)
  message(STATUS "Enabling Ccache support")
  set(ccacheEnv
    CCACHE_BASEDIR=${CMAKE_BINARY_DIR}
    CCACHE_SLOPPINESS=clang_index_store,include_file_ctime,include_file_mtime,locale,pch_defines,time_macros
  )
  foreach(lang IN ITEMS C CXX)
    set(CMAKE_${lang}_COMPILER_LAUNCHER
      ${CMAKE_COMMAND} -E env ${ccacheEnv} ${CCACHE_PROGRAM}
    )
  endforeach()
endif()

################################################################################
# CMake Policies
################################################################################

cmake_policy(SET CMP0054 NEW) # Only interpret if() arguments as variables or keywords when unquoted.
cmake_policy(SET CMP0076 NEW) # target_sources() command converts relative paths to absolute.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW) # Set the timestamps of all extracted contents to the time of the extraction.
endif()

################################################################################
project(PolyFEM DESCRIPTION "A polyvalent C++ FEM library" LANGUAGES C CXX)

# Polyfem options
set(POLYFEM_SMALL_N  80 CACHE STRING "Maximum length for stack-allocated vectors (gradient + Hessian).")
set(POLYFEM_BIG_N   1000 CACHE STRING "Maximum length for stack-allocated vectors (gradient only).")

# Polyfem options for enabling/disabling optional libraries
option(POLYFEM_WITH_TESTS     "Build tests"                                 ON)
option(POLYFEM_WITH_CLIPPER   "Use clipper, necessary for polygonal bases"  ON)
option(POLYFEM_WITH_REMESHING "Uses WMTK for remeshing"                    OFF)
option(POLYFEM_WITH_MMG       "Build MMG utils for remeshing"              OFF)
option(POLYFEM_WITH_TRIANGLE  "Build target igl_restricted::triangle"      OFF)
option(POLYFEM_BUILD_DOCS     "Build documentation using Doxygen"          OFF)
option(POLYFEM_REGENERATE_AUTOGEN    "Generate the python autogen files" OFF)
set(POLYFEM_THREADING "TBB" CACHE STRING "Multithreading library to use (options: CPP, TBB, NONE)")
set_property(CACHE POLYFEM_THREADING PROPERTY STRINGS "CPP" "TBB" "NONE")

# Set default minimum C++ standard
if(POLYFEM_TOPLEVEL_PROJECT)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

### Configuration
set(POLYFEM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/polyfem")
set(POLYFEM_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/polyfem/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")

# General CMake utils
include(polyfem_autogen)
include(polyfem_cpm_cache)
include(polyfem_use_colors)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Polyfem library
################################################################################

# Add an empty library and fill in the list of sources in `src/CMakeLists.txt`.
add_library(polyfem)
add_library(polyfem::polyfem ALIAS polyfem)

# Must be call in the same "CMakeLists.txt" where add_library(polyfem ...) is called
polyfem_autogen(polyfem eigs.py auto_eigs)
polyfem_autogen(polyfem p_bases.py auto_p_bases)
polyfem_autogen(polyfem q_bases.py auto_q_bases)
polyfem_autogen(polyfem elasticity_rhs.py auto_elasticity_rhs)

# Add source and header files to polyfem
add_subdirectory("${POLYFEM_SOURCE_DIR}")

# Public include directory for PolyFEM
target_include_directories(polyfem PUBLIC "${POLYFEM_INCLUDE_DIR}")

# Use C++17
target_compile_features(polyfem PUBLIC cxx_std_17)

# No limit yay
target_compile_definitions(polyfem PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=0)

# 8MB
# target_compile_definitions(polyfem PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=8388608)

# Max stack-size small vectors (for gradient and Hessian)
target_compile_definitions(polyfem PUBLIC -DPOLYFEM_SMALL_N=${POLYFEM_SMALL_N})
target_compile_definitions(polyfem PUBLIC -DPOLYFEM_BIG_N=${POLYFEM_BIG_N})

target_compile_definitions(polyfem PUBLIC POLYFEM_INPUT_SPEC="${CMAKE_SOURCE_DIR}/input-spec.json")
target_compile_definitions(polyfem PUBLIC POLYFEM_OPT_INPUT_SPEC="${CMAKE_SOURCE_DIR}/opt-input-spec.json")
target_compile_definitions(polyfem PUBLIC POLYFEM_OBJECTIVE_INPUT_SPEC="${CMAKE_SOURCE_DIR}/objective-spec.json")

if (MSVC)
    add_compile_options(/bigobj)
endif ()

################################################################################
# CUDA
################################################################################

if(IPC_TOOLKIT_WITH_CUDA OR POLYSOLVE_WITH_CUSOLVER)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(FATAL_ERROR "No CUDA support found!")
    endif()

    set_target_properties(polyfem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
        set(CMAKE_CUDA_ARCHITECTURES "native")
        set_target_properties(ipc_toolkit PROPERTIES CUDA_ARCHITECTURES "native")
    else()
        include(FindCUDA/select_compute_arch)
        CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCH_LIST)
        string(STRIP "${CUDA_ARCH_LIST}" CUDA_ARCH_LIST)
        string(REPLACE " " ";" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
        string(REPLACE "." "" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
        set_target_properties(ipc_toolkit PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
    endif()
endif()

################################################################################
# Required libraries
################################################################################

# Eigen
include(eigen)
target_link_libraries(polyfem PUBLIC Eigen3::Eigen)

# polysolve
include(polysolve)
target_link_libraries(polyfem PUBLIC polysolve)

# IPC Toolkit
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

if(POLYFEM_WITH_TRIANGLE)
    target_link_libraries(polyfem PUBLIC igl_restricted::triangle)
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_TRIANGLE)
endif()

# BVH
include(BVH)
target_link_libraries(polyfem PUBLIC BVH_lib)

# MSHIO
include(mshio)
target_link_libraries(polyfem PUBLIC mshio)

# Geogram
include(geogram)
target_link_libraries(polyfem PUBLIC geogram)

# paraviewo (includes HDF5 and h5pp)
include(paraviewo)
target_link_libraries(polyfem PUBLIC paraviewo::paraviewo)

# spdlog
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

# JSON Specification Engine library
include(jse)
target_link_libraries(polyfem PUBLIC jse::jse)

# natsort library
include(natsort)
target_link_libraries(polyfem PUBLIC natsort::natsort)

# GLOB library
include(glob)
target_link_libraries(polyfem PUBLIC Glob::Glob)

# units
include(units)
target_link_libraries(polyfem PUBLIC units::units)

# FCPW
# include(fcpw)
# target_link_libraries(polyfem PUBLIC fcpw::fcpw)

################################################################################
# Optional libraries
################################################################################

# # MMG wrapper
if(POLYFEM_WITH_MMG)
    include(mmg)
    target_link_libraries(polyfem PUBLIC mmg::mmg)
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_MMG)
endif()

if(POLYFEM_WITH_REMESHING)
    # wmtk library
    include(wmtk)
    target_link_libraries(polyfem PUBLIC wmtk::toolkit)
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_REMESHING)
endif()

if(POLYFEM_WITH_CLIPPER)
    include(clipper)
    target_link_libraries(polyfem PUBLIC clipper::clipper)
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_WITH_CLIPPER)
endif()

# Extra warnings (link this here so it has top priority)
include(polyfem_warnings)
target_link_libraries(polyfem PRIVATE polyfem::warnings)

################################################################################
# Polyfem binary
################################################################################

# Main executable
if(POLYFEM_TOPLEVEL_PROJECT)
    add_executable(${PROJECT_NAME}_bin src/polyfem/main.cpp)

    target_compile_features(${PROJECT_NAME}_bin PUBLIC ${CXX14_FEATURES})


    target_link_libraries(${PROJECT_NAME}_bin PUBLIC polyfem::polyfem polyfem::warnings)

    include(cli11)
    target_link_libraries(${PROJECT_NAME}_bin PUBLIC CLI11::CLI11)

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
# Tests
################################################################################

# Compile extras only if this is a top-level project
if(POLYFEM_TOPLEVEL_PROJECT AND POLYFEM_WITH_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
