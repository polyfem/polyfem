################################################################################
cmake_minimum_required(VERSION 3.14)
project(PolyFEM)
################################################################################

cmake_policy(SET CMP0079 NEW)

# cmake_policy(SET CMP0063 NEW)
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Detects whether this is a top-level project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(POLYFEM_TOPLEVEL_PROJECT ON)
else()
    set(POLYFEM_TOPLEVEL_PROJECT OFF)
endif()


if(INPUT_THIRD_PARTY_DIR)
    set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_THIRD_PARTY_DIR}/")
else()
    set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/recipes/")

if(${POLYFEM_TOPLEVEL_PROJECT})
    list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/Catch2/contrib)
endif()

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


# if(${CMAKE_VERSION} VERSION_LESS "3.15.0")
#     if(MSVC OR MSYS)
#         foreach(config ${CMAKE_CONFIGURATION_TYPES})
#             string(TOUPPER ${config} config)
#             string(REPLACE /MD /MT CMAKE_C_FLAGS_${config} "${CMAKE_C_FLAGS_${config}}")
#             string(REPLACE /MD /MT CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
#         endforeach()
#     endif()
# else()
#     set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
# endif()

################################################################################

# Polyfem options
option(POLYFEM_WITH_SANITIZERS "Enable sanitizers in compilation targets"   OFF)
set(POLYFEM_SMALL_N  80 CACHE STRING "Maximum length for stack-allocated vectors (gradient + Hessian).")
set(POLYFEM_BIG_N   1000 CACHE STRING "Maximum length for stack-allocated vectors (gradient only).")

option(POLYFEM_NO_UI            "Disables the user interface" OFF)

# Polyfem options for enabling/disabling optional libraries
option(POLYFEM_WITH_MMG              "Enable MMG library"                OFF)
option(POLYFEM_WITH_OPENCL           "Enable OpenCL"                     OFF)
option(POLYFEM_REGENERATE_AUTOGEN    "Generate the python autogen files" OFF)
set(POLYFEM_THREADING "TBB" CACHE STRING "Multithreading library to use (options: CPP, TBB, NONE)")
set_property(CACHE POLYFEM_THREADING PROPERTY STRINGS "CPP" "TBB" "NONE")

option(POLYFEM_WITH_TESTS     "Build tests"             ON)
option(POLYFEM_WITH_APPS      "Build the apps"          ON)
option(POLYFEM_WITH_MISC      "Build misc targes"       ON)

# Sanitizer options
option(POLYFEM_SANITIZE_ADDRESS   "Sanitize Address"       OFF)
option(POLYFEM_SANITIZE_MEMORY    "Sanitize Memory"        OFF)
option(POLYFEM_SANITIZE_THREAD    "Sanitize Thread"        OFF)
option(POLYFEM_SANITIZE_UNDEFINED "Sanitize Undefined"     OFF)

#Solver
option(POLYSOLVE_WITH_CHOLMOD          "Enable Cholmod library"            ON)
option(POLYSOLVE_WITH_UMFPACK          "Enable UmfPack library"            ON)
option(POLYSOLVE_WITH_SUPERLU          "Enable SuperLU library"            ON)
option(POLYSOLVE_WITH_MKL              "Enable MKL library"                ON)
option(POLYSOLVE_WITH_PARDISO          "Enable Pardiso library"            OFF)
option(POLYSOLVE_WITH_HYPRE            "Enable hypre"                      ON)
option(POLYSOLVE_WITH_AMGCL            "Use AMGCL"                         ON)
option(POLYSOLVE_WITH_SPECTRA          "Enable computing spectrum"         OFF)
option(POLYSOLVE_LARGE_INDEX           "Build for large indices"           OFF)


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
option(LIBIGL_WITH_OPENGL                 "Use OpenGL"                   ON)
option(LIBIGL_WITH_OPENGL_GLFW            "Use GLFW"                     ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI      "Use ImGui"                    ON)
option(LIBIGL_WITH_PNG                    "Use PNG"                      OFF)
option(LIBIGL_WITH_PYTHON                 "Use Python"                   OFF)
option(LIBIGL_WITH_TRIANGLE               "Use Triangle"                 OFF)
option(LIBIGL_WITH_VIEWER                 "Use OpenGL viewer"            ON)
option(LIBIGL_WITH_XML                    "Use XML"                      OFF)

if(POLYFEM_TOPLEVEL_PROJECT AND POLYFEM_WITH_APPS)
    set(LIBIGL_WITH_TETGEN ON CACHE BOOL "Use Tetgen" FORCE)
else()
    option(LIBIGL_WITH_TETGEN "Use Tetgen" OFF)
endif()


if(POLYFEM_NO_UI)
    set(LIBIGL_WITH_OPENGL             OFF  CACHE BOOL "" FORCE)
    set(LIBIGL_WITH_OPENGL_GLFW        OFF  CACHE BOOL "" FORCE)
    set(LIBIGL_WITH_OPENGL_GLFW_IMGUI  OFF  CACHE BOOL "" FORCE)
    set(LIBIGL_WITH_VIEWER             OFF  CACHE BOOL "" FORCE)
endif()

################################################################################


# set(POLYFEM_SMALL_N 105 CACHE STRING "" FORCE)

################################################################################
# Dependencies
################################################################################

# Sanitizers
if(POLYFEM_WITH_SANITIZERS)
    list(APPEND CMAKE_MODULE_PATH ${THIRD_PARTY_DIR}/sanitizers-cmake/cmake)
endif()

# Setup dependencies
include(PolyfemDependencies)

################################################################################
# Polyfem library
################################################################################

# add_library() can only be called without any source since CMake 3.11 ...
add_library(polyfem src/State.cpp)

# Must be call in the same "CMakeLists.txt" where add_library(polyfem ...) is called
polyfem_autogen(polyfem eigs.py auto_eigs)
polyfem_autogen(polyfem p_bases.py auto_p_bases)
polyfem_autogen(polyfem q_bases.py auto_q_bases)
polyfem_autogen(polyfem elasticity_rhs.py auto_elasticity_rhs)

# Public include directory for polyfem
target_include_directories(polyfem PUBLIC ${PROJECT_BINARY_DIR}/include)

set(MESH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/")
target_compile_definitions(polyfem PUBLIC -DPOLYFEM_MESH_PATH=\"${MESH_PATH}\")

# Dependencies
target_link_libraries(polyfem PUBLIC clipper::clipper)

# Extra warnings
target_link_libraries(polyfem PRIVATE warnings::all)

if(POLYFEM_THREADING STREQUAL "CPP")
    # Use C++17
    target_compile_features(polyfem PUBLIC ${CXX17_FEATURES})
else()
    # Use C++14
    target_compile_features(polyfem PUBLIC ${CXX14_FEATURES})
endif()

# No limit yay
target_compile_definitions(polyfem PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=0)

# 8MB
# target_compile_definitions(polyfem PUBLIC -DEIGEN_STACK_ALLOCATION_LIMIT=8388608)

# Max stack-size small vectors (for gradient and Hessian)
target_compile_definitions(polyfem PUBLIC -DPOLYFEM_SMALL_N=${POLYFEM_SMALL_N})
target_compile_definitions(polyfem PUBLIC -DPOLYFEM_BIG_N=${POLYFEM_BIG_N})

if("$ENV{CLUSTER}" STREQUAL "PRINCE")
    target_compile_definitions(polyfem PUBLIC -DPOLYFEM_ON_HPC)
endif()

if(POLYFEM_WITH_SANITIZERS)
    add_sanitizers(polyfem)
endif()

################################################################################
# Required libraries
################################################################################

# Eigen
# Uncomment to use the system's version of Eigen (e.g. to use Eigen 3.3)
# find_package(Eigen3 REQUIRED)
# target_link_libraries(polyfem PUBLIC Eigen3::Eigen)

# IPC
if(NOT TARGET ipc::toolkit)
    option(IPC_TOOLKIT_BUILD_UNIT_TESTS "Build unit-tests" OFF)
    polyfem_download_ipc()
    add_subdirectory(${THIRD_PARTY_DIR}/ipc)
endif()
target_link_libraries(polyfem PUBLIC ipc::toolkit)

# polysolve
polyfem_download_solvers()
add_subdirectory(${THIRD_PARTY_DIR}/solvers)
target_link_libraries(polyfem PUBLIC polysolve)

# TBB
if(POLYFEM_THREADING STREQUAL "CPP")
    target_compile_definitions(polyfem PUBLIC POLYFEM_WITH_CPP_THREADS)
elseif(POLYFEM_THREADING STREQUAL "TBB")
    if(NOT TARGET TBB::tbb)
        polyfem_download_tbb()
        set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
        set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
        set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
        set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
        set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)

        add_subdirectory(${THIRD_PARTY_DIR}/tbb tbb)
        set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")

        target_compile_definitions(tbb_static PUBLIC -DPOLYFEM_WITH_TBB)
        target_include_directories(polyfem SYSTEM PUBLIC ${THIRD_PARTY_DIR}/tbb/include)
        add_library(TBB::tbb ALIAS tbb_static)
    endif()

    target_link_libraries(polyfem PUBLIC TBB::tbb)
endif()

# libigl
if(NOT TARGET igl::core)
    polyfem_download_libigl()
    find_package(LIBIGL REQUIRED)
endif()
target_link_libraries(polyfem PUBLIC igl::core)

# BVH
polyfem_download_BVH()
add_subdirectory(${THIRD_PARTY_DIR}/BVH)
target_link_libraries(polyfem PUBLIC BVH_lib)

# MSHIO
polyfem_download_mshio()
add_subdirectory(${THIRD_PARTY_DIR}/mshio)
target_link_libraries(polyfem PUBLIC mshio)

# Geogram
polyfem_download_geogram()
include(geogram)
target_link_libraries(polyfem PUBLIC geogram)

#spdlog
target_link_libraries(polyfem PUBLIC spdlog::spdlog)

# CppNumericalSolvers
polyfem_download_CppNumericalSolvers()
add_library(cppoptlib INTERFACE)
target_include_directories(cppoptlib SYSTEM INTERFACE ${THIRD_PARTY_DIR}/CppNumericalSolvers/include)
target_link_libraries(polyfem PUBLIC cppoptlib)

# LBFGSpp
include(LBFGSpp)
target_link_libraries(polyfem PUBLIC LBFGSpp::LBFGSpp)

# TinyExpr library
include(tinyexpr)
target_link_libraries(polyfem PUBLIC tinyexpr::tinyexpr)

# TinyXML library
if(NOT TARGET tinyxml2)
    polyfem_download_tinyxml()
    SET(BUILD_SHARED_LIBS OFF)
    SET(BUILD_STATIC_LIBS ON)
    SET(BUILD_TESTING OFF)
    add_subdirectory(${THIRD_PARTY_DIR}/tinyxml)
endif()
target_link_libraries(polyfem PUBLIC tinyxml2)

# Catch2
if(${POLYFEM_TOPLEVEL_PROJECT})
    polyfem_download_catch2()
    add_library(catch INTERFACE)
    target_include_directories(catch SYSTEM INTERFACE ${THIRD_PARTY_DIR}/Catch2/single_include/catch2)
endif()

# Filesystem
target_link_libraries(polyfem PUBLIC ghc::filesystem)

################################################################################
# Optional libraries
################################################################################

# MMG wrapper
if(POLYFEM_WITH_MMG)
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


################################################################################
# Polyfem binary
################################################################################

# Main executable
if(POLYFEM_TOPLEVEL_PROJECT)
    add_executable(${PROJECT_NAME}_bin src/main.cpp)

    source_group("gui" FILES src/main.cpp)
    if(NOT POLYFEM_NO_UI)
        source_group("gui" FILES src/viewer/UIState.cpp)
        source_group("gui" FILES src/viewer/UIState.hpp)
        source_group("gui" FILES src/viewer/UIMenu.cpp)

        target_include_directories(${PROJECT_NAME}_bin PUBLIC src/viewer)
    else()
        target_compile_definitions(${PROJECT_NAME}_bin PUBLIC -DPOLYFEM_NO_UI)
    endif()

    target_compile_features(${PROJECT_NAME}_bin PUBLIC ${CXX14_FEATURES})
    target_link_libraries(${PROJECT_NAME}_bin
        PUBLIC
            polyfem
            warnings::all
            CLI11::CLI11
    )

    if(NOT POLYFEM_NO_UI)
        target_link_libraries(${PROJECT_NAME}_bin PUBLIC igl::opengl_glfw_imgui)
        # igl::png
        # tinyfiledialogs::tinyfiledialogs
    endif()

    if(POLYFEM_WITH_SANITIZERS)
        add_sanitizers(${PROJECT_NAME}_bin)
    endif()

    if(NOT (${CMAKE_VERSION} VERSION_LESS "3.6.0"))
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME}_bin)
    endif()
endif()

################################################################################
# Clutter management
################################################################################

if(TARGET igl_tetgen)
    set_target_properties(tetgen PROPERTIES FOLDER 3rdparty/other)
endif()

set_target_properties(geogram PROPERTIES FOLDER 3rdparty/other)
set_target_properties(clipper_clipper PROPERTIES FOLDER 3rdparty/other)

################################################################################
# Subdirectories
################################################################################

# Other sources
add_subdirectory(src)

# Compile extras only if this is a top-level project
if(POLYFEM_TOPLEVEL_PROJECT)
    # Auxiliary binaries
    if(POLYFEM_WITH_APPS)
        add_subdirectory(apps)
    endif()

    if(POLYFEM_WITH_MISC)
        add_subdirectory(misc)
    endif()

    # Unit tests
    if(POLYFEM_WITH_TESTS)
        include(CTest)
        enable_testing()
        add_subdirectory(tests)
    endif()
endif()
