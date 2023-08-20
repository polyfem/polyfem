# Geogram (https://github.com/polyfem/geogram)
# License: BSD

if(TARGET geogram)
    return()
endif()

message(STATUS "Third-party: creating target 'geogram'")

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	set(VORPALINE_ARCH_64 TRUE CACHE BOOL "" FORCE)
	set(VORPALINE_PLATFORM Win-vs-generic CACHE STRING "" FORCE)
	set(VORPALINE_BUILD_DYNAMIC false CACHE STRING "" FORCE)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	set(VORPALINE_PLATFORM Linux64-gcc CACHE STRING "" FORCE)
    set(VORPALINE_BUILD_DYNAMIC false CACHE STRING "" FORCE)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(VORPALINE_PLATFORM Darwin-clang CACHE STRING "" FORCE)
	set(VORPALINE_BUILD_DYNAMIC false CACHE STRING "" FORCE)
endif()

option(GEOGRAM_WITH_GRAPHICS "Viewers and geogram_gfx library" OFF)
option(GEOGRAM_WITH_LEGACY_NUMERICS "Legacy numerical libraries" OFF)
option(GEOGRAM_WITH_HLBFGS "Non-linear solver (Yang Liu's HLBFGS)" ON)
option(GEOGRAM_WITH_TETGEN "Tetrahedral mesher (Hang Si's TetGen)" OFF)
option(GEOGRAM_WITH_TRIANGLE "Triangle mesher (Jonathan Shewchuk's triangle)" OFF)
option(GEOGRAM_WITH_EXPLORAGRAM "Experimental code (hexahedral meshing vpipeline and optimal transport)" OFF)
option(GEOGRAM_WITH_LUA "Built-in LUA interpreter" OFF)
option(GEOGRAM_LIB_ONLY "Libraries only (no example programs/no viewer)" ON)
option(GEOGRAM_WITH_FPG "Predicate generator (Sylvain Pion's FPG)" OFF)
option(GEOGRAM_USE_SYSTEM_GLFW3 "Use the version of GLFW3 installed in the system if found" OFF)

# Workaround CMake limitation with geogram 1.5.4
if(TARGET glfw AND NOT TARGET glfw3)
	add_library(glfw3 ALIAS glfw)
endif()

################################################################################

include(CPM)
CPMAddPackage("gh:polyfem/geogram#c790c30daf05839fb4d64b83f2e1fababdf17622")

find_path(GEOGRAM_SOURCE_INCLUDE_DIR
		geogram/basic/common.h
		PATHS ${geogram_SOURCE_DIR}
		PATH_SUFFIXES src/lib
		NO_DEFAULT_PATH
)

set(GEOGRAM_ROOT ${GEOGRAM_SOURCE_INCLUDE_DIR}/../..)

message(STATUS "Found Geogram here: ${GEOGRAM_ROOT}")

# add_subdirectory(${GEOGRAM_ROOT} geogram)
target_include_directories(geogram SYSTEM PUBLIC ${GEOGRAM_SOURCE_INCLUDE_DIR})

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	SET_TARGET_PROPERTIES(geogram PROPERTIES COMPILE_FLAGS -fopenmp LINK_FLAGS -fopenmp)
	target_compile_options(geogram PUBLIC    -fopenmp)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

	find_package(OpenMP REQUIRED)
	if(NOT TARGET OpenMP::OpenMP_CXX)
   		add_library(OpenMP_TARGET INTERFACE)
    		add_library(OpenMP::OpenMP_CXX ALIAS OpenMP_TARGET)
    		target_compile_options(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
    		find_package(Threads REQUIRED)
    		target_link_libraries(OpenMP_TARGET INTERFACE Threads::Threads)
    		target_link_libraries(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
	endif()

	target_link_libraries(geogram PUBLIC OpenMP::OpenMP_CXX)
endif()

################################################################################

if(MSVC OR MSYS)
	# remove warning for multiply defined symbols (caused by multiple
	# instantiations of STL templates)
	#target_compile_options(geogram INTERFACE /wd4251)

	# remove all unused stuff from windows.h
	# target_compile_definitions(geogram INTERFACE -DWIN32_LEAN_AND_MEAN)
	# target_compile_definitions(geogram INTERFACE -DVC_EXTRALEAN)

	# do not define a min() and a max() macro, breaks
	# std::min() and std::max() !!
	target_compile_definitions(geogram INTERFACE -DNOMINMAX)

	# we want M_PI etc...
	target_compile_definitions(geogram INTERFACE -D_USE_MATH_DEFINES)
endif()



