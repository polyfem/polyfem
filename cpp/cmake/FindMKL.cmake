# Find MKL library
# ----------------
#
# Defines the following variables:
#   MKL_LIBRARIES    Path to the MKL libraries to link with
#   MKL_INCLUDE_DIR  Path to the MKL include directory
#
################################################################################

if(MKL_LIBRARIES)
	set(MKL_FIND_QUIETLY TRUE)
endif()

if(CMAKE_MINOR_VERSION GREATER 4)
	if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
		find_library(MKL_LIBRARIES
				mkl_core
			PATHS
				$ENV{MKLLIB}
				/opt/intel/mkl/*/lib/em64t
				/opt/intel/mkl/lib/intel64
				/opt/intel/Compiler/*/*/mkl/lib/em64t
				${LIB_INSTALL_DIR})

		find_library(MKL_GUIDE
				guide
			PATHS
				$ENV{MKLLIB}
				/opt/intel/mkl/*/lib/em64t
				/opt/intel/Compiler/*/*/mkl/lib/em64t
				/opt/intel/Compiler/*/*/lib/intel64
				${LIB_INSTALL_DIR})

		if(MKL_LIBRARIES AND MKL_GUIDE)
			set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_lp64 mkl_sequential ${MKL_GUIDE} pthread)
		else()
			set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_lp64 mkl_sequential pthread)
		endif()

	else()
		find_library(MKL_LIBRARIES
				mkl_core
			PATHS
				$ENV{MKLLIB}
				/opt/intel/mkl/*/lib/32
				/opt/intel/Compiler/*/*/mkl/lib/32
				${LIB_INSTALL_DIR})

		find_library(MKL_GUIDE
				guide
			PATHS
				$ENV{MKLLIB}
				/opt/intel/mkl/*/lib/32
				/opt/intel/Compiler/*/*/mkl/lib/32
				/opt/intel/Compiler/*/*/lib/intel32
				${LIB_INSTALL_DIR})

		if(MKL_LIBRARIES AND MKL_GUIDE)
			set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel mkl_sequential ${MKL_GUIDE} pthread)
		else()
			set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel mkl_sequential pthread)
		endif()

	endif()
endif()

find_path(MKL_INCLUDE_DIR
		mkl.h
	PATHS
		$ENV{MKLLIB}
		/opt/intel/mkl/include
		${LIB_INSTALL_DIR}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIR)

mark_as_advanced(MKL_LIBRARIES MKL_INCLUDE_DIR)
