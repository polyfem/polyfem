# Find MKL library
# ----------------
#
# Defines the following variables:
#   MKL_LIBRARIES    Path to the MKL libraries to link with
#   MKL_INCLUDE_DIR  Path to the MKL include directory
#
################################################################################

if(DEFINED MKL_LIBRARIES)
	set(MKL_FIND_QUIETLY TRUE)
endif()

if(CMAKE_MINOR_VERSION GREATER 4)
	if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
		set(MKL_LIBS mkl_core guide mkl_intel_lp64 mkl_sequential pthread)
	else()
		set(MKL_LIBS mkl_core guide mkl_intel mkl_sequential pthread)
	endif()
endif()


unset(MKL_LIBRARIES)
foreach(LIB ${MKL_LIBS})
	find_library(TMP
			${LIB}
		PATHS
			$ENV{MKLLIB}
			/opt/intel/mkl/*/lib/em64t
			/opt/intel/mkl/lib/intel64
			/opt/intel/Compiler/*/*/mkl/lib/em64t
			${LIB_INSTALL_DIR})

	if(TMP)
		set(MKL_LIBRARIES ${MKL_LIBRARIES} ${TMP})
	endif()

endforeach()

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
