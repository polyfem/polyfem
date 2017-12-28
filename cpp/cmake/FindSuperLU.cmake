
# Superlu lib usually requires linking to a blas library.
# It is up to the user of this module to find a BLAS and link to it.

if (SUPERLU_INCLUDES AND SUPERLU_LIBRARIES)
  set(SUPERLU_FIND_QUIETLY TRUE)
endif (SUPERLU_INCLUDES AND SUPERLU_LIBRARIES)

find_path(SUPERLU_INCLUDES
  NAMES
  supermatrix.h
  PATHS
  $ENV{SUPERLUDIR}
  ${INCLUDE_INSTALL_DIR}
  PATH_SUFFIXES
  superlu
  SRC
)

find_library(SUPERLU_LIBRARIES superlu PATHS $ENV{SUPERLUDIR} ${LIB_INSTALL_DIR} PATH_SUFFIXES lib)

################################################################################

# check version specific macros
include(CheckCSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state()

set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${SUPERLU_INCLUDES})
set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${SUPERLU_LIBRARIES})

# check wether version is new enough >= 4.0
check_c_source_compiles("
typedef int int_t;
#include <supermatrix.h>
#include <slu_util.h>
int main()
{
  SuperLUStat_t stat;
  stat.expansions=8;
  return 0;
}" SUPERLU_MIN_VERSION_4)

# check whether version is at least 4.3
check_c_source_compiles("
#include <slu_ddefs.h>
int main(void)
{
  return SLU_DOUBLE;
}"
SUPERLU_MIN_VERSION_4_3)

# check whether version is at least 5.0
check_c_source_compiles("
typedef int int_t;
#include <supermatrix.h>
#include <slu_util.h>
int main(void)
{
  GlobalLU_t glu;
  return 0;
}"
SUPERLU_MIN_VERSION_5)

cmake_pop_check_state()

if(SUPERLU_MIN_VERSION_5)
	set(SUPERLU_DEFINES "-DSUPERLU_MAJOR_VERSION=5")
elseif(SUPERLU_MIN_VERSION_4_3 OR SUPERLU_MIN_VERSION_4)
	set(SUPERLU_DEFINES "-DSUPERLU_MAJOR_VERSION=4")
else()
	set(SUPERLU_DEFINES "")
endif()

################################################################################

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SUPERLU DEFAULT_MSG
                                  SUPERLU_INCLUDES SUPERLU_LIBRARIES SUPERLU_DEFINES)

mark_as_advanced(SUPERLU_INCLUDES SUPERLU_LIBRARIES SUPERLU_DEFINES)
