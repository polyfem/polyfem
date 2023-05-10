# CUDA Support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "No CUDA support found!")
endif()

function(enable_cuda TARGET)
  # We need to explicitly state that we need all CUDA files in the particle
  # library to be built with -dc as the member functions could be called by
  # other libraries and executables.
  set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    set(CMAKE_CUDA_ARCHITECTURES "native")
  else()
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCH_LIST)
    string(STRIP "${CUDA_ARCH_LIST}" CUDA_ARCH_LIST)
    string(REPLACE " " ";" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  endif()
  set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

  if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET polysolve
                 PROPERTY
                 BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()

  find_package(CUDAToolkit)
  target_link_libraries(${TARGET} PRIVATE CUDA::cudart)
endfunction()