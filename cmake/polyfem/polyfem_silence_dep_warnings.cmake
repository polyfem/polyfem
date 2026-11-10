include_guard(GLOBAL)

function(polyfem_path_is_under_any path out_var)
  set(result FALSE)
  foreach(prefix IN LISTS ARGN)
    set(prefix_path "${prefix}")
    cmake_path(IS_PREFIX prefix_path "${path}" NORMALIZE is_under_prefix)
    if(is_under_prefix)
      set(result TRUE)
      break()
    endif()
  endforeach()

  set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

function(polyfem_is_project_target target out_var)
  get_target_property(target_source_dir ${target} SOURCE_DIR)
  get_target_property(target_sources ${target} SOURCES)

  set(is_project_target FALSE)
  if(target_sources)
    foreach(source IN LISTS target_sources)
      if(source MATCHES "^\\$<")
        continue()
      endif()

      if(IS_ABSOLUTE "${source}")
        set(source_path "${source}")
      else()
        set(source_path "${source}")
        cmake_path(ABSOLUTE_PATH source_path
          BASE_DIRECTORY "${target_source_dir}"
          NORMALIZE
        )
      endif()

      polyfem_path_is_under_any("${source_path}" source_is_project_source ${ARGN})
      if(source_is_project_source)
        set(is_project_target TRUE)
        break()
      endif()
    endforeach()
  endif()

  # Some targets are created from a whitelisted subdirectory and receive their
  # sources later. Keep those target directories classified as project-owned.
  if(NOT is_project_target)
    polyfem_path_is_under_any("${target_source_dir}" is_project_target ${ARGN})
  endif()

  set(${out_var} "${is_project_target}" PARENT_SCOPE)
endfunction()

function(polyfem_silence_non_whitelisted_targets dir)
  set(project_source_dirs ${ARGN})
  get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
      get_target_property(target_type ${target} TYPE)
      get_target_property(target_imported ${target} IMPORTED)

      if(NOT target_imported
          AND NOT target_type STREQUAL "INTERFACE_LIBRARY"
          AND NOT target_type STREQUAL "UTILITY")
          polyfem_is_project_target("${target}" is_project_target ${project_source_dirs})
          if(NOT is_project_target)
            target_compile_options(${target} PRIVATE
              $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-w>
              $<$<CXX_COMPILER_ID:MSVC>:/W0>
            )
          endif()
      endif()
  endforeach()

  # Recursively process subdirectories
  get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(subdir IN LISTS subdirs)
      polyfem_silence_non_whitelisted_targets("${subdir}" ${project_source_dirs})
  endforeach()
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL
  polyfem_silence_non_whitelisted_targets
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/app"
  "${PROJECT_SOURCE_DIR}/tests"
)
