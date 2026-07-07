if(NOT CPACK_POLYFEM_APP_PACKAGE_OUTPUT_DIRECTORY)
    message(FATAL_ERROR "CPACK_POLYFEM_APP_PACKAGE_OUTPUT_DIRECTORY is not set")
endif()

file(MAKE_DIRECTORY "${CPACK_POLYFEM_APP_PACKAGE_OUTPUT_DIRECTORY}")

foreach(_polyfem_app_package IN LISTS CPACK_PACKAGE_FILES)
    get_filename_component(_polyfem_app_package_name "${_polyfem_app_package}" NAME)
    set(_polyfem_app_destination "${CPACK_POLYFEM_APP_PACKAGE_OUTPUT_DIRECTORY}/${_polyfem_app_package_name}")

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_polyfem_app_package}"
            "${_polyfem_app_destination}"
        RESULT_VARIABLE _polyfem_app_copy_result
        OUTPUT_VARIABLE _polyfem_app_copy_output
        ERROR_VARIABLE _polyfem_app_copy_error
    )

    if(NOT _polyfem_app_copy_result EQUAL 0)
        message(FATAL_ERROR "Failed to copy package to ${_polyfem_app_destination}: ${_polyfem_app_copy_output}${_polyfem_app_copy_error}")
    endif()

    message(STATUS "Copied package to ${_polyfem_app_destination}")
endforeach()
