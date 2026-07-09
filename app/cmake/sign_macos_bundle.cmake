set(_polyfem_app_candidates
    "${CMAKE_INSTALL_PREFIX}/polyfem_app.app"
    "${CPACK_TEMPORARY_DIRECTORY}/polyfem_app.app"
    "${CPACK_TEMPORARY_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}/polyfem_app.app"
    "${CPACK_TOPLEVEL_DIRECTORY}/${CPACK_PACKAGE_FILE_NAME}/polyfem_app.app"
    "${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/${CPACK_SYSTEM_NAME}/${CPACK_GENERATOR}/${CPACK_PACKAGE_FILE_NAME}/polyfem_app.app"
)

foreach(_polyfem_app_candidate IN LISTS _polyfem_app_candidates)
    if(EXISTS "${_polyfem_app_candidate}")
        set(POLYFEM_APP_BUNDLE "${_polyfem_app_candidate}")
        break()
    endif()
endforeach()

if(NOT EXISTS "${POLYFEM_APP_BUNDLE}")
    file(GLOB_RECURSE _polyfem_app_bundles
        LIST_DIRECTORIES true
        "${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/*.app")

    foreach(_polyfem_app_bundle IN LISTS _polyfem_app_bundles)
        if(_polyfem_app_bundle MATCHES "/polyfem_app\\.app$")
            set(POLYFEM_APP_BUNDLE "${_polyfem_app_bundle}")
            break()
        endif()
    endforeach()
endif()

if(NOT EXISTS "${POLYFEM_APP_BUNDLE}")
    string(REPLACE ";" "\n  " _polyfem_app_candidates_text "${_polyfem_app_candidates}")
    message(FATAL_ERROR "Cannot find app bundle to sign. Checked:\n  ${_polyfem_app_candidates_text}")
endif()

message(STATUS "Signing app bundle: ${POLYFEM_APP_BUNDLE}")

execute_process(
    COMMAND /usr/bin/xattr -cr "${POLYFEM_APP_BUNDLE}"
    RESULT_VARIABLE polyfem_app_xattr_result
    OUTPUT_VARIABLE polyfem_app_xattr_output
    ERROR_VARIABLE polyfem_app_xattr_error
)
if(NOT polyfem_app_xattr_result EQUAL 0)
    message(FATAL_ERROR "xattr cleanup failed: ${polyfem_app_xattr_output}${polyfem_app_xattr_error}")
endif()

execute_process(
    COMMAND /usr/bin/codesign --force --deep --sign - "${POLYFEM_APP_BUNDLE}"
    RESULT_VARIABLE polyfem_app_codesign_result
    OUTPUT_VARIABLE polyfem_app_codesign_output
    ERROR_VARIABLE polyfem_app_codesign_error
)
if(NOT polyfem_app_codesign_result EQUAL 0)
    message(FATAL_ERROR "codesign failed: ${polyfem_app_codesign_output}${polyfem_app_codesign_error}")
endif()

execute_process(
    COMMAND /usr/bin/codesign --verify --deep --strict --verbose=2 "${POLYFEM_APP_BUNDLE}"
    RESULT_VARIABLE polyfem_app_verify_result
    OUTPUT_VARIABLE polyfem_app_verify_output
    ERROR_VARIABLE polyfem_app_verify_error
)
if(NOT polyfem_app_verify_result EQUAL 0)
    message(FATAL_ERROR "codesign verification failed: ${polyfem_app_verify_output}${polyfem_app_verify_error}")
endif()
