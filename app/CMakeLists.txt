list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")


include(imgui)
include(tinyfd)

if (APPLE)
    set(BUILD_TYPE MACOSX_BUNDLE)
endif()

if(WIN32)
    add_executable(polyfem_app
        app.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/resources/appicon.rc
    )
elseif(APPLE)
    set(MACOSX_BUNDLE_ICON_FILE ico.icns)
    set(app_icon ${CMAKE_CURRENT_SOURCE_DIR}/resources/ico.icns)
    set_source_files_properties(${app_icon} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
    add_executable(polyfem_app
        MACOSX_BUNDLE
        ${app_icon}
        app.cpp
    )
else()
    add_executable(polyfem_app app.cpp)
endif()

target_link_libraries(polyfem_app PRIVATE imgui::imgui polyfem::polyfem polyfem::warnings CLI11::CLI11 tinyfd::tinyfd)



if (APPLE)
    # Build an application bundle on OSX
    set_target_properties(polyfem_app PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/resources/Info.plist)
    set_target_properties(polyfem_app PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME polyfem_app)
    set_target_properties(polyfem_app PROPERTIES MACOSX_BUNDLE_EXECUTABLE_NAME polyfem_app)
    set_target_properties(polyfem_app PROPERTIES MACOSX_BUNDLE_GUI_IDENTIFIER com.im.polyfem)
    set_target_properties(polyfem_app PROPERTIES MACOSX_BUNDLE_ICON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/resources/ico.icns)

    set_target_properties(polyfem_app PROPERTIES MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0")
    set_target_properties(polyfem_app PROPERTIES MACOSX_BUNDLE_BUNDLE_VERSION "1.0.0")
    set_target_properties(polyfem_app PROPERTIES MACOSX_BUNDLE_COPYRIGHT "© 2025, PolyFEM")
endif()




set_target_properties(polyfem_app PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(APPLE)
    set_target_properties(polyfem_app PROPERTIES INSTALL_RPATH "@executable_path/../Frameworks")
elseif(UNIX)
    set_target_properties(polyfem_app PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endif()

if(NOT (${CMAKE_VERSION} VERSION_LESS "3.6.0"))
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT polyfem_app)
endif()


set(POLYFEM_APP_RUNTIME_COMPONENT Runtime)

if(APPLE)
    install(TARGETS polyfem_app
        BUNDLE DESTINATION .
        COMPONENT ${POLYFEM_APP_RUNTIME_COMPONENT})

    install(CODE "
        include(BundleUtilities)
        set(BU_CHMOD_BUNDLE_ITEMS ON)
        fixup_bundle(\"\${CMAKE_INSTALL_PREFIX}/polyfem_app.app\" \"\" \"${CMAKE_BINARY_DIR}\")
        " COMPONENT ${POLYFEM_APP_RUNTIME_COMPONENT})

    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_FORMAT "UDZO")
    set(CPACK_DMG_VOLUME_NAME PolyFEM)
    set(CPACK_BUNDLE_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/resources/Info.plist)
    set(CPACK_SYSTEM_NAME "macOS")
    set(CPACK_DMG_VOLUME_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/ico.icns")
    set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_CURRENT_SOURCE_DIR}/resources/splash.png")
    set(CPACK_DMG_DS_STORE "${CMAKE_CURRENT_SOURCE_DIR}/resources/DS_Store")
    set(CPACK_PRE_BUILD_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sign_macos_bundle.cmake")
    set(POLYFEM_APP_PACKAGE_PLATFORM "macOS")

elseif(WIN32)
    install(TARGETS polyfem_app
        RUNTIME_DEPENDENCY_SET polyfem_app_runtime_deps
        RUNTIME DESTINATION .
        COMPONENT ${POLYFEM_APP_RUNTIME_COMPONENT})

    install(RUNTIME_DEPENDENCY_SET polyfem_app_runtime_deps
        DIRECTORIES "$<TARGET_FILE_DIR:polyfem_app>"
        PRE_EXCLUDE_REGEXES
            "api-ms-"
            "ext-ms-"
            "[Hh][Vv][Ss][Ii][Ff][Ii][Ll][Ee][Tt][Rr][Uu][Ss][Tt]\\.dll"
        POST_EXCLUDE_REGEXES "[Ww][Ii][Nn][Dd][Oo][Ww][Ss]/[Ss]ystem32/.*"
        RUNTIME DESTINATION .
        COMPONENT ${POLYFEM_APP_RUNTIME_COMPONENT})

    set(CPACK_GENERATOR "ZIP")
    set(POLYFEM_APP_PACKAGE_PLATFORM "Windows")

else()
    install(TARGETS polyfem_app
        RUNTIME_DEPENDENCY_SET polyfem_app_runtime_deps
        RUNTIME DESTINATION bin
        COMPONENT ${POLYFEM_APP_RUNTIME_COMPONENT})

    install(RUNTIME_DEPENDENCY_SET polyfem_app_runtime_deps
        DIRECTORIES "$<TARGET_FILE_DIR:polyfem_app>"
        POST_EXCLUDE_REGEXES "^/lib/.*" "^/usr/lib/.*" "^/usr/lib64/.*"
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        COMPONENT ${POLYFEM_APP_RUNTIME_COMPONENT})

    set(CPACK_GENERATOR "TGZ")
    set(POLYFEM_APP_PACKAGE_PLATFORM "Linux")
endif()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../README.md"
    DESTINATION .
    RENAME "README.txt"
    COMPONENT ${POLYFEM_APP_RUNTIME_COMPONENT})

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE"
    DESTINATION .
    COMPONENT ${POLYFEM_APP_RUNTIME_COMPONENT}
    OPTIONAL)

set(CPACK_PACKAGE_VENDOR "PolyFEM")
set(CPACK_PACKAGE_NAME "polyfem_app")
set(CPACK_PACKAGE_FILE_NAME "PolyFEM-${POLYFEM_APP_PACKAGE_PLATFORM}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "PolyFEM - A simple C++ and Python finite element library")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/packages")
set(CPACK_PACKAGING_INSTALL_PREFIX "/")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")
set(CPACK_COMPONENTS_ALL ${POLYFEM_APP_RUNTIME_COMPONENT})
set(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_CURRENT_BINARY_DIR};${PROJECT_NAME};${POLYFEM_APP_RUNTIME_COMPONENT};/")
set(CPACK_VERBATIM_VARIABLES ON)


include(CPack)
