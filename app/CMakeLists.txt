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
install(TARGETS polyfem_app DESTINATION .)

if(NOT (${CMAKE_VERSION} VERSION_LESS "3.6.0"))
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT polyfem_app)
endif()



if(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_FORMAT "UDBZ")
    set(CPACK_DMG_VOLUME_NAME PolyFEM)
    set(CPACK_BUNDLE_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/resources/Info.plist)

    set(CPACK_SYSTEM_NAME "OSX")

    # For DragNDrop generator, you need to explicitly install the README
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../README.md"
            DESTINATION .
            RENAME "README.txt"
            COMPONENT Runtime)

    # Optionally install other documentation files
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE"
            DESTINATION .
            COMPONENT Runtime
            OPTIONAL)

    # DMG background and layout
    set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_CURRENT_SOURCE_DIR}/resources/splash.png")
elseif(WIN32)
    # For Windows, these resource files work as expected
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")
else()
    # For Linux packages, these resource files work as expected
    set(CPACK_GENERATOR "DEB")
    set(CPACK_SOURCE_GENERATOR "TGZ")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
endif()


# Destination paths below are relative to ${CMAKE_INSTALL_PREFIX}
install(TARGETS polyfem_app
    BUNDLE DESTINATION . COMPONENT Runtime
    RUNTIME DESTINATION bin COMPONENT Runtime
    )

# Note Mac specific extension .app
set(APPS "\${CMAKE_INSTALL_PREFIX}/polyfem_app.app")

# Directories to look for dependencies
set(DIRS ${CMAKE_BINARY_DIR})

install(CODE "include(BundleUtilities)
    fixup_bundle(\"${APPS}\" \"\" \"${DIRS}\")")

set(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};polyfem_app;Unspecified;/")

set(CPACK_PACKAGE_VENDOR "PolyFEM")
set(CPACK_PACKAGE_NAME "polyfem_app")
set(CPACK_PACKAGE_FILE_NAME "PolyFEM")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "PolyFEM - A simple C++ and Python finite element library")


include(CPack)