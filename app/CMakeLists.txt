list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")


include(imgui)
include(tinyfd)

if (APPLE)
    set(BUILD_TYPE MACOSX_BUNDLE)
endif()

if(WIN32)
    add_executable(polyfem_app
        WIN32
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


if(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_FORMAT "UDBZ")
    set(CPACK_DMG_VOLUME_NAME PolyFEM)
    set(CPACK_BUNDLE_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/resources/Info.plist)
    # set(CPACK_BUNDLE_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/resources/MacOSXBundleInfo.plist.in)
    # set(CPACK_OSX_PACKAGE_VERSION "10.13")

    set(CPACK_SYSTEM_NAME "OSX")

    # set(CPACK_DMG_DS_STORE "${CMAKE_CURRENT_SOURCE_DIR}/resources/DS_Store")
elseif(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_SOURCE_GENERATOR "ZIP")
else()
    set(CPACK_GENERATOR "DEB")
    set(CPACK_SOURCE_GENERATOR "TGZ")
endif()

set(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};polyfem_app;Unspecified;/")

set(CPACK_PACKAGE_VENDOR "PolyFEM")
set(CPACK_PACKAGE_NAME "polyfem_app")
set(CPACK_PACKAGE_FILE_NAME "PolyFEM")
set(CPACK_PACKAGE_ICON "${EXTRA_SOURCE}")

set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_CURRENT_SOURCE_DIR}/resources/splash.png")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
set(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")


include(CPack)